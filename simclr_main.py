import os
import torch
import torch.nn as nn

from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import torch.multiprocessing as mp

import numpy as np
from tqdm import tqdm
import wandb
import time
from utils.data import gen_loaders
from utils.meters import AverageMeter, accuracy
from utils.utils import precision_scheduler, WarmUpLR
from utils.plot import Tensor_Dict
from utils.utils import replace_layers, info_nce_loss

import args
import argparse
from utils.logging import print_current_opt, create_checkpoint
import random

from simclr_data_aug.contrastive_learning_dataset import ContrastiveLearningDataset
from model.resnet_simclr import ResNetSimCLR

from torch.cuda.amp import GradScaler, autocast

# Dataset class dict
dataset_class = {
    'cifar10' : 10,
    'cifar100' : 100,
    'ImageNet100' : 100,
    'ILSVRC2012': 1000,
}

tensor_dict = Tensor_Dict()

def forward_hook(module, input, output):
    tensor_dict.register('weight', module.weight, direction='forward')
    tensor_dict.register('input', input[0], direction='forward')

def backward_hook(module, grad_input, grad_output):
    tensor_dict.register('grad_out', grad_output[0], direction='backward')

def layer_wag_save_init(model):
    for name, layer in model.named_modules():
        if isinstance(layer, HQ_Conv2d) or isinstance(layer, HQ_Linear):
            tensor_dict.key_init(name)
            layer.register_forward_hook(forward_hook)
            layer.register_full_backward_hook(backward_hook)

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_args_parser():
    parser = argparse.ArgumentParser('HLQ training and evaluation script', add_help=False)
    
    parser.add_argument('--GPU_USE', default='4', type=str)
    parser.add_argument('--RUN_NAME', default='', type=str)
    parser.add_argument('--DEBUG_MODE', type=str2bool, default=False, help='False when train')
    parser.add_argument('--MODEL', type=str, help='Q_efficientformer_l1, Q_efficientformerv2_l, Q_swinv2_b, Q_swinv2_l')
    parser.add_argument('--DATASET', type=str, help='cifar10, cifar100, ImageNet100, ILSVRC2012, voc')
    parser.add_argument('--AMP', type=str2bool, default=False, help='False when HLQ used')
    parser.add_argument('--EPOCHS', default=200, type=int)
    parser.add_argument('--BATCH_SIZE', default=128, type=int)
    parser.add_argument('--LR', default=0.1, type=float,
                        help='0.1 for resnet, 1e-3 for Eformer, 1e-1 for EformerV2, 0.256 for EfNet, 0.001 for EfNet_pt, 0.0003 for simclr, 5e-04 for swinv2, 5e-5 for segformer')
    parser.add_argument('--WORKERS', default=8, type=int)
    parser.add_argument('--DATA_DIR', type=str, default='/SSD')
    parser.add_argument('--CKPT_DIR', type=str, default='./checkpoint') 
    parser.add_argument('--SEED', default=2023, type=int)

    parser.add_argument('--precisionScheduling', type=str2bool, default=False, help='Enable for quant scheme is stoch or int')
    parser.add_argument('--milestone', default='50', type=str)
    parser.add_argument('--GogiQuantBit', default=4, type=int)
    parser.add_argument('--weightQuantBit', default=4, type=int)
    parser.add_argument('--GogwQuantBit', default=4, type=int)
    parser.add_argument('--actQuantBit', default=4, type=int)
    parser.add_argument('--eps', type=float, default=1e-8)

    parser.add_argument('--quantBWDGogi', default='no', type=str, help='int, stoch, no, luq')
    parser.add_argument('--quantBWDWgt', default='no', type=str, help='int, stoch, no, luq')
    parser.add_argument('--quantBWDGogw', default='no', type=str, help='int, stoch, no, luq')
    parser.add_argument('--quantBWDAct', default='no', type=str, help='int, stoch, no, luq')

    parser.add_argument('--vectorPercentile', default=50, type=int)

    parser.add_argument('--transform_scheme', default='gih_gwlr', type=str, help='hadamard, low_rank, gih_gwlr(for matmul), gih_gwlrh(kernel), gilro_gwFP')

    parser.add_argument('--TransformGogi', type=str2bool, default=False)
    parser.add_argument('--TransformWgt', type=str2bool, default=False)
    parser.add_argument('--TransformGogw', type=str2bool, default=False)
    parser.add_argument('--TransformInput', type=str2bool, default=False)


    parser.add_argument('--KERNEL', type=str2bool, default=True)
    parser.add_argument('--DISTRIBUTED', type=str2bool, default=False)

    parser.add_argument('--wagSaveForPlot', type=str2bool, default=False)
    parser.add_argument('--wagSave_DIR', default='./pickle', type=str)
    parser.add_argument('--wagMilestone', default='0,50,100,150,199', type=str)
    
    return parser

def allocate_args(parsed_args):
    for name, _ in vars(parsed_args).items(): 
        setattr(args, name, getattr(parsed_args, name))

def main_worker(rank, run, parsed_args):    
    
    # allocate parsed_args to args
    allocate_args(parsed_args)
    if parsed_args.KERNEL == False:
        from utils.quant_matmul import HQ_Conv2d, HQ_Linear 
    else: 
        from utils.quant import HQ_Conv2d, HQ_Linear
    GPU_NUM = parsed_args.GPU_NUM
        
    if args.DISTRIBUTED:
        # DDP setup
        torch.cuda.set_device(rank)
        dist.init_process_group(backend='nccl', init_method='tcp://127.0.0.1:1111'+str(args.GPU_USE[0]), world_size=GPU_NUM, rank=rank)

    if args.MODEL.endswith('simclr'):
        model = ResNetSimCLR(base_model='resnet50', out_dim=128)
    else:
        raise NotImplementedError

    if args.MODEL.startswith('Q'):
        replace_layers(model, nn.Conv2d, HQ_Conv2d)
        replace_layers(model, nn.Linear, HQ_Linear)
    
    model.to('cuda')

    criterion = nn.CrossEntropyLoss().to('cuda')

    if args.wagSaveForPlot:
        layer_wag_save_init(model)

    # DDP sync
    if args.DISTRIBUTED:
        model = DDP(model, device_ids=[rank], output_device=0, find_unused_parameters=True)
        dist.barrier()

    dataset = ContrastiveLearningDataset(args.DATA_DIR)
    train_dataset = dataset.get_dataset(args.DATASET, 2, isTrain=True)
    test_dataset = dataset.get_dataset(args.DATASET, 2, isTrain=False)
    
    if args.DISTRIBUTED:
        train_sampler = torch.utils.data.DistributedSampler(train_dataset, shuffle=True)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, sampler=train_sampler, batch_size=int(args.BATCH_SIZE // GPU_NUM), shuffle=True,
            num_workers=args.WORKERS, pin_memory=True, drop_last=True)
        test_sampler = torch.utils.data.DistributedSampler(test_dataset, shuffle=True)
        test_loader = torch.utils.data.DataLoader(
            test_dataset, sampler=test_sampler, batch_size=int(args.BATCH_SIZE // GPU_NUM), shuffle=True,
            num_workers=args.WORKERS, pin_memory=True, drop_last=True)
    else:
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.BATCH_SIZE, shuffle=True,
            num_workers=args.WORKERS, pin_memory=True, drop_last=True)
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=args.BATCH_SIZE, shuffle=True,
            num_workers=args.WORKERS, pin_memory=True, drop_last=True)
        
    
    optimizer = torch.optim.Adam(model.parameters(), args.LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader), eta_min=0, last_epoch=-1)

    scaler = GradScaler(enabled=True)
    best_acc = 0
    for epoch in range(args.EPOCHS):
            
        train_loss, train_prec1, _= forward(
            epoch, train_loader, model, scheduler, scaler, criterion, optimizer, training=True)
        
        if rank == 0:
            with torch.no_grad():
                val_loss, val_prec1, _= forward(
                    epoch, test_loader, model, scheduler, scaler, criterion, optimizer, training=False)
            
            print('Epoch: {0} '
                        'Train Prec@1 {train_prec1:.3f} '
                        'Train Loss {train_loss:.3f} '
                        'Valid Prec@1 {val_prec1:.3f} '
                        'Valid Loss {val_loss:.3f} \n'
                        .format(epoch,
                                train_prec1=train_prec1, val_prec1=val_prec1,
                                train_loss=train_loss, val_loss=val_loss))
            
            # wandb recording
            run.log({
                'train acc':train_prec1,
                'valid acc':val_prec1,
                'train loss':train_loss,
                'valid loss':val_loss
            })

            if val_prec1 > best_acc and not args.DEBUG_MODE:
                best_acc = max(val_prec1, best_acc)
                create_checkpoint(model=model, optimizer=optimizer, lr_scheduler=scheduler, epoch=epoch, ckpt_dir=args.CKPT_DIR, run_name=args.RUN_NAME)
            
            # if args.wagSaveForPlot and epoch in list(map(int, args.wagMilestone.split(','))):
            if args.wagSaveForPlot:
                tensor_dict.pickle_save(epoch=epoch, wagsave_dir=args.wagSave_DIR, run_name=args.RUN_NAME)
                tensor_dict.value_clear()
        
    if args.DISTRIBUTED:
        # DDP clean
        dist.destroy_process_group()


def forward(epoch, data_loader, model, scheduler, scaler, criterion, optimizer, training):
    if training:
        model.train()
    else:
        model.eval()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    end = time.time()

    for images, _ in tqdm(data_loader):
        data_time.update(time.time() - end)
        images = torch.cat(images, dim=0).to('cuda')

        with autocast(enabled=args.AMP):
            features = model(images)
            logits, labels = info_nce_loss(features, batch_size=args.BATCH_SIZE, n_views=2)
            loss = criterion(logits, labels)
        
        # measure accuracy and record loss
        losses.update(loss, images.size(0))
        prec1, prec5 = accuracy(logits.detach(), labels, topk=(1, 5))
        top1.update(prec1, images.size(0))
        top5.update(prec5, images.size(0))

        optimizer.zero_grad()

        if training and args.AMP:            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if epoch >= 10:
                scheduler.step()
        elif training and not args.AMP:
            loss.backward()
            optimizer.step()

            if epoch >= 10:
                scheduler.step()
        
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    return losses.avg, top1.avg, top5.avg

if __name__ == '__main__':
    # args parsing
    parser = argparse.ArgumentParser('HLQ training and evaluation script', parents=[get_args_parser()])
    parsed_args = parser.parse_args()

    parsed_args.KERNEL = False if parsed_args.MODEL == 'HQ_EfficientNet_s' or \
                    parsed_args.MODEL == 'HQ_EfficientNet_m' or \
                    parsed_args.transform_scheme == "low_rank" or \
                    parsed_args.transform_scheme == "gih_gwlr"\
                else True
    
    # seed setting
    seed = parsed_args.SEED
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # GPU setting
    os.environ["CUDA_VISIBLE_DEVICES"] = parsed_args.GPU_USE
    parsed_args.GPU_NUM = len(os.environ["CUDA_VISIBLE_DEVICES"].split(','))
    parsed_args.DISTRIBUTED = True if parsed_args.GPU_NUM > 1 else False

    # wandb initialization
    run = wandb.init(project='Hadamard_Quant')
    run.name = parsed_args.RUN_NAME
    run.save

    # Logging
    print_current_opt(parsed_args)

    # Spawn multiprocess for each GPUs
    if parsed_args.DISTRIBUTED:
        try: 
            mp.spawn(main_worker, args=(run, parsed_args), nprocs=parsed_args.GPU_NUM)
        except KeyboardInterrupt:
            print("Keyboard interrupted")
            dist.destroy_process_group()
    else:
        main_worker(0, run, parsed_args)