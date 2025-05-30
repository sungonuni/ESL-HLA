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
from utils.utils import precision_scheduler, replace_layers
from utils.plot import Tensor_Dict
from model.resnet import ResNet18, ResNet34, ResNet50
from utils.quant_matmul import HQ_Conv2d, HQ_Linear, cluster_quantizer, stoch_quantizer, sawb_quantizer, luq_quantizer 

import args
import argparse
from utils.logging import print_current_opt, create_checkpoint
import random

import timm
from torch.cuda.amp import GradScaler, autocast
from torchvision import models
from byol_pytorch import BYOL

# Dataset class dict
dataset_class = {
    'cifar10' : 10,
    'cifar100' : 100,
    'ImageNet100' : 100,
    'ILSVRC2012': 1000,
}

quantScheme_gx = ["stoch_quantizer", "norm_quantizer"]
quantScheme_gw = ["stoch_quantizer", "norm_quantizer"]

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
    parser.add_argument('--GPU_NUM', default='', type=str)
    parser.add_argument('--RUN_NAME', default='', type=str)
    parser.add_argument('--DEBUG_MODE', type=str2bool, default=False, help='False when train')
    parser.add_argument('--MODEL', type=str, help='Q_resnet18, Q_resnet34, Q_efficientnetv2_s, Q_resnet50, Q_efficientformer_l1, Q_efficientformerv2_l, Q_swinv2_b, Q_swinv2_l ')
    parser.add_argument('--PRETRAINED', type=str2bool, help='imagenet pretrained')
    parser.add_argument('--CONTINUE', type=str2bool, default=False, help='continue training from ckpt file')
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

    parser.add_argument('--LoRA', type=str2bool, default=False) # Not for resnet, only for capability
    parser.add_argument('--LoRA_all', type=str2bool, default=False) # Not for resnet, only for capability
    parser.add_argument('--HLQ_on_base', type=str2bool, default=False) # Not for resnet, only for capability
    parser.add_argument('--HLQ_on_decompose', type=str2bool, default=False) # Not for resnet, only for capability

    parser.add_argument('--precisionScheduling', type=str2bool, default=False, help='Enable for quant scheme is stoch or int')
    parser.add_argument('--milestone', default='50', type=str)
    parser.add_argument('--GogiQuantBit', default=4, type=int)
    parser.add_argument('--weightQuantBit', default=4, type=int)
    parser.add_argument('--GogwQuantBit', default=4, type=int)
    parser.add_argument('--actQuantBit', default=4, type=int)
    parser.add_argument('--eps', type=float, default=1e-11)

    parser.add_argument('--quantAuto', default=False, type=str2bool, help='Auto quant sheme')
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

    parser.add_argument('--DISTRIBUTED', type=str2bool, default=False)

    parser.add_argument('--wagSaveForPlot', type=str2bool, default=False)
    parser.add_argument('--wagSave_DIR', default='./pickle', type=str)
    parser.add_argument('--wagMilestone', default='0,50,100,150,199', type=str)
    
    return parser

def allocate_args(parsed_args):
    for name, _ in vars(parsed_args).items(): 
        setattr(args, name, getattr(parsed_args, name))

def allocate_args_for_calibration(parsed_args):
    for name, _ in vars(parsed_args).items():
        if name.startswith('quant') or name.startswith('Transform'):
            continue
        setattr(args, name, getattr(parsed_args, name))


def main_worker(rank, run, parsed_args):    
    
    # allocate parsed_args to args
    allocate_args(parsed_args)
    GPU_NUM = parsed_args.GPU_NUM
    
    # DDP setup
    if args.DISTRIBUTED:
        # DDP setup
        torch.cuda.set_device(rank)
        dist.init_process_group(backend='nccl', init_method='tcp://127.0.0.1:1111'+str(args.GPU_USE[0]), world_size=GPU_NUM, rank=rank)

    if args.MODEL.endswith('resnet34'):
        # model = timm.create_model('resnet34', pretrained=parsed_args.PRETRAINED, num_classes=dataset_class[args.DATASET])
        model = models.resnet34(pretrained=True)
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, dataset_class[args.DATASET])
    elif args.MODEL.endswith('resnet50'):
        # model = timm.create_model('resnet50', pretrained=parsed_args.PRETRAINED, num_classes=dataset_class[args.DATASET])
        model = models.resnet50(pretrained=True)
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, dataset_class[args.DATASET])
    else:
        raise NotImplementedError

    model_name_parsed = args.MODEL.split("_")

    if model_name_parsed[0] == 'Q':
        replace_layers(model, nn.Conv2d, HQ_Conv2d)
        replace_layers(model, nn.Linear, HQ_Linear)
        for name, layer in model.named_modules():
            if isinstance(layer, HQ_Conv2d) or isinstance(layer, HQ_Linear):
                layer.layer_name = name
        model_name_parsed.pop(0)
    

    model.to('cuda')

    learner = None
    if model_name_parsed[0] == 'byol':
        learner = BYOL(
            model,
            image_size = 224,
            hidden_layer = 'avgpool'
        )

    criterion = nn.CrossEntropyLoss()
    criterion.to('cuda')

    if args.wagSaveForPlot:
        layer_wag_save_init(model)

    # DDP sync
    if args.DISTRIBUTED:
        model = DDP(model, device_ids=[rank], output_device=0, find_unused_parameters=True)
        dist.barrier()

    optimizer = torch.optim.SGD(model.parameters(), lr=args.LR, momentum=0.9, weight_decay=5e-4)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 120, 160], gamma=0.2)
    # scheduler = CosineAnnealingWarmUpRestarts(optimizer, T_0=50, T_mult=1, eta_max=args.LR, T_up=5, gamma=0.5)

    if args.CONTINUE:
        ckpt_name = args.RUN_NAME + ".ckpt"
        ckpt = torch.load(os.path.join(args.CKPT_DIR, ckpt_name))
        for key in list(ckpt['model'].keys()):
            if 'module.' in key:
                ckpt['model'][key.replace('module.', '')] = ckpt['model'][key]
                del ckpt['model'][key]
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['lr_scheduler'])

    train_loader, test_loader = gen_loaders(args.DATA_DIR, args.BATCH_SIZE, args.WORKERS, args.DATASET, DDP=args.DISTRIBUTED, GPU_NUM=GPU_NUM)

    scaler = GradScaler(enabled=True)
    best_acc = 0
    for epoch in range(args.EPOCHS):
        
        if args.precisionScheduling:
            precision_scheduler(current_epoch=epoch, max_bit=8, min_bit=4, milestones=list(map(int, args.milestone.split(','))))
        
        print(f"{args.GogiQuantBit}, {args.weightQuantBit}, {args.GogwQuantBit}, {args.actQuantBit}")
        print(f"{args.quantBWDGogi}, {args.quantBWDWgt}, {args.quantBWDGogw}, {args.quantBWDAct}")

        train_loss, train_prec1, train_prec5= forward(
            epoch, scaler, train_loader, model, learner, criterion, optimizer, training=True)
        
        if rank == 0:
            with torch.no_grad():
                val_loss, val_prec1, val_prec5= forward(
                    epoch, scaler, test_loader, model, learner, criterion, optimizer, training=False)
            
            scheduler.step()

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


def forward(epoch, scaler, data_loader, model, learner, criterion, optimizer, training):
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

    for i, data in enumerate(tqdm(data_loader)):
        # measure data loading time
        data_time.update(time.time() - end)
        inputs, target = data
        inputs = inputs.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        optimizer.zero_grad()

        output = model(inputs)
        if learner is not None:
            loss = learner(inputs)
        else:
            loss = criterion(output, target)

        # measure accuracy and record loss
        losses.update(loss, inputs.size(0))
        prec1, prec5 = accuracy(output.detach(), target, topk=(1, 5))
        top1.update(prec1, inputs.size(0))
        top5.update(prec5, inputs.size(0))

        if training:
            loss.backward()
            optimizer.step()
            if learner is not None:
                learner.update_moving_average()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    return losses.avg, top1.avg, top5.avg

if __name__ == '__main__':
    # args parsing
    parser = argparse.ArgumentParser('HLQ training and evaluation script', parents=[get_args_parser()])
    parsed_args = parser.parse_args()
    
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