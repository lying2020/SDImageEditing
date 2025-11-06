# Fix Qt/OpenCV GUI issues for headless environments
import os
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '0'

from tqdm import tqdm
from einops import rearrange
import PIL, time, json, datetime
import random
import numpy as np
from PIL import Image
from copy import deepcopy
from torchvision.utils import save_image
from typing import List, Optional, Union
import argparse, torch, inspect

from torch import autocast
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.optim.lr_scheduler import CosineAnnealingLR
import torchvision.transforms as T
from torchvision import utils as vutils
from torchvision.utils import draw_bounding_boxes
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torchvision import datasets, transforms

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print(sys.path)
import project as prj

from utils.lr_schedule import WarmupLinearLRSchedule
from utils.vis import *
from utils.util2 import compose_text_with_templates, get_augmentations_template
from utils.util import EditingJsonDataset, plot_images
from utils.engine import *
import utils.misc as misc
from models.model import RGN
from models.utils import visualize_images, read_image_from_url, draw_image_with_bbox_new, Bbox



def get_args_parser():
    parser = argparse.ArgumentParser(description="train models")

    # ============================================================================
    # Experiment Name and Basic Configuration
    # ============================================================================
    parser.add_argument('--run_name', type=str, default="exp",
                       help='Experiment run name for identifying different experiments')
    parser.add_argument("--nodes", default=1, type=int,
                       help='Number of nodes (machines) for distributed training, set to 1 for single machine')

    # ============================================================================
    # File Path Parameters
    # ============================================================================
    parser.add_argument('--image_dir_path', type=str, default=prj.IMAGES_DIR,
                       help='Directory path containing input images, used for batch processing')
    parser.add_argument('--json_file', type=str, default=prj.IMAGES_JSON_FILE,
                       help='Path to JSON configuration file containing original descriptions and editing prompts for each image')
    parser.add_argument('--output_dir', type=str, default=prj.OUTPUT_DIR,
                       help='Output directory to save edited images')
    parser.add_argument('--save_path', type=str, default=prj.CHECKPOINTS_DIR,
                       help='Path to save model checkpoints during training')
    parser.add_argument('--load_checkpoint_path', type=str, default=None,
                       help='Path to load existing checkpoint for resuming training or inference')

    # ============================================================================
    # Model Related Parameters
    # ============================================================================
    parser.add_argument('--diffusion_model_path', type=str, default=prj.INPAINTING_MODEL_PATH,
                       help='Path to diffusion model, can be HuggingFace model ID or local path. '
                            'Recommended: stabilityai/stable-diffusion-2-inpainting')
    parser.add_argument('--image_size', type=int, default=256,
                       help='Input image height and width in pixels, images will be resized to this size')
    parser.add_argument('--device', type=str, default="cuda",
                       help='Training device, use "cuda" for GPU or "cpu" for CPU')

    # ============================================================================
    # Training Core Parameters (affect editing quality and effectiveness)
    # ============================================================================
    parser.add_argument('--lr', type=float, default=5e-3,
                       help='Learning rate controlling the step size for model parameter updates. '
                            'Larger values train faster but may be unstable, smaller values train slower but more stable. '
                            'Recommended range: 1e-3 to 1e-2')
    parser.add_argument('--epochs', type=int, default=10,
                       help='Number of training epochs, how many times to iterate through the entire dataset. '
                            'For image editing tasks, usually 1-2 epochs are sufficient')
    parser.add_argument('--per_image_iteration', type=int, default=5,
                       help='Number of training iterations per image, controlling how many optimization iterations per image. '
                            'Larger values improve editing quality but take longer. Recommended: 5-20')
    parser.add_argument('--accum_grad', type=int, default=25,
                       help='Gradient accumulation steps, used to simulate large batch effects when batch_size is small. '
                            'Effective batch_size = batch_size * accum_grad')

    # ============================================================================
    # Sampling Parameters (affect editing region localization accuracy)
    # ============================================================================
    parser.add_argument('--max_window_size', type=int, default=10,
                       help='Maximum bounding box size in pixels, controlling the maximum range of editing region. '
                            'Larger values allow editing larger regions but increase computation. Recommended: 10-20')
    parser.add_argument('--point_number', type=int, default=9,
                       help='Number of sampled anchor points for locating editing regions. '
                            'Larger values improve localization accuracy but increase computation. Recommended: 5-12')

    # ============================================================================
    # Loss Function Parameters (control training objective weights)
    # ============================================================================
    parser.add_argument('--loss_alpha', type=int, default=1,
                       help='CLIP loss weight coefficient, controlling text-image similarity loss weight')
    parser.add_argument('--loss_beta', type=int, default=1,
                       help='Directional CLIP loss weight coefficient, controlling editing direction loss weight')
    parser.add_argument('--loss_gamma', type=int, default=1,
                       help='Structure loss weight coefficient, controlling original image structure preservation loss weight')
    parser.add_argument('--test_alpha', type=int, default=2,
                       help='Text-image similarity weight coefficient during testing')
    parser.add_argument('--test_beta', type=int, default=1,
                       help='Image-image similarity weight coefficient during testing')

    # ============================================================================
    # System Performance Parameters (affect training speed and resource usage)
    # ============================================================================
    parser.add_argument('--batch_size', type=int, default=1,
                       help='Batch size, number of images processed at once. '
                            'For image editing tasks, batch_size=1 usually works best, but can be increased for speed. '
                            'Note: GPU memory usage is proportional to batch_size')
    parser.add_argument('--num_workers', default=16, type=int,
                       help='Number of data loading processes for parallel data loading. '
                            'Recommended: half to full number of CPU cores, too many may cause high memory usage')
    parser.add_argument('--pin_mem', action='store_true', default=True,
                       help='Pin memory, keep data in pinned memory to speed up GPU transfer. '
                            'Can improve data loading speed by 10-20%% but requires more memory')

    # ============================================================================
    # Visualization Parameters
    # ============================================================================
    parser.add_argument('--draw_box', action='store_true', default=True,
                       help='Whether to draw bounding boxes of editing regions for visualization. '
                            'When enabled, creates a "boxes" subdirectory in output_dir to save visualization results')

    # ============================================================================
    # Random Seed and Other Configuration
    # ============================================================================
    parser.add_argument('--seed', default=42, type=int,
                       help='Random seed for controlling random number generation to ensure reproducibility. '
                            'Same seed + same parameters = same results')
    parser.add_argument('--ckpt_interval', type=int, default=5,
                       help='Checkpoint saving interval, save model every N epochs')

    # ============================================================================
    # Distributed Training Parameters (multi-GPU/multi-machine training)
    # ============================================================================
    parser.add_argument('--distributed', action='store_true', default=False,
                       help='Whether to run distributed training, default is False')
    parser.add_argument('--world_size', default=1, type=int,
                       help='Total number of distributed processes, usually equals number_of_nodes × GPUs_per_node')
    parser.add_argument('--local_rank', default=0, type=int,
                       help='Local process rank, automatically set by torchrun, no need to specify manually')
    parser.add_argument('--dist_on_itp', action='store_true', default=False,
                       help='Whether to run distributed training on ITP (Inter-Thread Parallelism)')
    parser.add_argument('--dist_url', default='env://',
                       help='Distributed training initialization URL for setting up distributed training environment. '
                            'Usually use "env://" to read from environment variables')

    args = parser.parse_args()

    return args

def configure_optimizers(model, lr, betas=(0.9, 0.96), weight_decay=4.5e-2, use_distributed=False):
    # Handle both distributed and non-distributed models
    if use_distributed:
        optimizer = torch.optim.Adam(model.module.anchor_net.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
    else:
        optimizer = torch.optim.Adam(model.anchor_net.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
    return optimizer

def train(args, lr_schedule, model, template, len_train_dataset, data_loader_train, optim, device_id, use_distributed=False):
    save_path = args.save_path
    if use_distributed:
        rank = dist.get_rank()
    else:
        rank = 0

    if not os.path.exists(save_path) and rank == 0:
        os.mkdir(save_path)

    for epoch in range(1, args.epochs+1):
        if use_distributed and hasattr(data_loader_train.sampler, 'set_epoch'):
            data_loader_train.sampler.set_epoch(epoch)
        metric_logger = misc.MetricLogger(delimiter="  ")
        metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
        if rank == 0:
            print(f'Epoch {epoch}:')
        for data_iter_step, (imgs, o_prompt, e_prompt) in enumerate(tqdm(data_loader_train)):
            lr_schedule.step()
            imgs = imgs.to(device=device_id, non_blocking=True)
            o_prompt, e_prompt = o_prompt[0], e_prompt[0]
            e_prompt = compose_text_with_templates(e_prompt, template)
            with torch.amp.autocast('cuda'):  # Use new API instead of deprecated torch.cuda.amp.autocast
                if use_distributed:
                    bboxs = torch.ceil(map_cooridates(model.module.get_anchor_box(imgs)))
                    imgs_new, mask_imgs = get_mask_imgs(imgs, bboxs)
                    results = model.module.generate_result(imgs_new.to(device_id), mask_imgs.to(device_id), e_prompt).to(device_id)
                    loss, loss_clip, loss_cip_dir, loss_structure = model.module.get_loss(imgs_new, results, e_prompt, o_prompt)
                else:
                    bboxs = torch.ceil(map_cooridates(model.get_anchor_box(imgs)))
                    imgs_new, mask_imgs = get_mask_imgs(imgs, bboxs)
                    results = model.generate_result(imgs_new.to(device_id), mask_imgs.to(device_id), e_prompt).to(device_id)
                    loss, loss_clip, loss_cip_dir, loss_structure = model.get_loss(imgs_new, results, e_prompt, o_prompt)
            loss.backward()
            if data_iter_step % args.accum_grad == 0:
                optim.step()
                optim.zero_grad()
            metric_logger.update(loss=loss.item())

        if rank == 0:
            if epoch % args.ckpt_interval == 0:
                if use_distributed:
                    torch.save(model.state_dict(), os.path.join(save_path, f'transformer_epoch_{epoch}.pth'))
                else:
                    torch.save(model.state_dict(), os.path.join(save_path, f'transformer_epoch_{epoch}.pth'))
            if use_distributed:
                torch.save(model.state_dict(), os.path.join(save_path, 'last.pth'))
            else:
                torch.save(model.state_dict(), os.path.join(save_path, 'last.pth'))

    return model

def main(args):
    # Check if running in distributed mode
    # Use environment variables as the source of truth, but also check args.distributed
    use_distributed = ('RANK' in os.environ and 'WORLD_SIZE' in os.environ) or args.distributed

    if use_distributed:
        print("Running in distributed mode")
        # Distributed training mode
        if not dist.is_initialized():
            dist.init_process_group("nccl", init_method='env://')
        rank = dist.get_rank()
        device_id = rank % torch.cuda.device_count()
        num_tasks = misc.get_world_size()
    else:
        # Single GPU mode (non-distributed)
        print("Running in single GPU mode (non-distributed)")
        rank = 0
        device_id = 0
        num_tasks = 1

    device = torch.device(args.device)

    if use_distributed:
        seed = args.seed + misc.get_rank()
    else:
        seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    template = get_augmentations_template()

    if not os.path.exists(args.save_path) and rank == 0:
        os.mkdir(args.save_path)

    model = RGN(image_size=args.image_size, device=device_id, args=args).to(device_id)
    if use_distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)

    if rank == 0 and not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    dirs = os.listdir(args.output_dir)

    train_dataset = EditingJsonDataset(args, args.per_image_iteration)
    test_dataset = EditingJsonDataset(args)

    len_train_dataset = len(train_dataset)

    if use_distributed:
        sampler = torch.utils.data.DistributedSampler(
            train_dataset, num_replicas=num_tasks, rank=rank, shuffle=False, drop_last=False
        )
        data_loader_train = torch.utils.data.DataLoader(
            train_dataset, sampler=sampler,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False,
            shuffle=False)
    else:
        data_loader_train = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False,
            shuffle=False)

    data_loader_test = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=args.pin_mem)

    optim = configure_optimizers(model, args.lr, use_distributed=use_distributed)
    total_steps = len_train_dataset / (args.batch_size * num_tasks)
    lr_schedule = CosineAnnealingLR(optim, T_max=args.epochs*total_steps)
    optim.zero_grad()
    model = train(args, lr_schedule, model, template, len_train_dataset, data_loader_train, optim, device_id, use_distributed=use_distributed)
    if rank == 0:
        print('Generating edited images!')
        model.eval()
        predict(args, model, template, data_loader_test, device_id, use_distributed=use_distributed)



if __name__ == '__main__':
    args = get_args_parser()
    main(args)
