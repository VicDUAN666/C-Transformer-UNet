import argparse
import logging
import os
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from networks.vit_seg_modeling import VisionTransformer as ViT_seg
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from trainer import trainer_crevasse  # <-- 改为 trainer_crevasse

parser = argparse.ArgumentParser()
# --- Updated Arguments for Crevasse Segmentation ---
parser.add_argument('--root_path', type=str,
                    default='../data/Crevasse/train_npz', help='root dir for training data')
parser.add_argument('--dataset', type=str,
                    default='Crevasse', help='experiment_name')
parser.add_argument('--list_dir', type=str,
                    default='./lists/lists_Crevasse', help='list dir for crevasse data')
parser.add_argument('--num_classes', type=int,
                    default=2, help='output channel of network (background + crevasse)')
parser.add_argument('--max_epochs', type=int,
                    default=350, help='maximum epoch number to train, matching paper')
parser.add_argument('--batch_size', type=int,
                    default=16, help='batch_size per gpu')  # Adjusted for potentially larger image size
parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')
parser.add_argument('--deterministic', type=int, default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float, default=0.001,
                    help='segmentation network learning rate')
parser.add_argument('--img_size', type=int,
                    default=512, help='input patch size of network input, matching paper')
parser.add_argument('--seed', type=int,
                    default=1234, help='random seed')
parser.add_argument('--n_skip', type=int,
                    default=3, help='using number of skip-connect, default is 3')
parser.add_argument('--vit_name', type=str,
                    default='C-TransUNet', help='select C-TransUNet model')
parser.add_argument('--vit_patches_size', type=int,
                    default=16, help='vit_patches_size, default is 16')
args = parser.parse_args()

if __name__ == "__main__":
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    dataset_name = args.dataset
    dataset_config = {
        'Crevasse': {
            'data_path': args.data_path,
            'num_classes': 2,
        },
    }

    if args.dataset not in dataset_config:
        raise ValueError(f"Dataset '{args.dataset}' not configured.")

    args.num_classes = dataset_config[dataset_name]['num_classes']
    args.root_path = dataset_config[dataset_name]['root_path']
    args.list_dir = dataset_config[dataset_name]['list_dir']
    args.is_pretrain = True

    # --- Snapshot path generation ---
    args.exp = 'C-TransUNet_' + dataset_name + '_is' + str(args.img_size)
    snapshot_path = "../model/{}/{}".format(args.exp, 'C-TransUNet')
    snapshot_path = snapshot_path + '_pretrain' if args.is_pretrain else snapshot_path
    snapshot_path += '_' + args.vit_name
    snapshot_path = snapshot_path + '_skip' + str(args.n_skip)
    snapshot_path = snapshot_path + '_vitpatch' + str(
        args.vit_patches_size) if args.vit_patches_size != 16 else snapshot_path
    snapshot_path = snapshot_path + '_epo' + str(args.max_epochs)
    snapshot_path = snapshot_path + '_bs' + str(args.batch_size)
    snapshot_path = snapshot_path + '_lr' + str(args.base_lr)
    snapshot_path = snapshot_path + '_s' + str(args.seed)

    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)

    config_vit = CONFIGS_ViT_seg[args.vit_name]
    config_vit.n_classes = args.num_classes
    config_vit.n_skip = args.n_skip

    if args.vit_name.find('R50') != -1 or args.vit_name.find('C-TransUNet') != -1:
        config_vit.patches.grid = (
        int(args.img_size / args.vit_patches_size), int(args.img_size / args.vit_patches_size))

    net = ViT_seg(config_vit, img_size=args.img_size, num_classes=config_vit.n_classes).cuda()

    # Load pre-trained weights
    if args.is_pretrain:
        net.load_from(weights=np.load(config_vit.pretrained_path))

    trainer = {'Crevasse': trainer_crevasse}
    trainer[dataset_name](args, net, snapshot_path)