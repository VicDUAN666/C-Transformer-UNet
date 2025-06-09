import argparse
import logging
import os
import random
import sys
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from tqdm import tqdm
from datasets.dataset_crevasse import Crevasse_dataset  # <-- Use Crevasse_dataset
from utils import test_single_volume
from networks.vit_seg_modeling import VisionTransformer as ViT_seg
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg

parser = argparse.ArgumentParser()

args = parser.parse_args()
# --- Updated argument for VOC data path ---
parser.add_argument('--data_path', type=str,
                    default='./VOCdevkit/VOC2007', help='root dir for VOC formatted data')
parser.add_argument('--dataset', type=str,
                    default='Crevasse', help='experiment_name')
parser.add_argument('--num_classes', type=int,
                    default=2, help='output channel of network (background + crevasse)')
parser.add_argument('--list_dir', type=str,
                    default='./lists/lists_Crevasse', help='list dir for crevasse data')
parser.add_argument('--max_epochs', type=int,
                    default=350, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int,
                    default=16, help='batch_size per gpu')
parser.add_argument('--img_size', type=int,
                    default=512, help='input patch size of network input')
parser.add_argument('--is_savenii', action="store_true", help='whether to save results during inference')
parser.add_argument('--n_skip', type=int,
                    default=3, help='using number of skip-connect, default is 3')
parser.add_argument('--vit_name', type=str,
                    default='C-TransUNet', help='select C-TransUNet model')
parser.add_argument('--test_save_dir', type=str,
                    default='../predictions', help='saving prediction as nii!')
parser.add_argument('--deterministic', type=int,
                    default=1, help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,
                    default=0.001, help='segmentation network learning rate')
parser.add_argument('--seed', type=int,
                    default=1234, help='random seed')
parser.add_argument('--vit_patches_size', type=int,
                    default=16, help='vit_patches_size, default is 16')
args = parser.parse_args()


def inference(args, model, test_save_path=None):
    # --- Updated to use VOCCrevasseDataset and test_single_image ---
    db_test = VOCCrevasseDataset(data_path=args.data_path, txt_name="test.txt")
    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=1)

    logging.info("{} test iterations per epoch".format(len(testloader)))
    model.eval()
    metric_list = 0.0

    for i_batch, sampled_batch in tqdm(enumerate(testloader)):
        image, label, case_name = sampled_batch["image"], sampled_batch["label"], sampled_batch['case_name'][0]

        metric_i = test_single_image(image, label, model, patch_size=[args.img_size, args.img_size],
                                     test_save_path=test_save_path, case=case_name)
        metric_list += np.array(metric_i)
        logging.info('idx %d case %s mean_dice %f mean_hd95 %f' % (
        i_batch, case_name, np.mean(metric_i, axis=0)[0], np.mean(metric_i, axis=0)[1]))

    metric_list = metric_list / len(db_test)
    logging.info('Mean class 1 (Crevasse) mean_dice %f mean_hd95 %f' % (metric_list[0][0], metric_list[0][1]))
    performance = np.mean(metric_list, axis=0)[0]
    mean_hd95 = np.mean(metric_list, axis=0)[1]
    logging.info('Testing performance: mean_dice : %f mean_hd95 : %f' % (performance, mean_hd95))
    return "Testing Finished!"


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

    dataset_config = {
        'Crevasse': {
            'Dataset': VOCCrevasseDataset,  # <-- Point to the new class
            'data_path': args.data_path,
            'num_classes': 2,
        },
    }

    dataset_name = args.dataset
    args.num_classes = dataset_config[dataset_name]['num_classes']
    args.data_path = dataset_config[dataset_name]['data_path']
    args.Dataset = dataset_config[dataset_name]['Dataset']
    args.list_dir = dataset_config[dataset_name]['list_dir']
    args.z_spacing = dataset_config[dataset_name]['z_spacing']
    args.is_pretrain = True

    # --- Construct snapshot path exactly as in train.py ---
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

    config_vit = CONFIGS_ViT_seg[args.vit_name]
    config_vit.n_classes = 1  # The output channel is 1 for binary segmentation logits
    config_vit.n_skip = args.n_skip
    config_vit.patches.size = (args.vit_patches_size, args.vit_patches_size)

    if args.vit_name.find('R50') != -1 or args.vit_name.find('C-TransUNet') != -1:
        config_vit.patches.grid = (
        int(args.img_size / args.vit_patches_size), int(args.img_size / args.vit_patches_size))

    net = ViT_seg(config_vit, img_size=args.img_size, num_classes=config_vit.n_classes).cuda()

    # Load the trained model
    snapshot = os.path.join(snapshot_path, 'epoch_' + str(args.max_epochs - 1) + '.pth')
    if not os.path.exists(snapshot):
        snapshot = os.path.join(snapshot_path, 'best_model.pth')  # Fallback to a potential best model
        if not os.path.exists(snapshot):
            print(f"Error: Snapshot not found at {snapshot}")
            sys.exit(1)

    net.load_state_dict(torch.load(snapshot))
    snapshot_name = snapshot_path.split('/')[-1]

    log_folder = './test_log/test_log_' + args.exp
    os.makedirs(log_folder, exist_ok=True)
    logging.basicConfig(filename=log_folder + '/' + snapshot_name + ".txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    logging.info(snapshot_name)

    if args.is_savenii:
        args.test_save_dir = os.path.join(args.test_save_dir, args.exp, snapshot_name)
        os.makedirs(args.test_save_dir, exist_ok=True)
        test_save_path = args.test_save_dir
    else:
        test_save_path = None

    inference(args, net, test_save_path)