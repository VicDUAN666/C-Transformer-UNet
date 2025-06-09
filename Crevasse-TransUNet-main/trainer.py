import argparse
import logging
import os
import random
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import BCEDiceLoss  # Import the new loss function
from torchvision import transforms


def trainer_crevasse(args, model, snapshot_path):
    from datasets.dataset_crevasse import Crevasse_dataset, RandomGenerator

    # ... (logging setup remains the same)

    # --- Instantiate the new Dataset ---
    db_train = VOCCrevasseDataset(data_path=args.data_path, txt_name="train.txt",
                                  transform=transforms.Compose(
                                      [RandomGenerator(output_size=[args.img_size, args.img_size])]))
    print("The length of train set is: {}".format(len(db_train)))

    # ... (DataLoader, model setup, optimizer, scheduler, etc. remain the same as the previous modification)
    # ... (The rest of the training loop logic is also correct for the new data format)

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True,
                             worker_init_fn=worker_init_fn)

    if args.n_gpu > 1:
        model = nn.DataParallel(model)
    model.train()

    # C-TransUNet Innovation: Use hybrid loss, AdamW optimizer, and Cosine Annealing scheduler
    # Loss function with optimized weights alpha=0.5, beta=1
    loss_fn = BCEDiceLoss(alpha=0.5, beta=1.0)

    # AdamW optimizer
    optimizer = optim.AdamW(model.parameters(), lr=base_lr, weight_decay=0.0001)

    # Cosine Annealing learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epochs, eta_min=1e-6)

    writer = SummaryWriter(snapshot_path + '/log')
    iter_num = 0
    max_epoch = args.max_epochs
    logging.info(f"{len(trainloader)} iterations per epoch. {max_epoch} max epochs.")

    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            image_batch, label_batch = image_batch.cuda(), label_batch.cuda()

            # The model output is a single channel logit for binary segmentation
            outputs = model(image_batch).squeeze(1)
            label_batch = label_batch.float()

            loss = loss_fn(outputs, label_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            iter_num = iter_num + 1
            writer.add_scalar('info/total_loss', loss.item(), iter_num)
            writer.add_scalar('info/lr', optimizer.param_groups[0]['lr'], iter_num)

            logging.info('iteration %d : loss : %f' % (iter_num, loss.item()))

            if iter_num % 20 == 0:
                # Visualize the first channel of the image batch
                image = image_batch[1, 0:1, :, :]
                image = (image - image.min()) / (image.max() - image.min())
                writer.add_image('train/Image', image, iter_num)

                # Visualize prediction
                preds = torch.sigmoid(outputs) > 0.5
                writer.add_image('train/Prediction', preds[1, ...].unsqueeze(0) * 255, iter_num)

                # Visualize ground truth
                writer.add_image('train/GroundTruth', label_batch[1, ...].unsqueeze(0) * 255, iter_num)

        # Update learning rate scheduler at the end of each epoch
        scheduler.step()

        save_interval = 50
        if (epoch_num + 1) % save_interval == 0:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))

        if epoch_num >= max_epoch - 1:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))
            iterator.close()
            break

    writer.close()
    return "Training Finished!"