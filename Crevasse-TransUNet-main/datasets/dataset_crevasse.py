import os
import random
import numpy as np
import torch
from PIL import Image
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset


# --- Augmentation functions remain the same ---
def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


def random_rotate(image, label):
    angle = np.random.randint(-30, 30)
    image = ndimage.rotate(image, angle, order=3, reshape=False, cval=0)  # Use cval to fill background
    label = ndimage.rotate(label, angle, order=0, reshape=False, cval=0)
    return image, label


def random_gaussian_noise(image):
    noise = np.random.normal(0, 0.02, image.shape)
    noisy_image = image + noise
    noisy_image = np.clip(noisy_image, 0, 1)
    return noisy_image


def random_gamma_correction(image):
    gamma = random.uniform(0.7, 1.5)
    corrected_image = np.power(image, gamma)
    return corrected_image


class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        # Apply augmentations
        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        if random.random() > 0.5:
            image, label = random_rotate(image, label)
        if random.random() > 0.5:
            image = random_gaussian_noise(image)
        if random.random() > 0.5:
            image = random_gamma_correction(image)

        # Resize
        x, y, _ = image.shape
        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y, 1), order=3)
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)

        # Transpose from (H, W, C) to (C, H, W) and convert to tensor
        image = torch.from_numpy(image.transpose((2, 0, 1)).astype(np.float32))
        label = torch.from_numpy(label.astype(np.float32))

        sample = {'image': image, 'label': label.long()}
        return sample


# --- New Dataset Class for VOC format ---
class VOCCrevasseDataset(Dataset):
    def __init__(self, data_path, txt_name="train.txt", transform=None):
        super(VOCCrevasseDataset, self).__init__()
        self.data_path = data_path
        self.txt_path = os.path.join(data_path, "ImageSets", "Segmentation", txt_name)
        self.transform = transform

        with open(self.txt_path, 'r') as f:
            self.file_list = [line.strip() for line in f.readlines()]

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_name = self.file_list[idx]

        # Construct paths
        image_path = os.path.join(self.data_path, "JPEGImages", file_name + ".jpg")
        label_path = os.path.join(self.data_path, "SegmentationClass", file_name + ".png")

        # Load image and label
        image = Image.open(image_path).convert('RGB')
        label = Image.open(label_path)  # Should be single channel (L mode)

        # Convert to numpy arrays
        image_np = np.array(image, np.float32) / 255.0  # Normalize to [0, 1]
        label_np = np.array(label, np.uint8)

        sample = {'image': image_np, 'label': label_np, 'case_name': file_name}

        if self.transform:
            sample = self.transform(sample)

        return sample