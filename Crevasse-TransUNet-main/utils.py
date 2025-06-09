import numpy as np
import torch
from medpy import metric
from scipy.ndimage import zoom
import torch.nn as nn
import SimpleITK as sitk
from PIL import Image


# BCEDiceLoss class remains the same as before

class BCEDiceLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=1.0):
        super(BCEDiceLoss, self).__init__()
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.alpha = alpha
        self.beta = beta

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        return 1 - loss

    def forward(self, inputs, target):
        bce_loss = self.bce_loss(inputs, target.float())
        probs = torch.sigmoid(inputs)
        dice_loss = self._dice_loss(probs, target)
        total_loss = self.alpha * bce_loss + self.beta * dice_loss
        return total_loss


def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0 and gt.sum() > 0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        return dice, hd95
    elif pred.sum() > 0 and gt.sum() == 0:
        return 1, 0
    else:
        return 0, 0


# --- New test function for single 2D images ---
def test_single_image(image, label, net, patch_size=[512, 512], test_save_path=None, case=None):
    # image and label are torch tensors from the dataloader
    image, label = image.squeeze(0), label.squeeze(0)

    # Get original image size
    c, h, w = image.shape

    # Resize if necessary
    if h != patch_size[0] or w != patch_size[1]:
        image_resized = torch.nn.functional.interpolate(image.unsqueeze(0), size=patch_size, mode='bilinear',
                                                        align_corners=False)
    else:
        image_resized = image.unsqueeze(0)

    net.eval()
    with torch.no_grad():
        outputs = net(image_resized.cuda())
        out = torch.sigmoid(outputs).squeeze(0)
        out = (out > 0.5).cpu()  # Binary prediction

    # Resize prediction back to original size
    if h != patch_size[0] or w != patch_size[1]:
        prediction = torch.nn.functional.interpolate(out.float(), size=(h, w), mode='nearest').squeeze(0)
    else:
        prediction = out.squeeze(0)

    prediction_np = prediction.numpy()
    label_np = label.numpy()

    metric_list = []
    metric_list.append(calculate_metric_percase(prediction_np, label_np))

    if test_save_path is not None:
        # Save original image (convert tensor to PIL Image)
        img_to_save = image.permute(1, 2, 0).numpy() * 255
        img_to_save = Image.fromarray(img_to_save.astype(np.uint8))
        img_to_save.save(os.path.join(test_save_path, case + "_img.jpg"))

        # Save prediction
        prd_to_save = Image.fromarray((prediction_np * 255).astype(np.uint8))
        prd_to_save.save(os.path.join(test_save_path, case + "_pred.png"))

        # Save ground truth
        lab_to_save = Image.fromarray((label_np * 255).astype(np.uint8))
        lab_to_save.save(os.path.join(test_save_path, case + "_gt.png"))

    return metric_list