import os
from PIL import Image

import torch
from torch import nn
from torchvision import transforms as T
from tqdm import tqdm

from tqdm import tqdm
from deeplabv3plus import network, utils
import os
import argparse

from deeplabv3plus.datasets import VOCSegmentation, Cityscapes
from deeplabv3plus.network import modeling
from torchvision import transforms as T

import torch
import torch.nn as nn

from PIL import Image
from glob import glob

from .datasets import Cityscapes


def load_deeplabv3plus_cityscapes_trained(ckpt: str = None, model_name="deeplabv3plus_mobilenet") -> nn.Module:
    num_classes = 19
    # decode_fn = Cityscapes.decode_target
    # os.environ['CUDA_VISIBLE_DEVICES'] = 0
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # print("Device: %s" % device)

    # Set up model (all models are 'constructed at network.modeling)

    model = modeling.__dict__[model_name](num_classes=num_classes, output_stride=16)
    separable_conv = False
    if separable_conv and 'plus' in model_name:
        network.convert_to_separable_conv(model.classifier)
    utils.set_bn_momentum(model.backbone, momentum=0.01)

    if ckpt is None:
        if model_name == "deeplabv3plus_mobilenet":
            ckpt = "/home/lukas/git/DeepLabV3Plus-Pytorch/checkpoints/best_deeplabv3plus_mobilenet_cityscapes_os16.pth"
        else:
            raise Exception(f"a checkpoint must be provided for model {model_name}")

    checkpoint = torch.load(ckpt, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint["model_state"])
    # model = nn.DataParallel(model)
    # model.to(device)
    print("Resume model from %s" % ckpt)
    del checkpoint
    return model


def predict(image_files, model, save_val_results_to=None):
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])

    os.environ['CUDA_VISIBLE_DEVICES'] = 0
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device: %s" % device)

    decode_fn = Cityscapes.decode_target

    with torch.no_grad():
        model = model.eval()
        for img_path in tqdm(image_files):
            ext = os.path.basename(img_path).split('.')[-1]
            img_name = os.path.basename(img_path)[:-len(ext) - 1]
            img = Image.open(img_path).convert('RGB')
            img = transform(img).unsqueeze(0)  # To tensor of NCHW
            img = img.to(device)

            pred = model(img).max(1)[1].cpu().numpy()[0]  # HW
            colorized_preds = decode_fn(pred).astype('uint8')
            colorized_preds = Image.fromarray(colorized_preds)
            if save_val_results_to:
                colorized_preds.save(os.path.join(save_val_results_to, img_name + '.png'))
