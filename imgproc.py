# -*- coding: utf-8 -*-
import cv2
import numpy as np


def loadImage(img_file):
    img = cv2.imread(img_file)  # BGR order
    img = img[:, :, ::-1]  # to RGB
    if img.shape[0] == 2:
        img = img[0]
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    if img.shape[2] == 4:
        img = img[:, :, :3]
    img = np.array(img)

    return img


def normalizeMeanVariance(in_img, mean=(0.485, 0.456, 0.406), variance=(0.229, 0.224, 0.225)):
    # should be RGB order
    img = in_img.copy().astype(np.float32)
    img /= 255.0
    img -= mean
    img /= variance
    return img


def denormalizeMeanVariance(in_img, mean=(0.485, 0.456, 0.406), variance=(0.229, 0.224, 0.225)):
    # should be RGB order
    img = in_img.copy()
    img *= variance
    img += mean
    img *= 255.0
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img


def resize_aspect_ratio(img, square_size, mag_ratio=1, scale_down=2):
    height, width, channel = img.shape
    interpolation = cv2.INTER_LINEAR
    if square_size < max(width, height):
        interpolation = cv2.INTER_AREA

    # magnify image size
    target_size = mag_ratio * max(height, width)

    # set original image size
    if target_size > square_size:
        target_size = square_size

    ratio = target_size / max(height, width)

    target_h, target_w = int(height * ratio), int(width * ratio)
    if ratio == 1:
        proc = np.copy(img)
    else:
        proc = cv2.resize(img, (target_w, target_h), interpolation=interpolation)

    step_size = 320  # compatibility with TensorRT
    min_size = 960
    target_h_new, target_w_new = target_h, target_w
    if target_h % step_size != 0:
        target_h_new = target_h + (step_size - target_h % step_size)
    if target_w % step_size != 0:
        target_w_new = target_w + (step_size - target_w % step_size)

    target_h_new = max(min_size, target_h_new)
    target_w_new = max(min_size, target_w_new)
    pad_x, pad_y = (target_w_new - target_w) // 2, (target_h_new - target_h) // 2
    img_resized = cv2.copyMakeBorder(
        proc,
        pad_y,
        target_h_new - target_h - pad_y,
        pad_x,
        target_w_new - target_w - pad_x,
        cv2.BORDER_REPLICATE,
    )

    # calculate image ratio for coordinate transformation
    mask_area = np.array([pad_x, pad_y, target_w_new - pad_x, target_h_new - pad_y]) / scale_down
    size_heatmap = (int(target_w_new / scale_down), int(target_h_new / scale_down))
    ratio = [
        ratio / scale_down,
        ratio / scale_down,
        pad_x / ratio,
        pad_y / ratio,
    ] + mask_area.astype(np.int32).tolist()

    return img_resized, ratio, size_heatmap
