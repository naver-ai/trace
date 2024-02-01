# -*- coding: utf-8 -*-
import random

import cv2
import numpy as np
import torch


class Compose(object):
    """Composes several augmentations together.
    Args:
        transforms (List[Transform]): list of transforms to compose.
    Example:
        >>> augmentations.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, boxes=None, labels=None, angles=None):
        for t in self.transforms:
            img, boxes, labels, angles = t(img, boxes, labels, angles)
        return img, boxes, labels, angles

class ConvertFromInts(object):
    def __call__(self, image, boxes=None, labels=None, angles=None):
        return image.astype(np.float32), boxes, labels, angles


class SubtractMeans(object):
    def __init__(self, mean):
        self.mean = np.array(mean, dtype=np.float32)

    def __call__(self, image, boxes=None, labels=None, angles=None):
        image = image.astype(np.float32)
        image -= self.mean
        return image.astype(np.float32), boxes, labels, angles


class ToAbsoluteCoords(object):
    def __call__(self, image, boxes=None, labels=None, angles=None):
        height, width, channels = image.shape
        num_pt = int(boxes.shape[1] / 2)
        boxes *= [width, height] * num_pt

        return image, boxes, labels, angles


class ToPercentCoords(object):
    def __call__(self, image, boxes=None, labels=None, angles=None):
        height, width, channels = image.shape
        num_pt = int(boxes.shape[1] / 2)
        boxes /= [width, height] * num_pt

        return image, boxes, labels, angles


class Resize(object):
    def __init__(self, size=300):
        self.size = size
        self.resize_option = (
            cv2.INTER_LINEAR,
            cv2.INTER_NEAREST,
            cv2.INTER_AREA,
            cv2.INTER_CUBIC,
            cv2.INTER_LANCZOS4,
        )

    def __call__(self, image, boxes=None, labels=None, angles=None):
        inter_mode = random.choice(self.resize_option)
        image = cv2.resize(image.astype(np.uint8), (self.size, self.size), interpolation=inter_mode)
        return image, boxes, labels, angles


class RandomPerspective(object):
    def __init__(self, dist_scale=0.1, p=0.5, interp=cv2.INTER_AREA):
        self.dist_scale = dist_scale
        self.p = p
        self.interp = interp

    def get_random_transform(self, w, h):
        hw, hh = w // 2, h // 2
        dw, dh = int(hw * self.dist_scale), int(hh * self.dist_scale)
        tl = [random.randint(-dw, dw), random.randint(-dh, dh)]
        tr = [random.randint(w - dw, w + dw - 1), random.randint(-dh, dh)]
        br = [random.randint(w - dw, w + dw - 1), random.randint(h - dh, h + dh - 1)]
        bl = [random.randint(-dw, dw), random.randint(h - dh, h + dh - 1)]
        pt_src = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)
        pt_dst = np.array([tl, tr, br, bl], dtype=np.float32)
        return cv2.getPerspectiveTransform(pt_src, pt_dst)

    def __call__(self, image, boxes, labels=None, angles=None):
        if random.random() < self.p:
            h, w, _ = image.shape
            H = self.get_random_transform(w, h)
            im_w = cv2.warpPerspective(image, H, (w, h))
            if len(boxes) == 0:
                return im_w, boxes, labels, angles
            pt_w = cv2.transform(boxes.reshape(1, -1, 2), H)
            boxes_w = (pt_w[:, :, :2] / pt_w[:, :, 2:]).reshape([-1, 8])
            return im_w, boxes_w, labels, angles
        return image, boxes, labels, angles
        pass


class RandomSaturation(object):
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    def __call__(self, image, boxes=None, labels=None, angles=None):
        if random.randrange(2):
            image[:, :, 1] *= random.uniform(self.lower, self.upper)
            image[image > 255] = 255
            image[image < 0] = 0
        return image, boxes, labels, angles


class RandomHue(object):
    def __init__(self, delta=36.0):
        assert delta >= 0.0 and delta <= 360.0
        self.delta = delta

    def __call__(self, image, boxes=None, labels=None, angles=None):
        if random.randrange(2):
            image[:, :, 0] += random.uniform(-self.delta, self.delta)
            image[:, :, 0][image[:, :, 0] > 360.0] -= 360.0
            image[:, :, 0][image[:, :, 0] < 0.0] += 360.0
        return image, boxes, labels, angles


class RandomLightingNoise(object):
    def __init__(self):
        self.perms = ((0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0))

    def __call__(self, image, boxes=None, labels=None, angles=None):
        if random.randrange(2):
            swap = self.perms[random.randrange(len(self.perms))]
            shuffle = SwapChannels(swap)  # shuffle channels
            image = shuffle(image)
        return image, boxes, labels, angles


class ConvertColor(object):
    def __init__(self, current="BGR", transform="HSV"):
        self.transform = transform
        self.current = current

    def __call__(self, image, boxes=None, labels=None, angles=None):
        if self.current == "BGR" and self.transform == "HSV":
            image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        elif self.current == "HSV" and self.transform == "BGR":
            image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
        else:
            raise NotImplementedError
        return image, boxes, labels, angles


class RandomContrast(object):
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    # expects float image
    def __call__(self, image, boxes=None, labels=None, angles=None):
        if random.randrange(2):
            alpha = random.uniform(self.lower, self.upper)
            image *= alpha
            image[image > 255] = 255
            image[image < 0] = 0
        return image, boxes, labels, angles


class RandomBrightness(object):
    def __init__(self, delta=16):
        assert delta >= 0.0
        assert delta <= 255.0
        self.delta = delta

    def __call__(self, image, boxes=None, labels=None, angles=None):
        if random.randrange(2):
            delta = random.uniform(-self.delta, self.delta)
            image += delta
            image[image > 255] = 255
            image[image < 0] = 0
        return image, boxes, labels, angles


class ToCV2Image(object):
    def __call__(self, tensor, boxes=None, labels=None):
        return (
            tensor.cpu().numpy().astype(np.float32).transpose((1, 2, 0)),
            boxes,
            labels,
        )


class ToTensor(object):
    def __call__(self, cvimage, boxes=None, labels=None):
        return (
            torch.from_numpy(cvimage.astype(np.float32)).permute(2, 0, 1),
            boxes,
            labels,
        )


class RandomResizeCrop(object):
    def __init__(self, sizes, min_ratio=0.5, max_ratio=2.0):
        self.sizes = sizes
        self.max_size = max(sizes)
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio
        self.inter = cv2.INTER_LINEAR_EXACT

    def __call__(self, image, boxes, labels=None, angles=None):
        size = random.choice(self.sizes)
        h, w, c = image.shape
        max_dim = max(h, w)
        ratio = size / max_dim
        if ratio < self.min_ratio:
            oh = round(h * self.min_ratio)
            ow = round(w * self.min_ratio)
        elif ratio > self.max_ratio:
            oh = round(h * self.max_ratio)
            ow = round(w * self.max_ratio)
        else:
            oh = round(h * ratio)
            ow = round(w * ratio)
        ratio = max(oh, ow) / max_dim
        res = np.zeros([self.max_size, self.max_size, c], dtype=image.dtype)
        if oh >= self.max_size and ow >= self.max_size:
            top = random.randint(0, oh - self.max_size)
            left = random.randint(0, ow - self.max_size)
            res = cv2.resize(image, (ow, oh), interpolation=self.inter)[
                top : top + self.max_size, left : left + self.max_size, :
            ]
            if len(boxes) > 0:
                boxes *= ratio
                boxes -= [left, top] * int(len(boxes[0]) / 2)
        elif oh < self.max_size and ow < self.max_size:
            res[:oh, :ow, :] = cv2.resize(image, (ow, oh), interpolation=self.inter)
            if len(boxes) > 0:
                boxes *= ratio
        elif oh < self.max_size:
            left = random.randint(0, ow - self.max_size)
            res[:oh, : self.max_size, :] = cv2.resize(image, (ow, oh), interpolation=self.inter)[
                :, left : left + self.max_size, :
            ]
            if len(boxes) > 0:
                boxes *= ratio
                boxes -= [left, 0] * int(len(boxes[0]) / 2)
        else:
            top = random.randint(0, oh - self.max_size)
            res[: self.max_size, :ow, :] = cv2.resize(image, (ow, oh), interpolation=self.inter)[
                top : top + self.max_size, :, :
            ]
            if len(boxes) > 0:
                boxes *= ratio
                boxes -= [0, top] * int(len(boxes[0]) / 2)
        return res, boxes, labels, angles



class PhotometricDistort(object):
    def __init__(self):
        self.pd = [
            RandomContrast(),
            ConvertColor(transform="HSV"),
            # RandomSaturation(),
            RandomHue(),
            ConvertColor(current="HSV", transform="BGR"),
            RandomContrast(),
        ]
        self.rand_brightness = RandomBrightness()
        self.rand_light_noise = RandomLightingNoise()

    def __call__(self, image, boxes, labels, angles):
        im = image.copy()
        im, boxes, labels, angles = self.rand_brightness(im, boxes, labels, angles)
        if random.randrange(2):
            distort = Compose(self.pd[:-1])
        else:
            distort = Compose(self.pd[1:])
        im, boxes, labels, angles = distort(im, boxes, labels, angles)
        # im, boxes, labels = self.rand_light_noise(im, boxes, labels)
        im[im > 255] = 255
        im[im < 0] = 0
        return im, boxes, labels, angles


class TRACEAugmentation(object):
    def __init__(self, size=512, mean=(104, 117, 123)):
        self.mean = mean
        self.size = size
        self.augment = Compose(
            [
                ConvertFromInts(),
                ToAbsoluteCoords(),
                PhotometricDistort(),
                RandomPerspective(),
                RandomResizeCrop(list(range(1024, self.size + 1, 32))),
                ToPercentCoords(),
                Resize(self.size),
            ]
        )

    def __call__(self, img, boxes, labels, angles=None):
        return self.augment(img, boxes, labels, angles)
