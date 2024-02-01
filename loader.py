# -*- coding: utf-8 -*- #

import math
import random
from copy import deepcopy

import cv2
import numpy as np
import torch
import torch.utils.data as data

import imgproc
from parsers.xml_parser import ParserTRACE  # ours

gaussian_map = np.zeros((201, 201), dtype=np.float32)
gaussian_line = np.zeros((201, 201), dtype=np.float32)
sigma = math.pow(40.0, 2)
for i in range(201):
    for j in range(201):
        gaussian_map[i, j] = math.exp(-(math.pow(i - 100, 2) / 2 / sigma + math.pow(j - 100, 2) / 2 / sigma))
        gaussian_line[i, j] = math.exp(-(math.pow(i - 100, 2) / 2 / sigma))
gaussian_map /= np.max(gaussian_map)
gaussian_line /= np.max(gaussian_line)
gaussian_poly = np.float32([[0, 0], [200, 0], [200, 200], [0, 200]])


def get_heatmap_patch(quad, im_size, kernel, poly=gaussian_poly):
    bbox = np.int32([np.floor(quad.min(axis=0)), np.ceil(quad.max(axis=0))])
    if any(bbox[0] > im_size) or any(bbox[1] < 0):
        return False, None, None
    bbox = np.int32([np.maximum(bbox[0], 0), np.minimum(bbox[1], im_size)])
    if any(bbox[0] >= bbox[1]):
        return False, None, None
    patch_size = tuple(bbox[1] - bbox[0])
    tl = np.minimum(np.maximum(np.floor(quad.min(axis=0)), 0), im_size).astype(np.float32)
    M = cv2.getPerspectiveTransform(poly, quad - tl)
    img_text = cv2.warpPerspective(kernel, M, patch_size)
    return True, img_text, bbox


def GTTransform(target, width, height):
    effective_conf = 0.05
    height = int(height)
    width = int(width)
    heatmap_gt = np.zeros((height, width), dtype=np.float32)
    heatmap_gt_hor = np.zeros((height, width), dtype=np.float32)
    heatmap_gt_ver = np.zeros((height, width), dtype=np.float32)
    heatmap_gt_ihor = np.zeros((height, width), dtype=np.float32)
    heatmap_gt_iver = np.zeros((height, width), dtype=np.float32)
    weight_mask = np.ones((height, width), dtype=np.float32)

    for k, obj in enumerate(target):
        attributes = obj.copy()
        obj, lines = obj["quad"], obj["line"]

        obj = np.array(obj)
        obj *= [width, height] * 4

        obj_poly = obj[:8].astype(np.float32).reshape((4, 2))
        # constant
        bs = 5

        for i in range(4):
            # calc center of box
            center = obj_poly[i]

            # warp gaussian image to magnified character box
            corner_poly = np.float32(
                [
                    [center[0] - bs, center[1] - bs],
                    [center[0] + bs, center[1] - bs],
                    [center[0] + bs, center[1] + bs],
                    [center[0] - bs, center[1] + bs],
                ]
            )
            M = cv2.getPerspectiveTransform(gaussian_poly, corner_poly)
            img_text = cv2.warpPerspective(gaussian_map, M, (width, height))
            heatmap_gt = np.maximum(heatmap_gt, img_text)

        # link representation
        thickness = 2
        # horizontal line
        if lines[0]:  # TOP
            cv2.line(
                heatmap_gt_hor,
                tuple(obj_poly[0].astype(np.int32)),
                tuple(obj_poly[1].astype(np.int32)),
                color=1,
                thickness=thickness,
            )
        else:
            cv2.line(
                heatmap_gt_ihor,
                tuple(obj_poly[0].astype(np.int32)),
                tuple(obj_poly[1].astype(np.int32)),
                color=1,
                thickness=thickness,
            )
        if lines[1]:  # BOTTOM
            cv2.line(
                heatmap_gt_hor,
                tuple(obj_poly[2].astype(np.int32)),
                tuple(obj_poly[3].astype(np.int32)),
                color=1,
                thickness=thickness,
            )
        else:
            cv2.line(
                heatmap_gt_ihor,
                tuple(obj_poly[2].astype(np.int32)),
                tuple(obj_poly[3].astype(np.int32)),
                color=1,
                thickness=thickness,
            )
        # vertical line
        if lines[3]:  # RIGHT
            cv2.line(
                heatmap_gt_ver,
                tuple(obj_poly[1].astype(np.int32)),
                tuple(obj_poly[2].astype(np.int32)),
                color=1,
                thickness=thickness,
            )
        else:
            cv2.line(
                heatmap_gt_iver,
                tuple(obj_poly[1].astype(np.int32)),
                tuple(obj_poly[2].astype(np.int32)),
                color=1,
                thickness=thickness,
            )
        if lines[2]:  # LEFT
            cv2.line(
                heatmap_gt_ver,
                tuple(obj_poly[3].astype(np.int32)),
                tuple(obj_poly[0].astype(np.int32)),
                color=1,
                thickness=thickness,
            )
        else:
            cv2.line(
                heatmap_gt_iver,
                tuple(obj_poly[3].astype(np.int32)),
                tuple(obj_poly[0].astype(np.int32)),
                color=1,
                thickness=thickness,
            )

    # clipping in heatmap
    heatmap_gt[np.where(heatmap_gt < effective_conf)] = 0

    # finalize gt
    heatmap_gt = np.concatenate(
        [
            heatmap_gt[..., np.newaxis],
            heatmap_gt_hor[..., np.newaxis],
            heatmap_gt_ver[..., np.newaxis],
        ],
        axis=-1,
    )
    heatmap_gt = np.concatenate(
        [
            heatmap_gt,
            heatmap_gt_ihor[..., np.newaxis],
            heatmap_gt_iver[..., np.newaxis],
        ],
        axis=-1,
    )

    return heatmap_gt, weight_mask


class TRACE_Dataset(data.Dataset):
    """OCR Dataset Object for TextAffinityField

    input is image, target is annotation

    Arguments:
        rootpath (string): filepath to OCR folder
        datasets (string): datasets name (paths to dataset are already defined in db_params.py)
        phase (string): set 'test' or 'train' phase (default is 'train')
        transform (callable, optional): transformation to perform on the input image
        gt_transform (callable, optional): transformation to perform on the GT annotation
    """

    def __init__(
        self,
        datasets,
        rootpath,
        scale_down=2,
        out_type="HEATMAP",
        phase="train",
        transform=None,
        mixratio=[1],
    ):
        self.rootpath = rootpath
        self.datasets = datasets
        self.mixratio = mixratio
        self.scale_down = scale_down
        self.out_type = out_type
        self.phase = phase
        self.transform = transform
        self.parsers = []
        self.dataset_size = 0

        for dataset in datasets.split(","):
            parser = ParserTRACE(rootpath, dataset, phase)
            self.dataset_size += parser.lenFiles()
            self.parsers.append({"name": dataset, "num": parser.lenFiles(), "parser": parser})
        print("Dataset size of Training Sets : {:d}".format(self.dataset_size))

        cv2.setNumThreads(0)  # prevent deadlock caused by conflict with pytorch

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, index):
        img, gt, weight = self.pull_item(index)
        return img, gt, weight

    def pull_item(self, index):
        mix_rand = random.randrange(0, sum(self.mixratio))
        parser_ind = 0
        parser_ind_sum = self.mixratio[0]
        while 1:
            if mix_rand < parser_ind_sum:
                break
            parser_ind += 1
            parser_ind_sum += self.mixratio[parser_ind]
        parser = self.parsers[parser_ind]["parser"]

        while 1:
            try:
                img_file, gt = parser.parseGT()
                if isinstance(img_file, str):
                    img = imgproc.loadImage(img_file)
                else:
                    img = img_file  # Some parser return image rather than image file name
                height, width, channels = img.shape
            except Exception as e:
                print(e)
                continue

            if gt is None:
                gt = []

            gt, gt_lines = gt["quads"], gt["lines"]
            gt = np.array(gt, dtype=np.float32).reshape(-1, 8)
            break

        # normalize GT coordinate
        num_pt = int(gt.shape[1] / 2)
        gt[:, : 2 * num_pt] /= [width, height] * num_pt

        # Transformation
        if self.transform is not None:
            img, gt, gt_lines, _ = self.transform(img, gt, gt_lines)
            width = height = self.transform.size

        # Create TextAffinityField GT Map
        gt_gathered = []
        for attr, quad in zip(gt_lines, gt):
            gt_gathered.append({"quad": quad, "line": attr})

        # GT transform
        width /= self.scale_down
        height /= self.scale_down
        gt_image, gt_weight = GTTransform(gt_gathered, width, height)
        _, _, gt_ch = gt_image.shape
        gt_weight = np.array([gt_weight] * gt_ch).transpose(1, 2, 0)

        # Preprocessing for pre-trained model
        img = imgproc.normalizeMeanVariance(img)

        return (
            torch.from_numpy(img.astype(np.float32)).permute(2, 0, 1),
            torch.from_numpy(gt_image.astype(np.float32)),
            torch.from_numpy(gt_weight.astype(np.float32)),
        )
