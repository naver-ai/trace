# -*- coding: utf-8 -*-
from copy import deepcopy

import cv2
import numpy as np
from shapely.geometry import Polygon, box


#########################################################################################
# Aux functions
#########################################################################################
def warpCoord(Hinv, pt):
    res = Hinv.dot(pt.T).T
    return res[:2] / res[2]


def get_projected_corner_points(corner_map):
    ver_pts, hor_pts = list(), list()

    # State [Nofound = 0, Increase = 1, Decrease = 2]
    # Point should be added after decrease state

    # vertical pts
    ver_bin = np.sum(corner_map, axis=1)
    acc_pt, num_pt, prev_bin, state = 0, 0, 0, 0
    for i in range(len(ver_bin)):
        cur_bin = ver_bin[i]
        add_flag = False
        if cur_bin > 5:
            if state == 0:  # Nofound
                state = 1  # MAKE Increase
            elif state == 1:  # Increase
                if cur_bin < prev_bin:  # MAKE Decrease
                    state = 2
            elif state == 2:  # Decrease
                if cur_bin > prev_bin:
                    add_flag = True
            if not add_flag:
                acc_pt += i * cur_bin
                num_pt += cur_bin
                prev_bin = cur_bin
        elif state > 0:
            add_flag = True

        if add_flag:
            state = 0
            ver_pts.append(int(acc_pt / num_pt + 0.5))
            acc_pt, num_pt, prev_bin, state = 0, 0, 0, 0
    if state > 0:
        ver_pts.append(int(acc_pt / num_pt + 0.5))

    return ver_pts, hor_pts


def group_line_segments_by_traverse(bin_img, tot_bin_img, is_vertical, is_implicit=False, min_threshold=10, corners=[]):
    def add_new_line(
        lines,
        seg_range,
        center,
        bin_img,
        tot_bin_img,
        is_vertical,
        is_implicit,
        corners,
    ):
        MIN_GAP_BETWEEN_CORNERS = 5

        # Find multiple line segment region
        if is_vertical:
            proc_img = bin_img[:, seg_range[0] : seg_range[1]].sum(axis=1)
            tot_proc_img = tot_bin_img[:, seg_range[0] : seg_range[1]].sum(axis=1)
        else:
            proc_img = bin_img[seg_range[0] : seg_range[1], :].sum(axis=0)
            tot_proc_img = tot_bin_img[seg_range[0] : seg_range[1], :].sum(axis=0)
        ind = np.where(tot_proc_img > 0)
        rs = ind[0].min()
        re = ind[0].max()

        region_list = []
        cur_pos = -1
        for k in range(rs, re):
            if cur_pos < 0:
                if tot_proc_img[k] > 0:
                    cur_pos = k
            elif tot_proc_img[k] == 0:
                region_list.append([cur_pos, k])
                cur_pos = -1
        if cur_pos >= 0:
            region_list.append([cur_pos, re])

        # Add new line segment
        new_line = {}
        new_line["range"] = seg_range
        new_line["region"] = [rs, re]
        new_line["region_list"] = region_list
        new_line["center"] = center
        new_line["implicit"] = is_implicit

        if is_implicit == False or len(corners) == 0:
            lines.append(new_line)
        else:  # Separate the implicit line using corner points
            foundCorners = list()
            for corner in corners:
                if corner > seg_range[0] and corner < seg_range[1]:
                    foundCorners.append(corner)

            if len(foundCorners) > 1:
                lastCorner = -9999
                for corner in foundCorners:
                    if corner - lastCorner >= MIN_GAP_BETWEEN_CORNERS:
                        new_line = deepcopy(new_line)
                        new_line["range"] = [corner - 1, corner + 1]
                        new_line["center"] = corner
                        lines.append(new_line)
                    lastCorner = corner
            else:
                lines.append(new_line)

    # Project on the perpendicular axis
    bin_proj = ((bin_img).sum(axis=int(not is_vertical))).astype(np.uint32)

    ret_lines = []
    bFound = False
    s, e = 0, 0
    sum_pos, num_pos = 0, 0
    max_length = 0
    for i in range(len(bin_proj)):
        sum_pos += bin_proj[i] * (i + 1)
        num_pos += bin_proj[i]
        max_length = max(max_length, bin_proj[i])
        if bFound == False:
            if bin_proj[i] > 0:
                bFound = True
                s = i
        elif bFound == True:
            if bin_proj[i] == 0:
                bFound = False
                e = i
                if max_length > min_threshold:
                    center = round(sum_pos / num_pos)
                    add_new_line(
                        ret_lines,
                        [s, e],
                        center,
                        bin_img,
                        tot_bin_img,
                        is_vertical,
                        is_implicit,
                        corners,
                    )
                sum_pos, num_pos = 0, 0
                max_length = 0

    # Add the right-most(bottom-most) line
    if bFound == True:
        e = len(bin_proj)
        if max_length > min_threshold:
            center = round(sum_pos / num_pos)
            add_new_line(
                ret_lines,
                [s, e],
                center,
                bin_img,
                tot_bin_img,
                is_vertical,
                is_implicit,
                corners,
            )

    return ret_lines


def align_length_implicit_lines(lines):
    min_pos, max_pos = 9999, -1
    for k in range(len(lines)):
        min_pos = min(min_pos, lines[k]["region"][0])
        max_pos = max(max_pos, lines[k]["region"][1])
    for k in range(len(lines)):
        lines[k]["region"] = [min_pos, max_pos]
        lines[k]["region_list"] = [[min_pos, max_pos]]

    return lines


def merge_and_sort_line_segments(ex_line, im_line, perpend_line, start, end):
    def merge_lines(tot_line, ex_line, im_line):
        if im_line is None:
            tot_line.append(ex_line)
        else:
            im_line["region_list"] = im_line["region_list"] + ex_line["region_list"]
            im_line["region"] = [
                min(im_line["region"][0], ex_line["region"][0]),
                max(im_line["region"][1], ex_line["region"][1]),
            ]
            tot_line.append(im_line)

    tot_line = list()
    cur_ex_idx = 0
    tot_ex_idx = len(ex_line)
    cur_im_idx = 0
    tot_im_idx = len(im_line)

    if tot_im_idx == 0:
        return ex_line

    while 1:
        operation = 0   # 0:merge, 1:ex, 2:im
        if cur_ex_idx >= tot_ex_idx and cur_im_idx >= tot_im_idx:
            break
        elif cur_ex_idx < tot_ex_idx and cur_im_idx < tot_im_idx:
            if ex_line[cur_ex_idx]["center"] < im_line[cur_im_idx]["center"]:
                if ex_line[cur_ex_idx]["range"][1] > im_line[cur_im_idx]["range"][0]:
                    operation = 0
                else:
                    operation = 1
            else:
                if ex_line[cur_ex_idx]["range"][0] < im_line[cur_im_idx]["range"][1]:
                    operation = 0
                else:
                    operation = 2
        elif cur_ex_idx < tot_ex_idx:
            operation = 1
        else:
            operation = 2

        if operation == 0:
            merge_lines(tot_line, ex_line[cur_ex_idx], im_line[cur_im_idx])
            cur_im_idx += 1
            cur_ex_idx += 1
        elif operation == 1:
            tot_line.append(ex_line[cur_ex_idx])
            cur_ex_idx += 1
        elif operation == 2:
            tot_line.append(im_line[cur_im_idx])
            cur_im_idx += 1

    return tot_line


def add_edge_lines(lines, start, end):
    def add_edge_line(tot_line, pos):
        new_line = {}
        new_line["range"] = [pos, pos]
        new_line["region"] = [0, 9999]
        new_line["region_list"] = [[0, 9999]]
        new_line["center"] = pos
        new_line["implicit"] = True
        tot_line.append(new_line)

    EDGE_BOUNDARY = 10

    tot_line = list()

    # Add the first line segments if not exist
    edge_boundary = EDGE_BOUNDARY if not lines[0]["implicit"] else 2 * EDGE_BOUNDARY
    if lines[0]["range"][0] > start + edge_boundary:
        add_edge_line(tot_line, start)
    tot_line += lines

    # Add the last line segments if not exist
    edge_boundary = EDGE_BOUNDARY if not lines[-1]["implicit"] else 2 * EDGE_BOUNDARY
    if lines[-1]["range"][1] < end - edge_boundary:
        add_edge_line(tot_line, end)

    return tot_line
#########################################################################################
# End of aux functions
#########################################################################################



def getCellsAccordingToCornerEdge(
    scoremap,
    threshold,
    low_bound,
    mask_area=None,
    transform_type=0,   # 0:None, 1:Rotation
):
    # params
    MIN_TABLE_SIZE_PIXEL = 20
    h, w, num_ch = scoremap.shape

    # Filter out padding area
    if mask_area is not None:
        x1, y1, x2, y2 = mask_area
        roi_mask = np.zeros(scoremap.shape[:2], dtype=np.uint8)
        roi_mask[y1:y2, x1:x2] = 1
        scoremap *= np.expand_dims(roi_mask, axis=2)

    # Map separation
    map_corner = scoremap[:, :, 0]
    map_hor = scoremap[:, :, 1]
    map_ver = scoremap[:, :, 2]
    map_ihor = scoremap[:, :, 3]
    map_iver = scoremap[:, :, 4]

    # binarization
    corner_bin = cv2.threshold(map_corner, low_bound, 1, 0)[1].astype(np.uint8)
    hor_bin = cv2.threshold(map_hor, threshold, 1, 0)[1].astype(np.uint8)
    ver_bin = cv2.threshold(map_ver, threshold, 1, 0)[1].astype(np.uint8)
    ihor_bin = cv2.threshold(map_ihor, low_bound, 1, 0)[1].astype(np.uint8)
    iver_bin = cv2.threshold(map_iver, low_bound, 1, 0)[1].astype(np.uint8)

    # Table localization
    tot_bin = ver_bin + hor_bin + iver_bin + ihor_bin
    tot_bin[tot_bin > 0] = 1
    nLabels, labels, stats, centroids = cv2.connectedComponentsWithStats(tot_bin, connectivity=8)
    max_width, max_height = 0, 0
    for tidx in range(1, nLabels):
        width, height = stats[tidx][2:4]
        max_width = max(max_width, width)
        max_height = max(max_height, height)

    # main process
    res_tables = []
    if nLabels < 1:
        return res_tables

    # sort w.r.t. area
    table_stats = sorted(stats[1:], key=lambda stat: (stat[cv2.CC_STAT_AREA]), reverse=True)  # 0 is BG

    for tidx, table_stat in enumerate(table_stats):
        # Check table conditions
        left, top, width, height = table_stat[:4]
        if min(width, height) < MIN_TABLE_SIZE_PIXEL:
            break

        # masking with label map
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[top : top + height, left : left + width] = 1
        cur_hor_bin = hor_bin * mask
        cur_ver_bin = ver_bin * mask
        cur_ihor_bin = ihor_bin * mask
        cur_iver_bin = iver_bin * mask
        cur_corner_bin = corner_bin * mask

        table_theta = 0
        rotate_flag, H = False, None
        if transform_type == 1:  # Rotation
            # Get thetas of hough transforms of ver/hor lines
            hor_edge = cv2.Sobel(cur_hor_bin + cur_ihor_bin, cv2.CV_8U, 0, 1, 3)
            hor_hough = cv2.HoughLines(hor_edge, 1, np.pi / 180 * 0.1, 400)
            ver_edge = cv2.Sobel(cur_ver_bin + cur_iver_bin, cv2.CV_8U, 1, 0, 3)
            ver_hough = cv2.HoughLines(ver_edge, 1, np.pi / 180 * 0.1, 400)
            if hor_hough is None:
                if ver_hough is None:
                    line_thetas = np.array([0])
                else:
                    line_thetas = ver_hough[:, 0, 1] * 180 / np.pi
            elif ver_hough is None:
                line_thetas = (hor_hough[:, 0, 1] - np.pi * 0.5) * 180 / np.pi
            else:
                line_thetas = np.concatenate(
                    (
                        (hor_hough[:, 0, 1] - np.pi * 0.5) * 180 / np.pi,
                        ver_hough[:, 0, 1] * 180 / np.pi,
                    )
                )
            line_thetas[line_thetas > 90] -= 180

            # Get image orientation
            table_theta = np.median(line_thetas)

            # Rotate score maps
            rotate_flag = True if abs(table_theta) > 0.1 else False
            if rotate_flag:
                w_warped, h_warped = w, h
                H = cv2.getRotationMatrix2D((w_warped / 2, h_warped / 2), table_theta, 1)
                H = np.vstack((H, np.array([[0, 0, 1]])))  # from RotationMatrix to Homography

        if rotate_flag:
            Hinv = np.linalg.inv(H)
            cur_corner_bin = cv2.warpPerspective(cur_corner_bin, H, (w_warped, h_warped))
            cur_hor_bin = cv2.warpPerspective(cur_hor_bin, H, (w_warped, h_warped))
            cur_ver_bin = cv2.warpPerspective(cur_ver_bin, H, (w_warped, h_warped))
            cur_ihor_bin = cv2.warpPerspective(cur_ihor_bin, H, (w_warped, h_warped))
            cur_iver_bin = cv2.warpPerspective(cur_iver_bin, H, (w_warped, h_warped))

        # horizontal dilation to group implicit vertical lines
        hor_kernel = (
            np.array(
                [
                    0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0,
                    1, 1, 1, 1, 1,
                    0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0,
                ]
            )
            .astype(np.uint8)
            .reshape(5, 5)
        )
        cv2.dilate(cur_iver_bin, hor_kernel, cur_iver_bin, iterations=1)

        # find table rect
        point_cur_hor_bin = np.where(cur_hor_bin)  # (y,x)
        point_cur_ver_bin = np.where(cur_ver_bin)
        point_cur_ihor_bin = np.where(cur_ihor_bin)
        point_cur_iver_bin = np.where(cur_iver_bin)
        minx, miny, maxx, maxy = 9999, 9999, -9999, -9999
        if len(point_cur_hor_bin[0]) > 0:
            minx = min(minx, point_cur_hor_bin[1].min())
            maxx = max(maxx, point_cur_hor_bin[1].max())
        if len(point_cur_ver_bin[0]) > 0:
            miny = min(miny, point_cur_ver_bin[0].min())
            maxy = max(maxy, point_cur_ver_bin[0].max())
        if len(point_cur_ihor_bin[0]) > 0:
            minx = min(minx, point_cur_ihor_bin[1].min())
            maxx = max(maxx, point_cur_ihor_bin[1].max())
        if len(point_cur_iver_bin[0]) > 0:
            miny = min(miny, point_cur_iver_bin[0].min())
            maxy = max(maxy, point_cur_iver_bin[0].max())
        table_rect = {"l": minx, "r": maxx, "t": miny, "b": maxy}

        # find explicit lines by projection
        ver_lines = group_line_segments_by_traverse(
            cur_ver_bin,
            cur_ver_bin,
            is_vertical=True,
            is_implicit=False
        )
        hor_lines = group_line_segments_by_traverse(
            cur_hor_bin,
            cur_hor_bin,
            is_vertical=False,
            is_implicit=False
        )

        # Check below if you use TRACE with 4 types of corners
        ver_pts, hor_pts = get_projected_corner_points(cur_corner_bin)

        # find implicit lines by projection
        h_mul, w_mul = 0.25, 0.25
        iver_lines = group_line_segments_by_traverse(
            cur_iver_bin,
            cur_iver_bin,
            is_vertical=True,
            is_implicit=True,
            min_threshold=max(height * h_mul, 30),
        )
        ihor_lines = group_line_segments_by_traverse(
            cur_ihor_bin,
            cur_ihor_bin,
            is_vertical=False,
            is_implicit=True,
            min_threshold=max(width * w_mul, 30),
            corners=ver_pts,
        )

        # Align lengths of implicit lines
        iver_lines = align_length_implicit_lines(iver_lines)
        ihor_lines = align_length_implicit_lines(ihor_lines)

        # Merge explicit and implicit lines with the condition with explicily merged cell
        ver_lines = merge_and_sort_line_segments(ver_lines, iver_lines, hor_lines, table_rect["l"], table_rect["r"])
        hor_lines = merge_and_sort_line_segments(hor_lines, ihor_lines, ver_lines, table_rect["t"], table_rect["b"])

        # make cells from hor/ver lines (minimum edges should be 2)
        if len(ver_lines) < 2 or len(hor_lines) < 2:
            continue

        # add edge lines
        ver_lines = add_edge_lines(ver_lines, table_rect["l"], table_rect["r"])
        hor_lines = add_edge_lines(hor_lines, table_rect["t"], table_rect["b"])

        cells = np.empty(shape=(len(ver_lines) - 1, len(hor_lines) - 1), dtype=np.object)
        for i in range(len(ver_lines) - 1):
            for j in range(len(hor_lines) - 1):
                sx, sy = ver_lines[i]["center"], hor_lines[j]["center"]
                ex, ey = ver_lines[i + 1]["center"], hor_lines[j + 1]["center"]
                x_pt, y_pt = list(), list()
                for weight in np.arange(0.25, 1, 0.25):
                    x_pt.append(weight * sx + (1 - weight) * ex)
                    y_pt.append(weight * sy + (1 - weight) * ey)

                # find explicitly merged cells
                found = False
                for segment in ver_lines[i + 1]["region_list"]:
                    found |= (
                        (segment[0] < y_pt[0] < segment[1])
                        | (segment[0] < y_pt[1] < segment[1])
                        | (segment[0] < y_pt[2] < segment[1])
                    )
                merge_col = not found

                found = False
                for segment in hor_lines[j + 1]["region_list"]:
                    found |= (
                        (segment[0] < x_pt[0] < segment[1])
                        | (segment[0] < x_pt[1] < segment[1])
                        | (segment[0] < x_pt[2] < segment[1])
                    )
                merge_row = not found

                # add cell
                cell = {
                    "row_id": i + 1,
                    "col_id": j + 1,
                    "sx": sx,
                    "sy": sy,
                    "ex": ex,
                    "ey": ey,
                    "merge_col": merge_col,
                    "merge_row": merge_row,
                    "valid": True,
                }
                cells[i, j] = cell

        # merge cells
        final_cells = []
        num_col, num_row = cells.shape

        check_flag = np.zeros(cells.shape, dtype=np.uint8)

        # Check valid table
        if num_col < 2 and num_row < 2:
            continue

        # Make file table result
        table_sx, table_sy, table_ex, table_ey = 9999, 9999, -1, -1
        for i in range(num_col):
            for j in range(num_row):
                if check_flag[i, j] == 1:
                    continue

                # check col merge
                si = ei = i
                for ii in range(si, num_col - 1):
                    if check_flag[ii + 1, j]:
                        break
                    if not cells[ii, j]["merge_col"]:
                        break
                    ei = ii + 1

                # check row merge
                sj = ej = j
                for jj in range(sj, num_row - 1):
                    more_merge = True
                    for ii in range(si, ei + 1):
                        if check_flag[ii, jj + 1]:
                            more_merge = False
                            break  # No more merge
                        more_merge &= cells[ii, jj]["merge_row"]
                    if not more_merge:
                        break
                    ej = jj + 1

                # check merged cell
                for ii in range(si, ei + 1):
                    for jj in range(sj, ej + 1):
                        check_flag[ii, jj] = 1

                sx, sy = cells[si, sj]["sx"], cells[si, sj]["sy"]
                ex, ey = cells[ei, ej]["ex"], cells[ei, ej]["ey"]

                # Find table rect
                if sx < table_sx:
                    table_sx = sx
                if sy < table_sy:
                    table_sy = sy
                if ex > table_ex:
                    table_ex = ex
                if ey > table_ey:
                    table_ey = ey

                # Unwarp coordinate if necessary
                if transform_type > 0 and rotate_flag:
                    pt1 = warpCoord(Hinv, np.array([sx, sy, 1])).astype(np.int32).tolist()
                    pt2 = warpCoord(Hinv, np.array([ex, sy, 1])).astype(np.int32).tolist()
                    pt3 = warpCoord(Hinv, np.array([ex, ey, 1])).astype(np.int32).tolist()
                    pt4 = warpCoord(Hinv, np.array([sx, ey, 1])).astype(np.int32).tolist()
                    quad = pt1 + pt2 + pt3 + pt4
                else:
                    quad = [sx, sy, ex, sy, ex, ey, sx, ey]

                cell = {
                    "row_range": [sj, ej],
                    "col_range": [si, ei],
                    "quad": quad,
                    "text": "",
                    "confidence": 0.0,
                }
                final_cells.append(cell)

        if len(final_cells) == 0:
            continue

        # Table quad
        if transform_type > 0 and rotate_flag:
            pt1 = warpCoord(Hinv, np.array([table_sx, table_sy, 1])).astype(np.int32).tolist()
            pt2 = warpCoord(Hinv, np.array([table_ex, table_sy, 1])).astype(np.int32).tolist()
            pt3 = warpCoord(Hinv, np.array([table_ex, table_ey, 1])).astype(np.int32).tolist()
            pt4 = warpCoord(Hinv, np.array([table_sx, table_ey, 1])).astype(np.int32).tolist()
            quad = pt1 + pt2 + pt3 + pt4
        else:
            quad = [
                table_sx,
                table_sy,
                table_ex,
                table_sy,
                table_ex,
                table_ey,
                table_sx,
                table_ey,
            ]

        res_table = {
            "id": len(res_tables) + 1,
            "angle": "{:.1f}".format(table_theta),
            "rect": [table_sx, table_sy, table_ex - table_sx, table_ey - table_sy],
            "quad": quad,
            "cells": final_cells,
        }
        res_tables.append(res_table)

    return res_tables


def adjustResultCoordinates(polys, ratio_w, ratio_h, padx=0, pady=0):
    if len(polys) > 0:
        polys = np.array(polys).astype(np.float32)
        for k in range(len(polys)):
            if polys[k] is not None:
                polys[k] *= (ratio_w, ratio_h)
                polys[k] -= np.array((padx, pady))
    return polys


def adjustResultMinMaxCoord(polys, ratio_w, ratio_h, padx=0, pady=0):
    if len(polys) > 0:
        polys = np.array(polys).astype(np.float32)
        for k in range(len(polys)):
            if polys[k] is not None:
                polys[k] = polys[k] * (ratio_w, ratio_h, ratio_w, ratio_h)
                polys[k] -= np.array((padx, pady))
    return polys


def adjustTableCoordinates(tables, ratio_w, ratio_h, padx=0, pady=0):
    for k in range(len(tables)):
        # rect of tables
        rect = np.array(tables[k]["rect"], dtype=np.float32).reshape(-1, 2)
        rect *= (ratio_w, ratio_h)
        rect[0] -= np.array((padx, pady))
        tables[k]["rect"] = rect.astype(int).reshape(-1).tolist()
        quad = np.array(tables[k]["quad"], dtype=np.float32).reshape(-1, 2)
        quad *= (ratio_w, ratio_h)
        quad -= np.array((padx, pady))
        tables[k]["quad"] = quad.astype(int).reshape(-1).tolist()
        # quad of cells
        tables[k]["cells"] = adjustCellCoordinates(tables[k]["cells"], ratio_w, ratio_h, padx, pady)
    return tables


def adjustCellCoordinates(cells, ratio_w, ratio_h, padx=0, pady=0):
    if len(cells) > 0:
        for k in range(len(cells)):
            quad = np.array(cells[k]["quad"], dtype=np.float32).reshape(-1, 2)
            quad *= (ratio_w, ratio_h)
            quad -= np.array((padx, pady))
            cells[k]["quad"] = quad.astype(int).reshape(-1).tolist()
    return cells


def run(heatmap, args, target_ratio):
    # get params
    padx, pady = 0, 0
    mask_area = None
    if isinstance(target_ratio, int):
        ratio_w, ratio_h = 1 / target_ratio, 1 / target_ratio
    else:
        ratio_w, ratio_h = 1 / target_ratio[0], 1 / target_ratio[1]
        # get padding length
        if len(target_ratio) > 2:
            padx, pady = target_ratio[2], target_ratio[3]
        # get masking area
        if len(target_ratio) > 4:
            mask_area = target_ratio[4:8]  # x1, y1, x2, y2

    result = getCellsAccordingToCornerEdge(
        heatmap,
        args.threshold1,
        args.threshold2,
        mask_area=mask_area,
        transform_type=1,
    )
    result = adjustTableCoordinates(result, ratio_w, ratio_h, padx, pady)

    return result
