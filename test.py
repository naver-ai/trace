# -*- coding: utf-8 -*-
import argparse
import os
import time
from collections import OrderedDict
from subprocess import PIPE, Popen
from xml.dom import minidom

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from pdf2image import convert_from_path
from PyPDF2 import PdfReader
from torch.autograd import Variable

import file_utils
import imgproc
import postprocessor
from model import TraceModel
from parse_config import parse_config


def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict


def str2bool(v):
    return v.lower() in ("yes", "y", "true", "t", "1")


parser = argparse.ArgumentParser(description="TRACE tester")
parser.add_argument("-i", "--image", type=str, required=True, nargs="+")
parser.add_argument("-c", "--config_file", type=str, required=False)
parser.add_argument("-m", "--trained_model", default="weights/pretrained.pth", type=str,help="Trained model")
parser.add_argument("--canvas_size", default=1024, type=int, help="Max size of image canvas")
parser.add_argument("--mag_ratio", default=10, type=float, help="Image magnification ratio")
parser.add_argument("--threshold1", default=0.2, type=float, help="Threshold for explicit edges")
parser.add_argument("--threshold2", default=0.1, type=float, help="Threshold for implicit edges")
parser.add_argument("--save_heatmap", action="store_true",default=False, help="Save intermediate heatmap image")
parser.add_argument("--res_postfix", default="", type=str, help="Postfix for result path")
parser.add_argument("--cuda", default=True, type=str2bool, help="Use cuda to train model")
parser.add_argument("--show_time", default=False, action="store_true", help="Show processing times")
parser.add_argument("--eval", action="store_true",default=False, help="Evaluation using CARTE")
args = parser.parse_args()

# parse config file
args = parse_config(args)

result_folder = "./result" + args.res_postfix + "/"
if not os.path.isdir(result_folder):
    os.mkdir(result_folder)


def test_net(net, image, args):
    t0 = time.time()

    # resize
    s = args.canvas_size
    mag_ratio = args.mag_ratio
    img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(image, s, mag_ratio=mag_ratio)

    # preprocessing
    x = imgproc.normalizeMeanVariance(img_resized)
    x = torch.from_numpy(x).permute(2, 0, 1)  # [h, w, c] to [c, h, w]
    x = Variable(x.unsqueeze(0))  # [c, h, w] to [b, c, h, w]
    if args.cuda:
        x = x.cuda()

    # forward pass
    with torch.no_grad():
        y = net(x)
        if isinstance(y, tuple):
            y = y[0]  # ignore feature map
    res_heatmap = y[0].cpu().data.numpy()

    t0 = time.time() - t0

    # Post-processing
    t1 = time.time()
    result = postprocessor.run(res_heatmap, args, target_ratio)
    t1 = time.time() - t1

    # render results
    render_img = render_img2 = None
    if args.save_heatmap:
        w, h = size_heatmap
        render_img = np.zeros((h, w, 3), dtype=np.uint8)
        render_img[:, :, 0] = np.clip(res_heatmap[:h, :w, 0] * 255, 0, 255).astype(np.uint8)

        img_resized = cv2.resize(img_resized, (w, h), interpolation=cv2.INTER_LINEAR)
        img_resized = cv2.cvtColor(img_resized, cv2.COLOR_RGB2BGR)
        render_img = cv2.addWeighted(img_resized, 0.4, render_img, 0.6, 1.0)

        # draw link
        render_img2 = np.zeros((h, w, 3), dtype=np.uint8)
        render_img2[:, :, 0] = np.clip(res_heatmap[:h, :w, 1] * 255, 0, 255).astype(np.uint8)
        render_img2[:, :, 1] = np.clip(res_heatmap[:h, :w, 2] * 255, 0, 255).astype(np.uint8)
        render_img2[:, :, 2] += np.clip(res_heatmap[:h, :w, 3] * 255, 0, 255).astype(np.uint8)
        render_img2[:, :, 1] += np.clip(res_heatmap[:h, :w, 3] * 255, 0, 255).astype(np.uint8)
        render_img2[:, :, 0] += np.clip(res_heatmap[:h, :w, 4] * 255, 0, 255).astype(np.uint8)
        render_img2[:, :, 2] += np.clip(res_heatmap[:h, :w, 4] * 255, 0, 255).astype(np.uint8)
        render_img2 = cv2.addWeighted(img_resized, 0.4, render_img2, 0.6, 1.0)

    if args.show_time:
        print("\ninfer/postproc time : {:.3f}/{:.3f}".format(t0, t1))

    return result, res_heatmap, render_img, render_img2


if __name__ == "__main__":
    # load net
    net = TraceModel()

    if args.cuda:
        net = net.cuda()
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = False

    if not args.eval:
        print("Loading weights from checkpoint (" + args.trained_model + ")")
    if args.cuda:
        net.load_state_dict(torch.load(args.trained_model))
    else:
        net.load_state_dict(copyStateDict(torch.load(args.trained_model, map_location="cpu")))
    net.eval()

    t = time.time()

    # supported input types = ['img', 'pdf']
    input_arg = args.image
    input_type = "img"
    pages = 1
    if os.path.isdir(input_arg[0]):  # directory
        image_list = file_utils.get_image_list(input_arg[0])  # image list

        if len(image_list) == 0:
            input_type = "pdf"
            image_list = file_utils.get_pdf_list(input_arg[0])  # pdf list
            if len(image_list) == 0:
                input_type = "xml"
                image_list = file_utils.get_xml_list(input_arg[0])  # xml list
    else:  # single file
        image_list = input_arg

        basename, ext = os.path.splitext(input_arg[0])
        if ext.lower() in [".pdf"]:
            input_type = "pdf"
        elif ext.lower() in [".xml"]:
            input_type = "xml"

    for k, image_path in enumerate(image_list):
        if not args.eval:
            print("Test image {:d}/{:d}: {:s}".format(k + 1, len(image_list), image_path), end="\r")
        if input_type == "pdf":
            # pdf_pages = convert_from_path(image_path, 72) # for IC13
            pdf_pages = convert_from_path(image_path, 150)  # for SciTSR
            pages = len(pdf_pages)
        elif input_type == "xml":
            xml_path = image_path
            table_dom = minidom.parse(image_path)
            image_path = table_dom.getElementsByTagName("document")[0].getAttribute("filename")
            image_path = os.path.dirname(xml_path) + "/../../../pdf/" + image_path
        for p in range(pages):
            # load image
            try:
                image = None
                if input_type == "img":
                    image = imgproc.loadImage(image_path)
                    res_filename = image_path
                elif input_type == "pdf":
                    image = np.asarray(pdf_pages[p], dtype="int32").astype(np.uint8)
                elif input_type == "xml":
                    pdf_page = PdfReader(open(image_path, "rb")).pages[0]
                    pdf_shape = pdf_page.mediabox
                    pdf_height = pdf_shape[3] - pdf_shape[1]
                    pdf_width = pdf_shape[2] - pdf_shape[0]

                    pdf_page = convert_from_path(image_path, size=(pdf_width, pdf_height))[0]
                    image = np.array(pdf_page, dtype=np.uint8)
                    image_path = xml_path
            except:
                print("Fail to read image file...")
                continue

            if input_type != "img":
                print("Page ({:d}/{:d})".format(p + 1, pages), end="\r")
                filename, file_ext = os.path.splitext(os.path.basename(image_path))
                res_filename = "{}_{}{}".format(filename, p, file_ext)

            # run prediction
            result, res_score, img_corner, img_line = test_net(net, image, args)

            # save score text
            if not args.eval:
                filename, file_ext = os.path.splitext(os.path.basename(image_path))
                if img_corner is not None:
                    mask_file = result_folder + "/res_" + filename + "_corner_mask.jpg"
                    cv2.imwrite(mask_file, img_corner)
                if img_line is not None:
                    mask_file = result_folder + "/res_" + filename + "_edge_mask.jpg"
                    cv2.imwrite(mask_file, img_line)

                file_utils.saveTraceResult(
                    image_path,
                    image[:, :, ::-1],
                    None,
                    result,
                    dirname=result_folder,
                    saveXlsx=False,
                )

            # save xml
            root = minidom.Document()
            xml = root.createElement("document")
            img_basename = os.path.splitext(os.path.basename(image_path))[0]
            result_xml_file = f"{img_basename}.xml"
            xml.setAttribute("filename", os.path.basename(image_path))

            for trace_res in result:
                table_elem = root.createElement("table")
                table_coords = root.createElement("Coords")
                table_xywh = trace_res["rect"]
                table_xywh = [int(x) for x in table_xywh]
                table_quad = [
                    [table_xywh[0], table_xywh[1]],
                    [table_xywh[0] + table_xywh[2], table_xywh[1]],
                    [table_xywh[0] + table_xywh[2], table_xywh[1] + table_xywh[3]],
                    [table_xywh[0], table_xywh[1] + table_xywh[3]],
                ]
                table_coords.setAttribute(
                    "points",
                    f"{table_quad[0][0]},{table_quad[0][1]} {table_quad[3][0]},{table_quad[3][1]} {table_quad[2][0]},{table_quad[2][1]} {table_quad[1][0]},{table_quad[1][1]}",
                )
                table_elem.appendChild(table_coords)

                for cell in trace_res["cells"]:
                    cell_elem = root.createElement("cell")
                    cell_elem.setAttribute("start-row", f"{cell['row_range'][0]}")
                    cell_elem.setAttribute("end-row", f"{cell['row_range'][1]}")
                    cell_elem.setAttribute("start-col", f"{cell['col_range'][0]}")
                    cell_elem.setAttribute("end-col", f"{cell['col_range'][1]}")
                    cell_coords = root.createElement("Coords")
                    quad = [int(x) for x in cell["quad"]]
                    x1, y1, x2, y2, x3, y3, x4, y4 = quad
                    cell_coords.setAttribute("points", f"{x1},{y1} {x4},{y4} {x3},{y3} {x2},{y2}")
                    cell_elem.appendChild(cell_coords)
                    table_elem.appendChild(cell_elem)
                xml.appendChild(table_elem)
            root.appendChild(xml)
            xml_string = root.toprettyxml(indent="\t")
            with open(os.path.join(result_folder, result_xml_file), "w") as fp:
                fp.write(xml_string)


    if args.eval:
        # new subprocess
        eval_cmd = "python evaluation/carte/eval.py -g {} -p {} -n".format(args.image[0], os.path.dirname(result_folder))
        print(eval_cmd)
        proc = Popen(eval_cmd, shell=True, stdout=PIPE, bufsize=1)
        out, err = proc.communicate()

        for line in str(out).split("\\n"):
            if not args.eval:
                print(line)
            if "f1:" in line:
                f1 = line.split("f1:")[1]
        print('"hmean": ', f1)
    else:
        print("elapsed time : {}s".format(time.time() - t))
