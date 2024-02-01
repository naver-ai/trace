"""
TRACE
Copyright (c) 2022-present NAVER Corp.
MIT License
"""
import argparse
import os

import gradio as gr
import numpy as np
import torch

import file_utils
import imgproc
import postprocessor
from model import TraceModel
from parse_config import parse_config


def demo_process(input_img):
    global net, args

    image = np.array(input_img)
    output = test_net(net, image, args)

    return output


def test_net(net, image, args):
    global result_path

    # resize
    s = args.canvas_size
    mag_ratio = args.mag_ratio
    img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(image, s, mag_ratio=mag_ratio)

    # preprocessing
    x = imgproc.normalizeMeanVariance(img_resized)
    x = torch.from_numpy(x).permute(2, 0, 1)  # [h, w, c] to [c, h, w]
    x = torch.autograd.Variable(x.unsqueeze(0))  # [c, h, w] to [b, c, h, w]
    if torch.cuda.is_available():
        x = x.cuda()

    # forward pass
    with torch.no_grad():
        y = net(x)
        if isinstance(y, tuple):
            y = y[0]  # ignore feature map
    res_heatmap = y[0].cpu().data.numpy()

    # Post-processing
    res_dict = postprocessor.run(res_heatmap, args, target_ratio)

    # render result image (saving xlsx file)
    res_image, res_xlsx_file = file_utils.saveTraceResult(
        "input.jpg",
        image[:, :, ::-1],
        None,
        res_dict,
        dirname=result_path,
        saveXlsx=True,
    )

    return res_image, res_dict, res_xlsx_file


if __name__ == "__main__":

    def str2bool(v):
        return v.lower() in ("yes", "y", "true", "t", "1")

    parser = argparse.ArgumentParser()
    # Base
    parser.add_argument("-c", "--config_file", default="configs/trace.json", type=str, required=True)
    parser.add_argument("-m", "--trained_model", type=str, required=True)
    parser.add_argument("--url", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--sample_path", type=str, default="./samples/")
    # parameters
    parser.add_argument("--canvas_size", default=1024, type=int, help="Max size of image canvas")
    parser.add_argument("--mag_ratio", default=10, type=float, help="Image magnification ratio")
    parser.add_argument("--threshold1", default=0.2, type=float, help="Threshold for explicit edges")
    parser.add_argument("--threshold2", default=0.1, type=float, help="Threshold for implicit edges")

    # parse config file
    args = parser.parse_args()
    args = parse_config(args)

    result_path = "./result_app/"
    if not os.path.isdir(result_path):
        os.mkdir(result_path)

    # prepare model
    print("Loading weights from checkpoint (" + args.trained_model + ")")
    net = TraceModel()
    if torch.cuda.is_available():
        net = net.cuda()
        net = torch.nn.DataParallel(net)
        net.load_state_dict(torch.load(args.trained_model))
    net.eval()

    # add sample images
    example_sample = []
    if args.sample_path:
        if os.path.isdir(args.sample_path):
            example_sample = file_utils.get_image_list(args.sample_path)
        else:
            example_sample = [args.sample_path]

    demo = gr.Interface(
        fn=demo_process,
        inputs="image",
        outputs=["image", "json", "file"],
        title=f"TRACE demonstration",
        examples=example_sample if example_sample else None,
    )
    demo.launch(server_name=args.url, server_port=args.port)
