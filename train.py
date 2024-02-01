# -*- coding: utf-8 -*-
import argparse
import os
import shutil
import sys
import time
from collections import OrderedDict
from datetime import datetime
from subprocess import PIPE, Popen

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
from tensorboardX import SummaryWriter
from torch.autograd import Variable

import file_utils
import imgproc
from augmentations import TRACEAugmentation
from loader import TRACE_Dataset
from loss import TRACELoss
from model import TraceModel
from parse_config import parse_config_train


def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        if not name.endswith("num_batches_tracked"):
            new_state_dict[name] = v
    return new_state_dict


def resumeStateDict(net, state_dict):
    state_dict = copyStateDict(state_dict)
    new_state_dict = net.state_dict()
    for k in list(state_dict.keys()):
        if not k in new_state_dict.keys() or new_state_dict[k].shape != state_dict[k].shape:
            del state_dict[k]
            print("Warning!!! : different shape of tensors for {}".format(k))
    new_state_dict.update(state_dict)
    net.load_state_dict(new_state_dict)


def str2bool(v):
    return v.lower() in ("yes", "y", "true", "t", "1")


parser = argparse.ArgumentParser(description="TRACE Trainer")
parser.add_argument("-c", "--config_file", type=str, required=False)
parser.add_argument("--train_size", default=768, type=int, help="Image size for training")
parser.add_argument("--batch_size", default=32, type=int, help="Batch size for training")
parser.add_argument("--resume", default=None, type=str, help="Resume from checkpoint")
parser.add_argument("--num_workers", default=12, type=int, help="Number of workers used in dataloading")
parser.add_argument("--max_iter", default=100000, type=int, help="Number of training iterations")
parser.add_argument("--start_iter", default=0, type=int, help="Begin counting iterations starting from this value")
parser.add_argument("--cuda", default=True, type=str2bool, help="Use cuda to train model")
parser.add_argument("--optimizer", default="adamw", type=str, help="Optimizer (adamw/adam/sgd)")
parser.add_argument("--lr", "--learning-rate", default=3e-4, type=float, help="initial learning rate")
parser.add_argument("--gamma", default=0.8, type=float, help="Gamma update for LR")
parser.add_argument("--eval", action="store_true", default=False,help="Enable evaluation during training")
parser.add_argument("--save_folder", default="eval/", help="Location to save checkpoint models")
parser.add_argument("--save_interval", default=1000, type=int, help="Checkpoint save and evaluation interval")
parser.add_argument("--data_path", default="/data/db/table/", help="Location of table datasets")
parser.add_argument("--train_sets", default="SubTableBank", help="Datasets for training")
parser.add_argument("--mixratio", default=[1], help="Mixture ratio of datasaets")
parser.add_argument("--eval_set", default=None, type=str, help="Evaluation dataset")
parser.add_argument("--comment", default="write_comment_here", type=str, help="Tensorboard log comment")
args = parser.parse_args()

# parse config file
args = parse_config_train(args)


# configurations
means = (123, 117, 104)  # RGB order
batch_size = args.batch_size
max_iter = args.max_iter
stepvalues = [10000 * (k + 1) for k in range(10)]
gamma = args.gamma
scale_down = 2
out_dim = int(args.train_size / scale_down)

if args.cuda and torch.cuda.is_available():
    torch.set_default_tensor_type("torch.cuda.FloatTensor")
else:
    torch.set_default_tensor_type("torch.FloatTensor")


def train():
    # prepare save folders
    current_folder = "./"

    args.save_folder = os.path.join(current_folder, args.save_folder)
    print("Save folder: ", args.save_folder)
    if not os.path.exists(args.save_folder):
        os.mkdir(args.save_folder)
    file_utils.change_permissions_recursive(args.save_folder, 0o777)

    # build network
    net = TraceModel()

    print("the number of model parameters: {}".format(sum([p.data.nelement() for p in net.parameters()])))

    if args.resume:
        args.resume = os.path.join(current_folder, args.resume)
        print("Resuming training, loading {}...".format(args.resume))
        resumeStateDict(net, torch.load(args.resume))

    # Set optimizer
    if args.optimizer == "adamw":
        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, net.parameters()), lr=args.lr)
    elif args.optimizer == "adam":
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=args.lr)
    else:
        print("Unknown optimizer...")
        sys.exit(1)

    criterion = TRACELoss(neg_pos_ratio=3)

    transform = TRACEAugmentation(args.train_size, means)
    print("Loading Training Dataset... {}".format(str(args.train_sets)))
    dataset = TRACE_Dataset(
        args.train_sets,
        rootpath=args.data_path,
        phase="train",
        scale_down=scale_down,
        transform=transform,
        mixratio=args.mixratio,
    )
    if args.eval:
        print("Evaluation Dataset... {}".format(str(args.eval_set)))
    print("Start training...")

    data_loader = data.DataLoader(
        dataset,
        batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        pin_memory=True,
        generator=torch.Generator(device="cuda"),
    )

    # init summary
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    writer = SummaryWriter(logdir=os.path.join(current_folder, "runs", current_time + "_" + args.comment))

    # multi-process eval
    num_gpus = torch.cuda.device_count()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    if "CUDA_VISIBLE_DEVICES" in os.environ.keys():
        available_gpus = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
        num_gpus = len(available_gpus)
    else:
        available_gpus = list(range(num_gpus))

    # select GPU for evaluation
    train_gpus = list(range(max(1, num_gpus - 1)))
    print("available_gpus:", available_gpus)

    if args.cuda:
        net = torch.nn.DataParallel(net, device_ids=train_gpus)
        cudnn.benchmark = False

    net.train()

    # init variables
    epoch_size = int(len(dataset) / args.batch_size)
    step_index = 0
    epoch = 0
    batch_iterator = None
    pEval, lastEvalIter, lastEvalModel = None, 0, None
    best_score = 0
    display_iter = 10
    t0 = time.time()
    for iteration in range(args.start_iter, max_iter):
        if (not batch_iterator) or (iteration % epoch_size == 0):
            # create batch iterator
            batch_iterator = iter(data_loader)
            epoch += 1

        if iteration in stepvalues:
            step_index += 1
            adjust_learning_rate(optimizer, args.gamma, step_index)

        # load train data
        images, gts, weights = next(batch_iterator)

        if iteration % display_iter == 0:
            from copy import deepcopy

            orig_images = deepcopy(images[0]).cpu()
            orig_gts = deepcopy(gts[0]).cpu()
            orig_weights = deepcopy(weights[0]).cpu()

        if args.cuda:
            images = Variable(images.cuda())
            gts = Variable(gts.cuda())
            weights = Variable(weights.cuda())
        else:
            images = Variable(images)
            gts = Variable(gts)
            weights = Variable(weights)

        # forward
        out = net(images)
        if isinstance(out, tuple):
            out = out[0]  # ignore feature map

        # back-propagation
        optimizer.zero_grad()
        loss = criterion(out, gts, weights)
        loss.backward()
        optimizer.step()

        # display
        pred_img = out[0, :, :, :].cpu().data.numpy()
        if iteration % display_iter == 0:
            t1 = time.time()
            print("iter " + repr(iteration) + " || loss: %.4f (Time : %.1f)" % (loss.sum().item(), (t1 - t0)))
            t0 = time.time()

            render_img = cv2.resize(orig_images.numpy().transpose((1, 2, 0)), (out_dim, out_dim))
            render_img = imgproc.denormalizeMeanVariance(render_img)
            in_img = np.clip(render_img.transpose(2, 0, 1), 0, 255).astype(np.uint8)

            render_img = np.zeros((3, out_dim, out_dim * 8), dtype=np.uint8)
            render_img[:, :, :out_dim] = in_img
            render_gt_weight = np.clip(orig_weights[:, :, 0].numpy().reshape(1, out_dim, out_dim) * 255, 0, 255).astype(
                np.uint8
            )

            # Corner map
            render_img[0, :, out_dim : 2 * out_dim] = np.clip(
                orig_gts[:, :, 0].numpy().reshape(1, out_dim, out_dim) * 255, 0, 255
            ).astype(np.uint8)
            render_img[0, :, 2 * out_dim : 3 * out_dim] = np.clip((pred_img[:, :, 0]) * 255, 0, 255).astype(np.uint8)

            # Link map
            render_img[0, :, 3 * out_dim : 4 * out_dim] = np.clip(
                orig_gts[:, :, 1].numpy().reshape(1, out_dim, out_dim) * 255, 0, 255
            ).astype(np.uint8)
            render_img[1, :, 3 * out_dim : 4 * out_dim] = np.clip(
                orig_gts[:, :, 2].numpy().reshape(1, out_dim, out_dim) * 255, 0, 255
            ).astype(np.uint8)
            render_img[0, :, 4 * out_dim : 5 * out_dim] = np.clip(pred_img[:, :, 1] * 255, 0, 255).astype(np.uint8)
            render_img[1, :, 4 * out_dim : 5 * out_dim] = np.clip(pred_img[:, :, 2] * 255, 0, 255).astype(np.uint8)
            render_img[0, :, 5 * out_dim : 6 * out_dim] = np.clip(
                orig_gts[:, :, 3].numpy().reshape(1, out_dim, out_dim) * 255, 0, 255
            ).astype(np.uint8)
            render_img[1, :, 5 * out_dim : 6 * out_dim] = np.clip(
                orig_gts[:, :, 4].numpy().reshape(1, out_dim, out_dim) * 255, 0, 255
            ).astype(np.uint8)
            render_img[0, :, 6 * out_dim : 7 * out_dim] = np.clip(pred_img[:, :, 3] * 255, 0, 255).astype(np.uint8)
            render_img[1, :, 6 * out_dim : 7 * out_dim] = np.clip(pred_img[:, :, 4] * 255, 0, 255).astype(np.uint8)
            render_img[:, :, -out_dim - 1 : -1] = render_gt_weight

            writer.add_scalar("loss", loss.sum().item(), iteration)
            writer.add_image("image", render_img, iteration)
            writer.add_scalar("lr", optimizer.param_groups[0]["lr"], iteration)
        bModelSaved = False

        if iteration % args.save_interval == 0:
            print("Saving state, iter : ", iteration)
            model_file = os.path.join(args.save_folder, "ckpt_" + repr(iteration) + ".pth")

            torch.save(net.state_dict(), model_file)
            os.chmod(model_file, 0o777)
            bModelSaved = True

        # Start evaluation
        if args.eval:
            try:
                if pEval is None:
                    if bModelSaved:
                        lastEvalIter = iteration
                        lastEvalModel = model_file
                        print("Evaluation started at iteration {} on {}...".format(lastEvalIter, args.eval_set))
                        eval_cmd = (
                            "CUDA_VISIBLE_DEVICES="
                            + str(available_gpus[-1])
                            + " python test.py --trained_model="
                            + model_file
                            + " --eval"
                            + " --res_postfix="
                            + args.comment
                            + " -c="
                            + args.config_file
                        )
                        eval_cmd += " -i " + os.path.join(args.data_path, args.eval_set, "test")
                        pEval = Popen(eval_cmd, shell=True, stdout=PIPE, stderr=PIPE)
                elif pEval.poll() is not None:
                    (scorestring, stderrdata) = pEval.communicate()
                    print("end of evaluation with {}, {}".format(scorestring, stderrdata))

                    hmean = float(
                        str(scorestring).strip().split('"hmean":')[1].split(",")[0].split("}")[0].split("\\")[0].strip()
                    )
                    writer.add_scalar("test_hmean", hmean, lastEvalIter)
                    print("test_hmean for {}-th iter : {:.4f}".format(lastEvalIter, hmean))

                    # Save best score model
                    if hmean > best_score:
                        best_score = hmean
                        shutil.copyfile(
                            lastEvalModel,
                            os.path.join(args.save_folder, "model_best.pth"),
                        )
                        print("New best score! The best model is saved.")
                    if pEval is not None:
                        pEval.kill()
                    pEval = None
            except Exception as e:
                print("exception happened in evaluation : " + str(e))
                if pEval is not None:
                    pEval.kill()
                pEval = None
                pass

    torch.save(net.state_dict(), os.path.join(args.save_folder, "model.pth"))
    os.chmod(os.path.join(args.save_folder, "model.pth"), 0o777)


def adjust_learning_rate(optimizer, gamma, step):
    lr = args.lr * (gamma ** (step))
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


if __name__ == "__main__":
    cv2.setNumThreads(0)  # prevent deadlock caused by conflict with pytorch
    train()
