import json


def parse_config(args):
    if args.config_file is not None:
        print("config_file = {}".format(args.config_file))
        with open(args.config_file, encoding="utf-8") as f:
            arguments = json.load(f)
            args.canvas_size = arguments.get("canvas_size", args.canvas_size)
            args.mag_ratio = arguments.get("mag_ratio", args.mag_ratio)
            args.ocr_url = arguments.get("ocr_url", None)
            if "scale_down" in args.__dict__:
                args.scale_down = arguments.get("scale_down", args.scale_down)
            thresholds = arguments.get("thresholds", None)
            if thresholds is not None:
                for k, th in enumerate(thresholds):
                    if k == 0:
                        args.threshold1 = th
                    elif k == 1:
                        args.threshold2 = th
    return args


def parse_config_train(args):
    if args.config_file is not None:
        # print("config_file = {}".format(args.config_file))
        with open(args.config_file, encoding="utf-8") as f:
            arguments = json.load(f)
            # For training
            if "train_size" in args.__dict__:
                args.train_size = arguments.get("train_size", args.train_size)
            if "resume" in args.__dict__:
                args.resume = arguments.get("resume", args.resume)
            if "data_path" in args.__dict__:
                args.data_path = arguments.get("data_path", args.data_path)
            if "save_folder" in args.__dict__:
                args.save_folder = arguments.get("save_folder", args.save_folder)
            if "train_sets" in args.__dict__:
                args.train_sets = arguments.get("train_sets", args.train_sets)
            if "comment" in args.__dict__:
                args.comment = arguments.get("comment", args.comment)
            if "batch_size" in args.__dict__:
                args.batch_size = arguments.get("batch_size", args.batch_size)
            if "mixratio" in args.__dict__:
                args.mixratio = arguments.get("mixratio", args.mixratio)
            if "scale_down" in args.__dict__:
                args.scale_down = arguments.get("scale_down", args.scale_down)

            # For evaluation
            if "canvas_size" in args.__dict__:
                args.canvas_size = arguments.get("canvas_size", None)
            if "eval_set" in args.__dict__:
                args.eval_set = arguments.get("eval_set", None)
            if "mag_ratio" in args.__dict__:
                args.mag_ratio = arguments.get("mag_ratio", None)
            thresholds = arguments.get("thresholds", None)
            if thresholds is not None:
                for k, th in enumerate(thresholds):
                    if k == 0:
                        args.threshold1 = th
                    elif k == 1:
                        args.threshold2 = th
    return args
