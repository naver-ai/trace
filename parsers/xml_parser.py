import os
import random
from collections import Iterable
from xml.dom import minidom

import file_utils


def flatten(lis):
    for item in lis:
        if isinstance(item, Iterable) and not isinstance(item, str):
            for x in flatten(item):
                yield x
        else:
            yield item


class ParserTRACE:
    def __init__(self, root_path, dataset, phase):
        self.gt = []
        base_folder = os.path.join(root_path, dataset)
        base_folder = os.path.join(base_folder, phase)
        image_files, _, _ = file_utils.list_files(base_folder)

        for img_file in image_files:
            basename, ext = os.path.splitext(os.path.basename(img_file))
            gt_file = os.path.join(base_folder, f"{basename}.xml")

            if os.path.exists(gt_file):
                quads = []
                lines = []
                with open(gt_file, "r") as f:
                    dom = minidom.parse(f)

                elem_tables = dom.documentElement.getElementsByTagName("table")
                for e_table in elem_tables:
                    elem_cells = e_table.getElementsByTagName("cell")
                    for e_cell in elem_cells:
                        points = str(e_cell.getElementsByTagName("Coords")[0].getAttribute("points"))
                        new_points = []
                        for p in points.split():
                            new_points.append(p.split(","))
                        quad = list(flatten(new_points))
                        quads.append(quad)

                        line_t = bool(int(e_cell.getElementsByTagName("Lines")[0].getAttribute("top")))
                        line_b = bool(int(e_cell.getElementsByTagName("Lines")[0].getAttribute("bottom")))
                        line_l = bool(int(e_cell.getElementsByTagName("Lines")[0].getAttribute("left")))
                        line_r = bool(int(e_cell.getElementsByTagName("Lines")[0].getAttribute("right")))
                        line = [line_t, line_b, line_l, line_r]
                        lines.append(line)

                self.gt.append({"file_name": img_file, "quads": quads, "lines": lines})

    def getDatasetSize(self):
        return len(self.gt)

    def lenFiles(self):
        return len(self.gt)

    def parseGT(self, index=-1):
        if index == -1:
            gt = self.gt[random.randrange(0, len(self.gt))]
        else:
            gt = self.gt[index]

        return gt["file_name"], gt


if __name__ == "__main__":
    parser = ParserTRACE("/data/db/table", "CARTE", "train")
    print(parser.lenFiles())
    print(parser.parseGT(-1))

