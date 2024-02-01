# -*- coding: utf-8 -*-
import glob
import os

import cv2
import numpy as np

IMAGE_EXTENTIONS = ["jpg", "jpeg", "png", "JPG", "tiff", "TIFF"]


def get_image_list(dir):
    image_list = []
    for ext in IMAGE_EXTENTIONS:
        image_list.extend(get_file_list(dir, ext))
    return image_list


def get_pdf_list(dir):
    image_list = []
    for ext in ["pdf"]:
        image_list.extend(get_file_list(dir, ext))
    return image_list


def get_xml_list(dir):
    image_list = []
    for ext in ["xml"]:
        image_list.extend(get_file_list(dir, ext))
    return image_list


def get_file_list(dir, ext):
    return glob.glob("%s/**/*.%s" % (dir, ext), recursive=True)


def get_files(img_dir):
    imgs, masks, xmls = list_files(img_dir)
    return imgs, masks, xmls


def list_files(in_path):
    img_files = []
    mask_files = []
    gt_files = []
    for dirpath, dirnames, filenames in os.walk(in_path):
        for file in filenames:
            filename, ext = os.path.splitext(file)
            ext = str.lower(ext)
            if (
                ext == ".jpg"
                or ext == ".jpeg"
                or ext == ".gif"
                or ext == ".png"
                or ext == ".pgm"
                or ext == ".tif"
                or ext == ".bmp"
            ):
                img_files.append(os.path.join(dirpath, file))
            elif ext == ".bmp":
                mask_files.append(os.path.join(dirpath, file))
            elif ext == ".xml" or ext == ".gt" or ext == ".txt":
                gt_files.append(os.path.join(dirpath, file))
            elif ext == ".zip":
                continue
    # img_files.sort()
    # mask_files.sort()
    # gt_files.sort()
    return img_files, mask_files, gt_files


def list_pdf_files(in_path):
    pdf_files = []
    for dirpath, dirnames, filenames in os.walk(in_path):
        for file in filenames:
            filename, ext = os.path.splitext(file)
            ext = str.lower(ext)
            if ext == ".pdf":
                pdf_files.append(os.path.join(dirpath, file))
    return pdf_files


def list_files_with_json(in_path):
    img_files = []
    gt_files = []
    for dirpath, dirnames, filenames in os.walk(in_path):
        for file in filenames:
            filename, ext = os.path.splitext(file)
            ext = str.lower(ext)
            if ext == ".jpg" or ext == ".jpeg" or ext == ".gif" or ext == ".png" or ext == ".pgm":
                gt_file = os.path.join(dirpath, filename + ".json")
                if os.path.exists(gt_file):
                    img_files.append(os.path.join(dirpath, file))
                    gt_files.append(gt_file)
                else:
                    print("No GT file fond for {}".format(gt_file))
                    continue
    return img_files, gt_files


def saveTraceResult(
    img_file,
    img,
    page,
    tables,
    dirname="./result/",
    saveXlsx=False,
):
    """save text detection result one by one
    Args:
        img_file (str): image file name
        img (array): raw image context
        tables (array): array of tables
    Return:
        None
    """
    img = np.array(img).astype(np.float32)

    # make result file list
    filename, file_ext = os.path.splitext(os.path.basename(img_file))

    # result directory
    page = "" if page is None else "-" + page
    res_file = dirname + "res_" + filename + page + ".txt"
    res_img_file = dirname + "res_" + filename + page + ".jpg"
    res_xls_file = dirname + "res_" + filename + page + ".xlsx"

    if not os.path.isdir(dirname):
        os.mkdir(dirname)

    if saveXlsx:
        # Create an new Excel file and add a worksheet.
        import xlsxwriter

        workbook = xlsxwriter.Workbook(res_xls_file)
        for tidx in range(len(tables)):
            worksheet = workbook.add_worksheet("table_" + str(tables[tidx]["id"]))
            cells = tables[tidx]["cells"]

            # Add cell contents
            for i, cell in enumerate(cells):
                # save cell in sheet
                si, ei = cell["col_range"]
                if si > 25:
                    si = chr((si + 1) // 26 + 64) + chr(si % 26 + 65)
                else:
                    si = chr(si + 65)
                if ei > 25:
                    ei = chr((ei + 1) // 26 + 64) + chr(ei % 26 + 65)
                else:
                    ei = chr(ei + 65)
                sj, ej = cell["row_range"]
                sj, ej = str(sj + 1), str(ej + 1)
                cell_name = si + sj + ":" + ei + ej

                if si == ei and sj == ej:  # single cell
                    worksheet.write(si + sj, cell_name)
                else:  # merge cell
                    worksheet.merge_range(cell_name, cell_name)
        workbook.close()

    # Render result image
    fill_img = img.copy()
    for tidx in range(len(tables)):
        # # draw table rect
        # poly = np.array(tables[tidx]['quad']).astype(np.int32).reshape((-1))
        # cv2.polylines(img, [poly.reshape((-1, 1, 2))], True, color=(255, 0, 0), thickness=10)

        # draw cells
        cells = tables[tidx]["cells"]

        for cell in cells:
            # render img
            box = cell["quad"]
            poly = np.array(box).astype(np.int32).reshape((-1))

            poly = poly.reshape(-1, 2)
            cv2.polylines(
                img,
                [poly.reshape((-1, 1, 2))],
                True,
                color=(0, 0, 255),
                thickness=2,
            )
            cv2.fillPoly(fill_img, [poly], (0, 128, 0), 8)

    cv2.addWeighted(fill_img, 0.2, img, 0.8, 0, img)

    # Save result image
    cv2.imwrite(res_img_file, img)

    return img[:, :, ::-1].astype(np.uint8), res_xls_file



def change_permissions_recursive(path, mode):
    # example : change_permissions_recursive('my_folder', 0o777)
    for root, dirs, files in os.walk(path, topdown=False):
        for dir in [os.path.join(root, d) for d in dirs]:
            os.chmod(dir, mode)
    for file in [os.path.join(root, f) for f in files]:
        os.chmod(file, mode)
