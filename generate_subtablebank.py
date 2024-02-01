import argparse
import os
import shutil
from xml.dom import minidom

import numpy as np
import tqdm

from file_utils import get_image_list, get_xml_list

parser = argparse.ArgumentParser(description="Copy images from the original TableBank to create the SubTableBank dataset.")
parser.add_argument("-s", "--source_path", type=str, default="/data/db/table/TableBank_data")
parser.add_argument("-t", "--target_path", type=str, default="/data/db/table/SubTableBank")
args = parser.parse_args()

# get list of image files from source path and list of gt files from the target path
print("Generating file list...")
filelist_src = get_image_list(args.source_path)
imgfile_src = [os.path.basename(f) for f in filelist_src]
filelist_dst = get_xml_list(args.target_path)

if len(filelist_src) == 0 or len(filelist_dst) == 0:
    print("Please check the source or target paths.")
    exit(1)

print("Copying necessary files...")
matched_file = []
num_found = 0
for k, xml_file in enumerate(tqdm.tqdm(filelist_dst)):
    table_dom = minidom.parse(xml_file)
    image_path = table_dom.getElementsByTagName("document")[0].getAttribute("filename")
            
    # duplication check
    indices = np.where(image_path == np.array(imgfile_src))[0]
    if len(indices) == 0:
        raise FileNotFoundError
    if len(indices) > 1:
        raise NameError

    src_filename = filelist_src[indices[0]]
    tar_path = os.path.dirname(xml_file)
    shutil.copy(src_filename, tar_path)
    
print(f'samplig done...')
