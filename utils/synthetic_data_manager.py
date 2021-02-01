import json
import os
import urllib.request

import pandas as pd
import numpy as np
from skimage import io
import cv2
from PIL import Image
import csv


# resizes the images
def data_resize(csv_loc, new_csv_loc, image_dir):
    df = pd.read_csv(csv_loc, header=None)
    df.columns.name = None
    for index, row in df.iterrows():
        image_name = row[0]
        bbox = np.array(row[1:])
        image = io.imread(os.path.join(image_dir, image_name))
        # numpy is like y * x which is a bit odd for this use but whatevs
        x_trans = 256 / image.shape[1]
        y_trans = 256 / image.shape[0]
        bbox[0] *= x_trans
        bbox[2] *= x_trans
        bbox[1] *= y_trans
        bbox[3] *= y_trans
        df.loc[index:index, 1:4] = bbox[0], bbox[1], bbox[2], bbox[3]
        resized_image = cv2.resize(image, (256, 256))
        cv2.imwrite("data/synthetic_data/resized_images/" + image_name, resized_image)
    df.to_csv(new_csv_loc, header=False, index=False)


def txt_to_csv(txt_loc, csv_loc):
    file = open(txt_loc)
    csv_file = csv_loc
    with open(csv_file, "w") as csvfile:
        writer = csv.DictWriter(
            csvfile, fieldnames=["name", "x", "y", "width", "height"]
        )
        for line in file:
            first_comma_idx = line.find(",")
            name = {"name": str(line[0:first_comma_idx]) + ".png"}
            rect_dict = line[first_comma_idx + 1 :].replace("(", "{").replace(")", "}")
            print(rect_dict)
            rect_dict = json.loads(rect_dict)
            name.update(rect_dict)
            writer.writerow(name)


def validate_csv_img(img_dir, csv_inp, csv_out):
    with open(csv_inp) as csvinp, open(csv_out, "w") as csvout:
        readCSV = csv.reader(csvinp, delimiter=",")
        writeCSV = csv.writer(csvout)
        for row in readCSV:
            print(row)
            if float(row[1]) > 1363 or float(row[1]) < 0 or float(row[2]) > 752 or float(row[2]) < 0:
                try:
                    os.remove(img_dir + row[0])
                    continue
                except:
                    continue
            else:
                writeCSV.writerow(row)


# txt_to_csv("data/synthetic_data/annotations.txt", "data/synthetic_data/annotations.csv")
# validate_csv_img(
#     "data/synthetic_data/images/",
#     "data/synthetic_data/annotations.csv",
#     "data/synthetic_data/fixed_annotations.csv",
# )
data_resize("data/synthetic_data/annotations.csv", "data/synthetic_data/resized_annotations.csv", "data/synthetic_data/images/")