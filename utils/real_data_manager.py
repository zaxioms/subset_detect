import json
import os
import urllib.request

import pandas as pd
import numpy as np
from skimage import io
import cv2
from PIL import Image


def load_json(path):
    "Loads JSON as a Python object from a given file path"
    with open(path) as f:
        data = json.loads(f.read())
    return data


# grabs COCO data of a specific ID
def data_grab(instance_json_loc, custom_json_loc, image_dir, idee):
    for filename in os.listdir(instance_json_loc):
        print(filename)
        annotations = load_json(os.path.join(instance_json_loc, filename))
        image_ids = []
        custom_ann = {}
        for ann in annotations["annotations"]:
            if ann["category_id"] == idee:
                image_ids.append(ann["image_id"])
                custom_ann[str(ann["image_id"]) + ".jpg"] = ann["bbox"]

        with open(os.path.join(custom_json_loc, filename), "w") as f:
            json.dump(custom_ann, f)
            print("Wrote JSON")

        for image in annotations["images"]:
            if image["id"] in image_ids:
                urllib.request.urlretrieve(
                    image["coco_url"], image_dir + str(image["id"]) + ".jpg"
                )


# iterates over the custom jsons and removes duplicates
def data_mng(custom_json_loc, image_dir):
    definitive_json = {}
    for filename in os.listdir(custom_json_loc):
        annotations = load_json(os.path.join(custom_json_loc, filename))
        for annotation in annotations:
            if annotation not in definitive_json:
                definitive_json[annotation] = annotations[annotation]

    with open(os.path.join(custom_json_loc, "definitive.json"), "w") as f:
        json.dump(definitive_json, f)
        print("Wrote JSON")


def json_to_csv(json_loc, csv_loc, header=False):
    df = pd.read_json(json_loc).transpose()
    df.to_csv(csv_loc, header=header, index=False)


# asserts that for every file there is a json entry
def assert_data(custom_json_loc, image_dir):
    valid = True
    assert os.path.isfile(
        os.path.join(custom_json_loc, "definitive.json")
    ), "definitive.json not found"
    definitive_json = load_json(os.path.join(custom_json_loc, "definitive.json"))
    for filename in os.listdir(image_dir):
        if filename not in definitive_json.keys():
            print("file missing: ", filename)
            valid = False
    if valid:
        print("All files asserted and valid!")


# resizes the images
def data_resize(csv_loc, new_csv_loc, image_dir):
    df = pd.read_csv(csv_loc, header=None)
    df.columns.name = None
    for index, row in df.iterrows():
        print(row)
        image_name = row[0]
        bbox = np.array(row[1:])
        image = io.imread(os.path.join(image_dir, image_name))
        print(image.shape)
        # numpy is like y * x which is a bit odd for this use but whatevs
        x_trans = 256 / image.shape[1]
        y_trans = 256 / image.shape[0]
        bbox[0] *= x_trans
        bbox[2] *= x_trans
        bbox[1] *= y_trans
        bbox[3] *= y_trans
        df.loc[index:index, 1:4] = bbox[0], bbox[1], bbox[2], bbox[3]
        resized_image = cv2.resize(image, (256, 256))
        cv2.imwrite("data/resized_images/" + image_name, resized_image)
    df.to_csv(new_csv_loc, header=False, index=False)


def grey_to_rgb(image_loc):
    def is_grey_scale(img_path):
        img = Image.open(img_path).convert("RGB")
        w, h = img.size
        for i in range(w):
            for j in range(h):
                r, g, b = img.getpixel((i, j))
                if r != g != b:
                    return False
        return True

    for filename in os.listdir(image_loc):
        if is_grey_scale(os.path.join(image_loc, filename)):
            img = cv2.imread(os.path.join(image_loc, filename))
            print(img.shape)
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            cv2.imwrite(os.path.join(image_loc, filename), img)


if __name__ == "__main__":
    # data_grab("data/orig_instance_json/", "data/custom_json/", "data/images/", 37)
    # data_mng("data/custom_json/", "data/images/")
    # assert_data("data/custom_json/", "data/images/")

    # json_to_csv(
    #     "data/custom_json/definitive.json", "data/true_annotations/annotations.csv"
    # )

    # data_resize(
    #     "data/true_annotations/annotations.csv",
    #     "data/true_annotations/resized_annotations.csv",
    #     "data/orig_images",
    # )

    # grey_to_rgb("data/resized_images")
    pass
