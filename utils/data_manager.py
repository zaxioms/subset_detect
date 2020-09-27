import os
import json
import urllib.request


def load_json(path):
    "Loads JSON as a Python object from a given file path"
    with open(path) as f:
        data = json.loads(f.read())
    return data


def data_grab(instance_json_loc, custom_json_loc, image_dir, idee):
    for filename in os.listdir(instance_json_loc):
        print(filename)
        annotations = load_json(os.path.join(instance_json_loc, filename))
        image_ids = []
        custom_ann = {}
        for ann in annotations["annotations"]:
            if ann["category_id"] == idee:
                image_ids.append(ann["image_id"])
                custom_ann[ann["image_id"]] = ann["bbox"]

        with open(os.path.join(custom_json_loc, filename), "w") as f:
            json.dump(custom_ann, f)
            print("Wrote JSON")

        for image in annotations["images"]:
            if image["id"] in image_ids:
                urllib.request.urlretrieve(
                    image["coco_url"], image_dir + str(image["id"]) + ".jpg"
                )


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


def assert_data(custom_json_loc, image_dir):
    valid = True
    assert os.path.isfile(
        os.path.join(custom_json_loc, "definitive.json")
    ), "definitive.json not found"
    definitive_json = load_json(os.path.join(custom_json_loc, "definitive.json"))
    for filename in os.listdir(image_dir):
        if os.path.splitext(filename)[0] not in definitive_json.keys():
            print("file missing: ", filename)
            valid = False
    if valid:
        print("All files asserted and valid!")


# data_grab("data/orig_instance_json/", "data/custom_json/", "data/images/", 37)
# data_mng("data/custom_json/", "data/images/")
assert_data("data/custom_json/", "data/images/")
