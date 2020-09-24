from pycocotools.coco import COCO
import requests
import os


def get_balls():
    for filename in os.listdir("./annotation_json"):
        coco = COCO('./annotation_json/' + filename)
        # Specify a list of category names of interest
        catIds = coco.getCatIds(catNms=['sports ball'])
        # Get the corresponding image ids and images using loadImgs
        imgIds = coco.getImgIds(catIds=catIds)
        print(imgIds)
        images = coco.loadImgs(imgIds)
        for im in images:
            img_data = requests.get(im['coco_url']).content
            with open('./data/images/' + im['file_name'], 'wb') as handler:
                handler.write(img_data)


get_balls()