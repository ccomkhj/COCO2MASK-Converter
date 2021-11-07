from pycocotools.coco import COCO
from PIL import Image
import numpy as np
import os
from matplotlib import pyplot as plt

__title__ = 'Annotation Converter From COCO to (MMsegmentation friendly) Mask'
__version__ = '1.0.0'
__author__ = 'Kim, Huijo'

def converter(image_id , coco, img_dir, input_show = False, ann_show = False):
    
    img = coco.imgs[image_id]

    if input_show:
        ''' Show the input image '''
        image = np.array(Image.open(os.path.join(img_dir, img['file_name'])))
        plt.imshow(image, interpolation='nearest')
        plt.show()

    cat_ids = coco.getCatIds()
    anns_ids = coco.getAnnIds(imgIds=img['id'], catIds=cat_ids, iscrowd=None)
    anns = coco.loadAnns(anns_ids)

    ''' Draw the annotation mask '''
    mask = np.zeros((img['height'],img['width']))
    for ann in anns:
        mask = np.maximum(mask,coco.annToMask(ann)*ann['category_id'])

    if ann_show:
        ''' Show the annotation mask '''
        plt.imshow(mask, cmap='Accent')
        plt.show()

    ''' Save the annotation mask '''
    mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    mask.save('output'+r'/'+coco.imgs[image_id]['path'])

if __name__ == '__main__':
    loc_ann = "input/annotations/annotations.json" # location of annotation
    img_dir = "input/images" # location of image folder
    input_show = False
    ann_show = False

    coco = COCO(loc_ann)
    for i in list(coco.imgs.values()):
        image_id = i['id']
        converter(image_id, coco, img_dir)