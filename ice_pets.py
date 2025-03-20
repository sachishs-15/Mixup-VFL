import os 
import xmltodict
from pdb import set_trace
import cv2
import numpy as np

def modify_bbox_coordinates(bndbox, original_image_size, new_image_size):

    bndbox['xmin'] = int(int(bndbox['xmin']) * new_image_size[1] / original_image_size[1])
    bndbox['xmax'] = int(int(bndbox['xmax']) * new_image_size[1] / original_image_size[1])
    bndbox['ymin'] = int(int(bndbox['ymin']) * new_image_size[0] / original_image_size[0])
    bndbox['ymax'] = int(int(bndbox['ymax']) * new_image_size[0] / original_image_size[0])

    return bndbox

def fetch_ice_pets():
    
    images_path = 'Datasets/ice_pets/images/'
    labels_path = 'Datasets/ice_pets/annotations/xmls/'

    images = []
    labels = []
    image_size = (400, 600)


    #sorted list of images
    for filename in sorted(os.listdir(labels_path)):
        # print(filename)

        image_name = filename.split('.')[0] + '.jpg'
        # open image and save it to a list
        image_np = cv2.imread(images_path + image_name)
        # if corrupt image
        if image_np is None:
            continue
        image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
        original_image_size = image_np.shape
        image_np = cv2.resize(image_np, image_size)
        image_np = image_np.flatten()
        # add to list
        images.append(image_np)

        with open(labels_path + filename) as xml_file:
            data_dict = xmltodict.parse(xml_file.read())

            if isinstance(data_dict['annotation']['object'], list):
                bndbox = data_dict['annotation']['object'][0]['bndbox']
            else:
                bndbox = data_dict['annotation']['object']['bndbox']

            bndbox = modify_bbox_coordinates(bndbox, original_image_size, image_size)
            label = [int(bndbox['xmin']), int(bndbox['ymin']), int(bndbox['xmax']), int(bndbox['ymax'])]
            labels.append(label)

    images = np.vstack(images)
    labels = np.array(labels)
    
    return images, labels, image_size

if __name__ == "__main__":
    fetch_ice_pets()