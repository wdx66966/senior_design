import os
import glob
import random
import numpy as np
import cv2
import caffe
from caffe.proto import caffe_pb2
import lmdb

#Size of image
IMAGE_WIDTH = 227
IMAGE_HEIGHT = 227


def transform_img(img, img_width=IMAGE_WIDTH, img_height=IMAGE_HEIGHT):

    #Histogram Equalization
    
    #Image Resizing
    gray = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
    gray[:, :, 0] = cv2.equalizeHist(gray[:, :, 0])
    gray[:, :, 1] = cv2.equalizeHist(gray[:, :, 1])
    gray[:, :, 2] = cv2.equalizeHist(gray[:, :, 2])
    img2 = cv2.resize(gray, (img_width, img_height), interpolation = cv2.INTER_CUBIC)

    return img2
def make_datum(img, label):
    #image is numpy.ndarray format. BGR instead of RGB
    return caffe_pb2.Datum(
        channels=3,
        width=IMAGE_WIDTH,
        height=IMAGE_HEIGHT,
        label=label,
        data=np.rollaxis(img, 2).tostring())

#train_lmdb = '/home/ubuntu/senior_design/FE/facial_expression/input/train_lmdb'
#validation_lmdb = '/home/ubuntu/senior_design/FE/facial_expression/input/validation_lmdb'
train_lmdb = '/home/ubuntu/senior_design/FE/CK+_JAFFE_KDEF/input/train_lmdb'
validation_lmdb = '/home/ubuntu/senior_design/FE/CK+_JAFFE_KDEF/input/validation_lmdb'


os.system('rm -rf  ' + train_lmdb)
os.system('rm -rf  ' + validation_lmdb)

train_data = [img for img in glob.glob("CK+_JAFFE_KDEF/input/train/*/*")]
test_data = [img for img in glob.glob("CK+_JAFFE_KDEF/input/validation/*/*")]

#Shuffle train_data
random.shuffle(train_data)

print ("Creating train_lmdb")
x=0
in_db = lmdb.open(train_lmdb, map_size=int(1e12))
with in_db.begin(write=True) as in_txn:
    for in_idx, img_path in enumerate(train_data):
        img = cv2.imread(img_path,cv2.IMREAD_GRAYSCALE)
        print("image path:",img_path)
        img = transform_img(img, img_width=IMAGE_WIDTH, img_height=IMAGE_HEIGHT)
        if 'anger' in img_path:
            label = 0
        elif 'disgust' in img_path:
            label = 1
        elif 'fear' in img_path:
            label = 2
        elif 'happiness' in img_path:
            label = 3
        elif 'neutral' in img_path:
            label = 4
        elif 'sadness' in img_path:
            label = 5
        else:
            label = 6
        x=x+1
        datum = make_datum(img, label)
        in_txn.put('{:0>5d}'.format(in_idx).encode(), datum.SerializeToString())
in_db.close()
print (x)
print ("\nCreating validation_lmdb")

in_db = lmdb.open(validation_lmdb, map_size=int(1e12))
with in_db.begin(write=True) as in_txn:
    for in_idx, img_path in enumerate(test_data):
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        print("image path:",img_path)
        img = transform_img(img, img_width=IMAGE_WIDTH, img_height=IMAGE_HEIGHT)
        if 'anger' in img_path:
            label = 0
        elif 'disgust' in img_path:
            label = 1
        elif 'fear' in img_path:
            label = 2
        elif 'happy' in img_path:
            label = 3
        elif 'neutral' in img_path:
            label = 4
        elif 'sad' in img_path:
            label = 5
        else:
            label = 6
        datum = make_datum(img, label)
        in_txn.put('{:0>5d}'.format(in_idx).encode(), datum.SerializeToString())
in_db.close()

