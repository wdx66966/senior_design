import os
import glob
import random
import numpy as np
import cv2
import caffe
from caffe.proto import caffe_pb2
import lmdb
import dlib
from imutils import face_utils
from scipy.spatial import distance as dist
import imutils
import numpy as np
import os

print("[INFO] loading model...")
prototxt1="/home/ubuntu/senior_design/FE/deploy.prototxt"
model1="/home/ubuntu/senior_design/FE/res10_300x300_ssd_iter_140000.caffemodel"
net1 = cv2.dnn.readNetFromCaffe(prototxt1,model1)

#Size of image
IMAGE_WIDTH = 227
IMAGE_HEIGHT = 227


def transform_img(img, img_width=IMAGE_WIDTH, img_height=IMAGE_HEIGHT):

    #Histogram Equalization
    
    #Image Resizing
    gray = cv2.cvtColor(img,cv2.cv2.COLOR_BGR2GRAY)
    img2 = np.zeros_like(img)
    img2[:, :, 0] = gray
    img2[:, :, 1] = gray
    img2[:, :, 2] = gray
    img2 = cv2.resize(img2, (img_width, img_height), interpolation = cv2.INTER_CUBIC)
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
        img = cv2.imread(img_path)
        (h,w)=img.shape[:2]
        blob1=cv2.dnn.blobFromImage(cv2.resize(img,(300,300)),1.0,(300,300),(104.0,177.0,123.0))
        net1.setInput(blob1)
        detections = net1.forward()
        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence < 0.8:
                continue
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            print("image path:",img_path)
            img2 = img[startY:endY,startX:endX]
            img2 = transform_img(img2, img_width=IMAGE_WIDTH, img_height=IMAGE_HEIGHT)
            cv2.imwrite("/home/ubuntu/senior_design/FE/test_img/"+str(in_idex)+str(i)+".png", img2)
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
            datum = make_datum(img2, label)
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

