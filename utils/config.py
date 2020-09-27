import numpy as np
from os.path import join

seed = 12345
epochs = 50
batch_size = 16
max_boxes = 150
image_size = 512
data_dir = join('..', 'Dataset', 'VOC2012')
image_dir = 'IMAGES'
label_dir = 'LABELS'
classes = {'aeroplane': 0,
           'bicycle': 1,
           'bird': 2,
           'boat': 3,
           'bottle': 4,
           'bus': 5,
           'car': 6,
           'cat': 7,
           'chair': 8,
           'cow': 9,
           'diningtable': 10,
           'dog': 11,
           'horse': 12,
           'motorbike': 13,
           'person': 14,
           'pottedplant': 15,
           'sheep': 16,
           'sofa': 17,
           'train': 18,
           'tvmonitor': 19}
strides = [8, 16, 32]
anchors = np.array([[[12, 16], [19, 36], [40, 28]],
                    [[36, 75], [76, 55], [72, 146]],
                    [[142, 110], [192, 243], [459, 401]]], np.float32)
