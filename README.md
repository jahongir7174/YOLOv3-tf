[YOLOV3](https://pjreddie.com/media/files/papers/YOLOv3.pdf) implementation for Object Detection using TensorFlow 2

#### Train
* Change variables in `utils\config.py` based on your dataset
* Run `python generate.py` for generating tf record
* Run `python train.py` for training

#### Test
* Run `python test.py`

#### Dataset structure
    ├── Dataset folder 
        ├── IMAGES
            ├── 1111.jpg
            ├── 2222.jpg
        ├── LABELS
            ├── 1111.xml
            ├── 2222.xml
        ├── train.txt
        
#### Note 
* xml file should be in PascalVOC format
* for making `train.txt`, see `VOC2012/ImageSets/Main/train.txt` 

