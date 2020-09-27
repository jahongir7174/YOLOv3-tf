import colorsys
import random

import cv2
import numpy as np
import tensorflow as tf

import generate
from nets import nn
from utils import util, config


def postprocess_boxes(pred_bbox, org_img_shape, input_size, score_threshold):
    valid_scale = [0, np.inf]
    pred_bbox = np.array(pred_bbox)

    pred_xywh = pred_bbox[:, 0:4]
    pred_conf = pred_bbox[:, 4]
    pred_prob = pred_bbox[:, 5:]

    pred_coordinate = np.concatenate([pred_xywh[:, :2] - pred_xywh[:, 2:] * 0.5,
                                      pred_xywh[:, :2] + pred_xywh[:, 2:] * 0.5], axis=-1)
    org_h, org_w = org_img_shape
    resize_ratio = min(input_size / org_w, input_size / org_h)

    dw = (input_size - resize_ratio * org_w) / 2
    dh = (input_size - resize_ratio * org_h) / 2

    pred_coordinate[:, 0::2] = 1.0 * (pred_coordinate[:, 0::2] - dw) / resize_ratio
    pred_coordinate[:, 1::2] = 1.0 * (pred_coordinate[:, 1::2] - dh) / resize_ratio

    pred_coordinate = np.concatenate([np.maximum(pred_coordinate[:, :2], [0, 0]),
                                      np.minimum(pred_coordinate[:, 2:], [org_w - 1, org_h - 1])], axis=-1)
    invalid_mask = np.logical_or((pred_coordinate[:, 0] > pred_coordinate[:, 2]),
                                 (pred_coordinate[:, 1] > pred_coordinate[:, 3]))
    pred_coordinate[invalid_mask] = 0

    bbox_scale = np.sqrt(np.multiply.reduce(pred_coordinate[:, 2:4] - pred_coordinate[:, 0:2], axis=-1))
    scale_mask = np.logical_and((valid_scale[0] < bbox_scale), (bbox_scale < valid_scale[1]))

    classes = np.argmax(pred_prob, axis=-1)
    scores = pred_conf * pred_prob[np.arange(len(pred_coordinate)), classes]
    score_mask = scores > score_threshold
    mask = np.logical_and(scale_mask, score_mask)
    coors, scores, classes = pred_coordinate[mask], scores[mask], classes[mask]

    return np.concatenate([coors, scores[:, np.newaxis], classes[:, np.newaxis]], axis=-1)


def nms(boxes, iou_threshold=0.5, sigma=0.3, method='nms'):
    classes_in_img = list(set(boxes[:, 5]))
    best_boxes = []

    for cls in classes_in_img:
        cls_mask = (boxes[:, 5] == cls)
        cls_boxes = boxes[cls_mask]

        while len(cls_boxes) > 0:
            max_ind = np.argmax(cls_boxes[:, 4])
            best_bbox = cls_boxes[max_ind]
            best_boxes.append(best_bbox)
            cls_boxes = np.concatenate([cls_boxes[: max_ind], cls_boxes[max_ind + 1:]])
            iou = generate.bbox_iou(best_bbox[np.newaxis, :4], cls_boxes[:, :4])
            weight = np.ones((len(iou),), dtype=np.float32)

            assert method in ['nms', 'soft-nms']

            if method == 'nms':
                iou_mask = iou > iou_threshold
                weight[iou_mask] = 0.0

            if method == 'soft-nms':
                weight = np.exp(-(1.0 * iou ** 2 / sigma))

            cls_boxes[:, 4] = cls_boxes[:, 4] * weight
            score_mask = cls_boxes[:, 4] > 0.
            cls_boxes = cls_boxes[score_mask]

    return best_boxes


def draw_bbox(image, boxes, nb_classes, show_label=True):
    image_h, image_w, _ = image.shape
    hsv_tuples = [(1.0 * x / nb_classes, 1., 1.) for x in range(nb_classes)]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))

    random.seed(0)
    random.shuffle(colors)
    random.seed(None)

    for box in boxes:
        coordinate = np.array(box[:4], dtype=np.int32)
        font = 0.5
        class_ind = int(box[5])
        bbox_color = colors[class_ind]
        bbox_thick = int(0.6 * (image_h + image_w) / 600)
        c1, c2 = (coordinate[0], coordinate[1]), (coordinate[2], coordinate[3])
        cv2.rectangle(image, c1, c2, bbox_color, bbox_thick)

        if show_label:
            keys = list(config.classes.keys())
            text = keys[class_ind]
            t_size = cv2.getTextSize(text, 0, font, thickness=bbox_thick // 2)[0]
            cv2.rectangle(image, c1, (c1[0] + t_size[0], c1[1] - t_size[1] - 3), bbox_color, -1)  # filled

            cv2.putText(image, text, (c1[0], c1[1] - 2), cv2.FONT_HERSHEY_SIMPLEX,
                        font, (0, 0, 0), bbox_thick // 2, lineType=cv2.LINE_AA)

    return image


def main():
    nb_classes = len(config.classes)

    input_layer = tf.keras.layers.Input([config.image_size, config.image_size, 3])
    feature_maps = nn.build_model(input_layer, nb_classes)

    bbox_tensors = []
    for i, fm in enumerate(feature_maps):
        bbox_tensor = nn.decode(fm, i, nb_classes)
        bbox_tensors.append(bbox_tensor)

    model = tf.keras.Model(input_layer, bbox_tensors)
    model.load_weights("weights/model.h5")

    image = cv2.imread('../Dataset/VOC2012/IMAGES/2007_000039.jpg')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image_size = image.shape[:2]
    image_data = util.resize(image)

    pred_bbox = model.predict(image_data[np.newaxis, ...].astype(np.float32) / 255.0)
    pred_bbox = [tf.reshape(x, (-1, tf.shape(x)[-1])) for x in pred_bbox]
    pred_bbox = tf.concat(pred_bbox, axis=0)
    bbox = postprocess_boxes(pred_bbox, image_size, config.image_size, 0.3)
    bbox = nms(bbox)

    image = draw_bbox(image, bbox, nb_classes)
    cv2.imwrite('result.png', image)


if __name__ == '__main__':
    main()
