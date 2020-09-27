import multiprocessing
import os
from multiprocessing import Process
from multiprocessing import cpu_count
from os.path import exists
from os.path import join
from xml.etree.ElementTree import ParseError
from xml.etree.ElementTree import parse as parse_fn

import cv2
import numpy as np
import tqdm
from six import raise_from

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from utils import util, config


def find_node(parent, name, debug_name=None, parse=None):
    if debug_name is None:
        debug_name = name

    result = parent.find(name)
    if result is None:
        raise ValueError('missing element \'{}\''.format(debug_name))
    if parse is not None:
        try:
            return parse(result.text)
        except ValueError as e:
            raise_from(ValueError('illegal value for \'{}\': {}'.format(debug_name, e)), None)
    return result


def name_to_label(name):
    return config.classes[name]


def load_image(f_name):
    path = join(config.data_dir, config.image_dir, f_name + '.jpg')
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def parse_annotation(element):
    truncated = find_node(element, 'truncated', parse=int)
    difficult = find_node(element, 'difficult', parse=int)

    class_name = find_node(element, 'name').text
    if class_name not in config.classes:
        raise ValueError('class name \'{}\' not found in classes: {}'.format(class_name, list(config.classes.keys())))

    label = config.classes[class_name]

    box = find_node(element, 'bndbox')
    x_min = find_node(box, 'xmin', 'bndbox.xmin', parse=int)
    y_min = find_node(box, 'ymin', 'bndbox.ymin', parse=int)
    x_max = find_node(box, 'xmax', 'bndbox.xmax', parse=int)
    y_max = find_node(box, 'ymax', 'bndbox.ymax', parse=int)

    return truncated, difficult, [x_min, y_min, x_max, y_max, label]


def parse_annotations(xml_root):
    annotations = []
    for i, element in enumerate(xml_root.iter('object')):
        truncated, difficult, box = parse_annotation(element)

        annotations.append(box)

    return np.array(annotations)


def load_label(f_name):
    try:
        tree = parse_fn(join(config.data_dir, config.label_dir, f_name + '.xml'))
        return parse_annotations(tree.getroot())
    except ParseError as error:
        raise_from(ValueError('invalid annotations file: {}: {}'.format(f_name, error)), None)
    except ValueError as error:
        raise_from(ValueError('invalid annotations file: {}: {}'.format(f_name, error)), None)


def byte_feature(value):
    import tensorflow as tf
    if not isinstance(value, bytes):
        if not isinstance(value, list):
            value = value.encode('utf-8')
        else:
            value = [val.encode('utf-8') for val in value]
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def bbox_iou(boxes1, boxes2):
    boxes1 = np.array(boxes1)
    boxes2 = np.array(boxes2)

    boxes1_area = boxes1[..., 2] * boxes1[..., 3]
    boxes2_area = boxes2[..., 2] * boxes2[..., 3]

    boxes1 = np.concatenate([boxes1[..., :2] - boxes1[..., 2:] * 0.5,
                             boxes1[..., :2] + boxes1[..., 2:] * 0.5, ], axis=-1, )
    boxes2 = np.concatenate([boxes2[..., :2] - boxes2[..., 2:] * 0.5,
                             boxes2[..., :2] + boxes2[..., 2:] * 0.5, ], axis=-1, )

    left_up = np.maximum(boxes1[..., :2], boxes2[..., :2])
    right_down = np.minimum(boxes1[..., 2:], boxes2[..., 2:])

    inter_section = np.maximum(right_down - left_up, 0.0)
    inter_area = inter_section[..., 0] * inter_section[..., 1]
    union_area = boxes1_area + boxes2_area - inter_area

    return inter_area / union_area


def preprocess(boxes):
    num_classes = len(config.classes)
    train_output_sizes = [64, 32, 16]
    label = [np.zeros((train_output_sizes[i], train_output_sizes[i], 3, 5 + num_classes)) for i in range(3)]
    boxes_xywh = [np.zeros((config.max_boxes, 4)) for _ in range(3)]
    bbox_count = np.zeros((3,))

    for box in boxes:
        box_coordinate = box[:4]
        box_class_ind = box[4]

        one_hot = np.zeros(num_classes, dtype=np.float)
        one_hot[box_class_ind] = 1.0
        uniform_distribution = np.full(num_classes, 1.0 / num_classes)
        delta = 0.01
        smooth_one_hot = one_hot * (1 - delta) + delta * uniform_distribution

        box_xywh = np.concatenate([(box_coordinate[2:] + box_coordinate[:2]) * 0.5,
                                   box_coordinate[2:] - box_coordinate[:2]], axis=-1)
        box_xywh_scaled = 1.0 * box_xywh[np.newaxis, :] / np.array(config.strides)[:, np.newaxis]

        iou = []
        exist_positive = False
        for i in range(3):
            anchors_xywh = np.zeros((3, 4))
            anchors_xywh[:, 0:2] = np.floor(box_xywh_scaled[i, 0:2]).astype(np.int32) + 0.5
            anchors_xywh[:, 2:4] = config.anchors[i]

            iou_scale = bbox_iou(box_xywh_scaled[i][np.newaxis, :], anchors_xywh)
            iou.append(iou_scale)
            iou_mask = iou_scale > 0.3

            if np.any(iou_mask):
                x_ind, y_ind = np.floor(box_xywh_scaled[i, 0:2]).astype(np.int32)

                label[i][y_ind, x_ind, iou_mask, :] = 0
                label[i][y_ind, x_ind, iou_mask, 0:4] = box_xywh
                label[i][y_ind, x_ind, iou_mask, 4:5] = 1.0
                label[i][y_ind, x_ind, iou_mask, 5:] = smooth_one_hot

                bbox_ind = int(bbox_count[i] % config.max_boxes)
                boxes_xywh[i][bbox_ind, :4] = box_xywh
                bbox_count[i] += 1

                exist_positive = True

        if not exist_positive:
            best_anchor_ind = np.argmax(np.array(iou).reshape(-1), axis=-1)
            best_detect = int(best_anchor_ind / 3)
            best_anchor = int(best_anchor_ind % 3)
            x_ind, y_ind = np.floor(box_xywh_scaled[best_detect, 0:2]).astype(np.int32)

            label[best_detect][y_ind, x_ind, best_anchor, :] = 0
            label[best_detect][y_ind, x_ind, best_anchor, 0:4] = box_xywh
            label[best_detect][y_ind, x_ind, best_anchor, 4:5] = 1.0
            label[best_detect][y_ind, x_ind, best_anchor, 5:] = smooth_one_hot

            bbox_ind = int(bbox_count[best_detect] % config.max_boxes)
            boxes_xywh[best_detect][bbox_ind, :4] = box_xywh
            bbox_count[best_detect] += 1
    l_s_box, l_m_box, l_l_box = label
    s_boxes, m_boxes, l_boxes = boxes_xywh
    return l_s_box, l_m_box, l_l_box, s_boxes, m_boxes, l_boxes


def build_example(f_name):
    import tensorflow as tf
    image = load_image(f_name)
    label = load_label(f_name)
    image, label = util.resize(image, label)
    s_label, m_label, l_label, s_boxes, m_boxes, l_boxes = preprocess(label)

    path = join(config.data_dir, 'record', f_name + '.jpg')

    util.write_image(path, image)

    s_label = s_label.astype('float32')
    m_label = m_label.astype('float32')
    l_label = l_label.astype('float32')
    s_boxes = s_boxes.astype('float32')
    m_boxes = m_boxes.astype('float32')
    l_boxes = l_boxes.astype('float32')

    s_label = s_label.tobytes()
    m_label = m_label.tobytes()
    l_label = l_label.tobytes()

    s_boxes = s_boxes.tobytes()
    m_boxes = m_boxes.tobytes()
    l_boxes = l_boxes.tobytes()

    features = tf.train.Features(feature={'path': byte_feature(path.encode('utf-8')),
                                          's_label': byte_feature(s_label),
                                          'm_label': byte_feature(m_label),
                                          'l_label': byte_feature(l_label),
                                          's_boxes': byte_feature(s_boxes),
                                          'm_boxes': byte_feature(m_boxes),
                                          'l_boxes': byte_feature(l_boxes)})

    return tf.train.Example(features=features)


def write_tf_record(_queue, _sentinel):
    import tensorflow as tf
    while True:
        f_name = _queue.get()

        if f_name == _sentinel:
            break
        tf_example = build_example(f_name)
        if not exists(join(config.data_dir, 'record')):
            os.makedirs(join(config.data_dir, 'record'))
        with tf.io.TFRecordWriter(join(config.data_dir, 'record', f_name + ".tf")) as writer:
            writer.write(tf_example.SerializeToString())


def main():
    f_names = []
    with open(join(config.data_dir, 'train.txt')) as reader:
        for line in reader.readlines():
            f_names.append(line.rstrip().split(' ')[0])
    sentinel = ("", [])
    queue = multiprocessing.Manager().Queue()
    for f_name in tqdm.tqdm(f_names):
        queue.put(f_name)
    for _ in range(cpu_count()):
        queue.put(sentinel)
    print('[INFO] generating TF record')
    process_pool = []
    for i in range(cpu_count()):
        process = Process(target=write_tf_record, args=(queue, sentinel))
        process_pool.append(process)
        process.start()
    for process in process_pool:
        process.join()


if __name__ == '__main__':
    main()
