import os
import sys
from os.path import exists
from os.path import join

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

import numpy as np

from nets import nn
from utils import util, config, data_loader

tf.random.set_seed(config.seed)

tf_path = join(config.data_dir, 'record')
tf_paths = [join(tf_path, name) for name in os.listdir(tf_path) if name.endswith('.tf')]

np.random.shuffle(tf_paths)

strategy = tf.distribute.MirroredStrategy()

nb_gpu = strategy.num_replicas_in_sync
global_batch = nb_gpu * config.batch_size
nb_classes = len(config.classes)

dataset = data_loader.TFRecordLoader(global_batch, config.epochs, nb_classes).load_data(tf_paths)
dataset = strategy.experimental_distribute_dataset(dataset)

with strategy.scope():
    optimizer = tf.keras.optimizers.RMSprop(1e-4)
    input_tensor = tf.keras.layers.Input([config.image_size, config.image_size, 3])
    outputs = nn.build_model(input_tensor, nb_classes)
    output_tensors = []
    for i, output in enumerate(outputs):
        pred_tensor = nn.decode(output, i, nb_classes)
        output_tensors.append(output)
        output_tensors.append(pred_tensor)
    model = tf.keras.Model(input_tensor, output_tensors)

print(f'[INFO] {len(tf_paths)} train data')

with strategy.scope():
    loss_object = nn.compute_loss


    def compute_loss(y_true, y_pred):
        iou_loss = conf_loss = prob_loss = 0

        for ind in range(3):
            loss_items = loss_object(y_pred[ind * 2 + 1], y_pred[ind * 2], y_true[ind][0], y_true[ind][1], ind)
            iou_loss += loss_items[0]
            conf_loss += loss_items[1]
            prob_loss += loss_items[2]

        iou_loss = tf.reduce_sum(iou_loss) * 1. / nb_gpu
        conf_loss = tf.reduce_sum(conf_loss) * 1. / nb_gpu
        prob_loss = tf.reduce_sum(prob_loss) * 1. / nb_gpu

        total_loss = iou_loss + conf_loss + prob_loss

        return iou_loss, conf_loss, prob_loss, total_loss

with strategy.scope():
    def train_step(image, y_true):
        with tf.GradientTape() as tape:
            y_pred = model(image, training=True)
            iou_loss, conf_loss, prob_loss, total_loss = compute_loss(y_true, y_pred)

            gradients = tape.gradient(total_loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            return iou_loss, conf_loss, prob_loss, total_loss

with strategy.scope():
    @tf.function
    def distribute_train_step(image, target):
        iou_loss, conf_loss, prob_loss, total_loss = strategy.run(train_step, args=(image, target))
        total_loss = strategy.reduce(tf.distribute.ReduceOp.SUM, total_loss, axis=None)
        iou_loss = strategy.reduce(tf.distribute.ReduceOp.SUM, iou_loss, axis=None)
        conf_loss = strategy.reduce(tf.distribute.ReduceOp.SUM, conf_loss, axis=None)
        prob_loss = strategy.reduce(tf.distribute.ReduceOp.SUM, prob_loss, axis=None)
        return iou_loss, conf_loss, prob_loss, total_loss


def main():
    steps = 1000
    print("--- Start Training ---")
    if not exists('weights'):
        os.makedirs('weights')
    for step, inputs in enumerate(dataset):
        step += 1
        image, s_label, s_boxes, m_label, m_boxes, l_label, l_boxes = inputs
        target = ((s_label, s_boxes), (m_label, m_boxes), (l_label, l_boxes))
        iou_loss, conf_loss, prob_loss, total_loss = distribute_train_step(image, target)
        print(f"{step} {total_loss.numpy():.4f}")
        if step % steps == 0:
            model.save_weights(join("weights", f"model{step // steps}.h5"))
        if step // steps == config.epochs:
            sys.exit("--- Stop Training ---")


if __name__ == '__main__':
    main()
