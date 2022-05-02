import tensorflow as tf
import numpy as np

# def train_step(images, labels):
#     with tf.GradientTape() as tape:
#         logits = model(images, training=True)
#         loss_value = loss_object(labels, logits)
#
#     loss_history.append(loss_value.numpy().mean())
#     grads = tape.gradient(loss_value, model.trainable_variables)
#     optimizer.apply_gradients(zip(grads, model.trainable_variables))


gt_box_np=[
        np.array([[0.27333333, 0.41500586, 0.74333333, 0.57678781]]),
        np.array([[0.29833333, 0.45955451, 0.75666667, 0.61078546]]),
        np.array([[0.40833333, 0.18288394, 0.945, 0.34818288]]),
        np.array([[0.16166667, 0.61899179, 0.8, 0.91910903]]),
        np.array([[0.28833333, 0.12543962, 0.835, 0.35052755]]),
      ]



zero_indexed_groundtruth_classes = tf.convert_to_tensor(
    np.ones(shape=[gt_box_np.shape[0]], dtype=np.int32) - 3)

print(zero_indexed_groundtruth_classes)