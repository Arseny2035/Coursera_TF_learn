import tensorflow as tf
#
#
# # m = tf.keras.metrics.SparseCategoricalAccuracy()
# # m.update_state([[2], [1]], [[0.1, 0.6, 0.3], [0.05, 0.95, 0]])
# # print(m.result().numpy())
#
# #
# # values = ["a", "b", "c"]
# # e = enumerate(values)
# # print(list(e)[1])
#
#
# import tensorflow as tf
# print(tf.test.is_gpu_available(cuda_only=False, min_cuda_compute_capability=None))


def func(str):
    print(str)
    tf.print(str)

for i in range(3):
    func(i)