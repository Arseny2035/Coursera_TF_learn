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


# def func(str):
#     print(str)
#     tf.print(str)
#
# for i in range(3):
#     func(i)

# import tensorflow as tf
#
# class Net(tf.keras.Model):
#   """A simple linear model."""
#
#   def __init__(self):
#     super(Net, self).__init__()
#     self.l1 = tf.keras.layers.Dense(5)
#
#   def call(self, x):
#     return self.l1(x)
#
# net = Net()
#
# net.save_weights('easy_checkpoint')
#
#
# def toy_dataset():
#   inputs = tf.range(10.)[:, None]
#   labels = inputs * 5. + tf.range(5.)[None, :]
#   return tf.data.Dataset.from_tensor_slices(
#     dict(x=inputs, y=labels)).repeat().batch(2)
#
# # dic = toy_dataset()
# # print("dic: ", dic)
#
#
# def train_step(net, example, optimizer):
#   """Trains `net` on `example` using `optimizer`."""
#   with tf.GradientTape() as tape:
#     output = net(example['x'])
#     loss = tf.reduce_mean(tf.abs(output - example['y']))
#   variables = net.trainable_variables
#   gradients = tape.gradient(loss, variables)
#   optimizer.apply_gradients(zip(gradients, variables))
#   return loss
#
#
# opt = tf.keras.optimizers.Adam(0.1)
# dataset = toy_dataset()gt_boxes_list
# iterator = iter(dataset)
# print("iterator: ", iterator)
# ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=opt, net=net, iterator=iterator)
# manager = tf.train.CheckpointManager(ckpt, './tf_ckpts', max_to_keep=3)
#
#
# def train_and_checkpoint(net, manager):
#   ckpt.restore(manager.latest_checkpoint)
#   if manager.latest_checkpoint:
#     print("Restored from {}".format(manager.latest_checkpoint))
#   else:
#     print("Initializing from scratch.")
#
#   for _ in range(50):
#     example = next(iterator)
#     loss = train_step(net, example, opt)
#     ckpt.step.assign_add(1)
#     if int(ckpt.step) % 10 == 0:
#       save_path = manager.save()
#       print("Saved checkpoint for step {}: {}".format(int(ckpt.step), save_path))
#       print("loss {:1.2f}".format(loss.numpy()))
#
#
# train_and_checkpoint(net, manager)
#
# opt = tf.keras.optimizers.Adam(0.1)
# net = Net()
# dataset = toy_dataset()
# iterator = iter(dataset)
# ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=opt, net=net, iterator=iterator)
# manager = tf.train.CheckpointManager(ckpt, './tf_ckpts', max_to_keep=3)
#
# train_and_checkpoint(net, manager)
#
# print("manager.checkpoints: ", manager.checkpoints)
#
#
# to_restore = tf.Variable(tf.zeros([5]))
# print("to_restore.numpy() :", to_restore.numpy())  # All zeros
# fake_layer = tf.train.Checkpoint(bias=to_restore)
# fake_net = tf.train.Checkpoint(l1=fake_layer)
# new_root = tf.train.Checkpoint(net=fake_net)
# status = new_root.restore(tf.train.latest_checkpoint('./tf_ckpts/'))
# print("to_restore.numpy() :", to_restore.numpy())   # This gets the restored value.

# def twoSum(nums: list[int], target: int) -> list[int]:
#     nums_set = set(nums)
#     final = []
#     print(nums_set)
#     for i in range(len(nums)):
#         difference = target - nums[i]
#         if difference in nums_set:
#             final.append(i)
#             try:
#                 final.append(nums.index(difference, i+1))
#                 break
#             except:
#                 final.clear()
#
#     return final

# def twoSum(nums: list[int], target: int) -> list[int]:
#     # nums_set = set(nums)
#     final = []
#     # print(nums)
#     # print(nums_set)
#     for a in range(len(nums)):
#         # print(a)
#         difference = target - nums[a]
#         print('difference:', difference)
#         try:
#             b = nums.index(difference, a + 1)
#             final.append(a)
#             final.append(b)
#
#         except:
#             pass
#
#
#         return final
#
#     # new = set.add(nums[i])
#     # print(new)
#
#
# print(twoSum(nums=[0, 4, 3, 0], target=0))
# print(twoSum(nums=[5, 7, 2, 15], target=9))
# print(twoSum(nums=[-1,-2,-3,-4,-5], target=-8))


# A = [1, 2, 3, 4, 5 ,6, 7, 8, 9, 9]
#
# B = ['c', 'a', 'b', 'c', 'a', 'b', 'c', 'a', 'b', 'c']
#
# summary = dict()
# for i in range(len(A)):
#     if B[i] not in summary:
#         summary[B[i]] = 0
#     summary[B[i]] += A[i]
#
# print(summary)

