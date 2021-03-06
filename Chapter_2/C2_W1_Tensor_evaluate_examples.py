import tensorflow as tf
import numpy as np

# a = tf.constant([[5, 7],
#                  [2, 1]])
#
# b = tf.add(a, 2)
#
# c = b ** 2
#
# d = tf.reduce_sum(c)
#
# print(d)

##########################

#
# w = tf.Variable([[1.0]])
# with tf.GradientTape() as tape:
#     loss = w * w
#
# print(tape.gradient(loss, w))


##########################

#
# tf.constant([-1, -1, -1, -1, -1, -1], shape=[2, 3])
#
#
# print(w )
#
# w = w * 2
# print(w)


##########################
#
# x = tf.ones(2, 2)
# print('x = tf.ones(2, 2)\n', x)
# x = x + 1
# print('x = x + 1 \n', x)
# with tf.GradientTape(persistent=True) as t:
#     t.watch(x)
#     y = tf.reduce_sum(x)
#     z = tf.square(y)
#
# dz_dx = t.gradient(z, x)
# dy_dx = t.gradient(y, x)
#
# print('t.gradient(z, x) : ', dz_dx)
# print('t.gradient(y, x) : ', dy_dx)

###########################################################
a = tf.constant([[1, 2, 3],
                 [4, 5, 6]])
b = tf.constant([[7, 8],
                 [9, 10],
                 [11, 12]])

print('tf.matmul(a, b)', tf.matmul(a, b))

print(tf.multiply(a, 3))
print(tf.multiply([2, 3], 5))


############################
# x = tf.constant(3.0)
# print(x)
# x = x + 1
# print(x)
# with tf.GradientTape(persistent=True) as t:
#     t.watch(x)
#     y = x * x
#     print(y)
#     z = y * y
#     print(z)
#
# dz_dx = t.gradient(z, x)
# print('dz_dx: ', dz_dx)
# dy_dx = t.gradient(y, x)
# print('dy_dx: ', dy_dx)
# del t
#
# print(dz_dx)
# print(dy_dx)

############################
# x = tf.Variable(5.0)
#
# with tf.GradientTape() as tape_2:
#     with tf.GradientTape() as tape_1:
#         y = x * x * x
#     dy_dx = tape_1.gradient(y, x)
# d2y_dx2 = tape_2.gradient(dy_dx, x)
#
# print('dy_dx: ', dy_dx)
# print('d2y_dx2 :', d2y_dx2)
#
# assert dy_dx.numpy() == 3.0
# assert d2y_dx2.numpy() == 6.0

# ############################
# a = np.array([1, 2, 3])
# print(a)
# tf_constant_array = tf.constant(a)
# ### END CODE HERE ###
# print(tf_constant_array)

# ############################
def tf_gradient_tape(x):
    """
    Args:
        x (EagerTensor): a tensor.

    Returns:
        EagerTensor: Derivative of z with respect to the input tensor x.
    """
    with tf.GradientTape() as t:
        ### START CODE HERE ###
        # Record the actions performed on tensor x with `watch`
        t.watch(x)

        # Define a polynomial of form 3x^3 - 2x^2 + x
        y = 3 * x * x * x - 2 * x * x + x
        print(y)

        # Obtain the sum of the elements in variable y
        z = tf.reduce_sum(y)
        print(z)

    # Get the derivative of z with respect to the original input tensor x
    dz_dx = t.gradient(z, x)
    print(dz_dx)
    ### END CODE HERE

    return dz_dx

# Check your function
# tmp_x = tf.constant(2.0)
# dz_dx = tf_gradient_tape(tmp_x)
# result = dz_dx.numpy()
# print(result)

# j = 5
# vector = tf.constant(0.0, shape=(4,))
# print(vector)
# vector = tf.constant(0.0, shape=[2, 3])
# print(vector)


#########################################################################
# x = tf.constant([[1, 2, 3], [4, 5, 6]])
# print(x.numpy())
# print('tf.reduce_sum(x).numpy()\n', tf.reduce_sum(x).numpy())
# print('tf.reduce_sum(x, 0).numpy()\n', tf.reduce_sum(x, 0).numpy())
# print('tf.reduce_sum(x, 1).numpy()\n', tf.reduce_sum(x, 1).numpy())
# print('tf.reduce_sum(x, 1, keepdims=True).numpy()\n',
#       tf.reduce_sum(x, 1, keepdims=True).numpy())
# print('tf.add([1, 2], [3, 4]).numpy()\n', tf.add([1, 2], [3, 5]).numpy())
# print('tf.square(5).numpy())\n', tf.square(5).numpy())
