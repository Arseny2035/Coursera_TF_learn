import tensorflow as tf


a = tf.Variable(1.0)
b = tf.Variable(2.0)

@tf.function
def f(x,y):
    a.assign(y * b)
    b.assign_add(x * a)
    return a + b

print(f(1.0, 2.0))

print(tf.autograph.to_code(f.python_function))

###################################

@tf.function
def add(a, b):
    return a + b


a = tf.Variable([[1., 2.], [3., 4.]])
b = tf.Variable([[4., 0.], [1., 5.]])
print(tf.add(a, b))

print(tf.autograph.to_code(add.python_function))

###################################
@tf.function
def f(x):
    if x > 0:
        x = x * x
    return x


print(tf.autograph.to_code(f.python_function))

###################################
@tf.function
def fizzbuzz(max_num):
    counter = 0
    for num in range(max_num):
        if num % 3 == 0 and num % 5 == 0:
            print('FizzBuzz')
        elif num % 3 == 0:
            print('Fizz')
        elif num % 5 == 0:
            print('Buzz')
        else:
            print(num)
        counter += 1
    return counter


print(tf.autograph.to_code(fizzbuzz.python_function))

###################################
@tf.function
def sign(x):
    if x > 0:
        return 'Positive'
    else:
        return 'Negative or zero'

print("Sign = {}".format(sign(tf.constant(2))))
print("Sign = {}".format(sign(tf.constant(-2))))

print(tf.autograph.to_code(sign.python_function))

###################################
@tf.function
def f(x):
    while tf.reduce_sum(x) > 1:
        tf.print(x)
        x = tf.tanh(x)
    return x

print(tf.autograph.to_code(f.python_function))


###################################
@tf.function
def sum_even(items):
    s = 0
    for c in items:
        if c % 2 > 0:
            continue
        s += c
    return s

print(tf.autograph.to_code(sum_even.python_function))