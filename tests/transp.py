import tensorflow as tf
import numpy as np


def print_grad(label, val, grad):
    print("\n", label, ":\n", "val:\n", val.numpy(), "\ngrad:\n", grad.numpy())


a_elements = np.arange(30).reshape(3, 5, 2).astype(np.float32)
a = tf.Variable(a_elements)

with tf.GradientTape() as tape:

    c = tf.transpose(a)

g = tape.gradient(c, [a, c])

print_grad("A", a, g[0])
print_grad("C", c, g[1])
