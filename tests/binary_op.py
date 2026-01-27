import tensorflow as tf
import numpy as np


def print_grad(label, val, grad):
    print("\n", label, ":\n", "val:\n", val.numpy(), "\ngrad:\n", grad.numpy())


a_data = np.full((3, 2), 10.0)
b_data = np.full((3, 2), 30.0)
c_data = np.full((1,), 50.0)
d_data = np.full((2, 3, 1), 20.0)

a = tf.Variable(a_data)
b = tf.Variable(b_data)
c = tf.Variable(c_data)
d = tf.Variable(d_data)

with tf.GradientTape() as tape:
    ab = a + b
    abc = ab + c
    res = abc + d

grad_a, grad_b, grad_d, grad_res = tape.gradient(res, [a, b, d, res])

print_grad("A", a, grad_a)
print_grad("B", b, grad_b)
print_grad("D", d, grad_d)
print_grad("res", res, grad_res)
