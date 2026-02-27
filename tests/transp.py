import tensorflow as tf
import numpy as np


def print_grad(label, val, grad):
    print("\n", label, ":\n", "val:\n", val.numpy(), "\ngrad:\n", grad.numpy())


a_elements = np.arange(30).reshape(3, 5, 2).astype(np.float32)
a = tf.Variable(a_elements)

with tf.GradientTape() as tape:

    res = tf.transpose(a)

grad_a, grad_res = tape.gradient(res, [a, res])


def print_all(label, val, grad):
    print(label + " shape:", val.numpy().shape)
    print(label + " val:", val.numpy().ravel().tolist())
    print(label + " grad:", grad.numpy().ravel().tolist())


print_all("A", a, grad_a)
# A shape: (3, 5, 2)
# A val: [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0]
# A grad: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

print_all("Res", res, grad_res)
# Res shape: (2, 5, 3)
# Res val: [0.0, 10.0, 20.0, 2.0, 12.0, 22.0, 4.0, 14.0, 24.0, 6.0, 16.0, 26.0, 8.0, 18.0, 28.0, 1.0, 11.0, 21.0, 3.0, 13.0, 23.0, 5.0, 15.0, 25.0, 7.0, 17.0, 27.0, 9.0, 19.0, 29.0]
# Res grad: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
