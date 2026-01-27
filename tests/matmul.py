import tensorflow as tf


def print_grad(label, val, grad):
    print("\n", label, ":\n", "val:\n", val.numpy(), "\ngrad:\n", grad.numpy())


A = tf.Variable(tf.fill([3, 5], 10.0))
B = tf.Variable(tf.fill([5, 8], 50.0))
D = tf.Variable(tf.fill([2, 2, 8, 2], 20.0))

with tf.GradientTape() as tape:
    AB = tf.matmul(A, B)
    C = tf.matmul(AB, D)
    loss = tf.reduce_sum(C)


g = tape.gradient(loss, [A, B, D, C])

print_grad("A", A, g[0])
print_grad("B", B, g[1])
print_grad("C", D, g[2])
print_grad("D", C, g[3])
