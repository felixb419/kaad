import tensorflow as tf


def print_grad(label, val, grad):
    print("\n", label, ":\n", "val:\n", val.numpy(), "\ngrad:\n", grad.numpy())


A = tf.Variable(tf.fill([5], 10.0))
B = tf.Variable(tf.fill([5], 50.0))
D = tf.Variable(tf.fill([5], 20.0))

with tf.GradientTape() as tape:
    AB = tf.tensordot(A, B, axes=1)

    # C = tf.tensordot(AB, D, axes=1)
    C = tf.reduce_sum(AB * D)  # Analog to dot-product with braodcasting

    loss = tf.reduce_sum(C)


g = tape.gradient(loss, [A, B, D, C])

print_grad("A", A, g[0])
print_grad("B", B, g[1])
print_grad("D", D, g[2])
print_grad("C", C, g[3])
