import tensorflow as tf

A = tf.Variable(tf.fill([3, 5], 10.0))
B = tf.Variable(tf.fill([5, 8], 50.0))
D = tf.Variable(tf.fill([2, 2, 8, 2], 20.0))

print("A:\n", A.numpy())
print("B:\n", B.numpy())
print("D:\n", D.numpy())

with tf.GradientTape() as tape:
    AB = tf.matmul(A, B)
    C = tf.matmul(AB, D)
    loss = tf.reduce_sum(C)

print("C:\n", C.numpy())

g = tape.gradient(loss, [A, B, D])

print("dA:\n", g[0].numpy())
print("dB:\n", g[1].numpy())
print("dD:\n", g[2].numpy())
