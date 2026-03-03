#include "../include/kaad/kaad.hpp"
#include "check_tensor.hpp" // for check_tensor
#include <array>            // for array
#include <cassert>          // for assert
#include <iostream>         // for operator<<
#include <numeric>          // for iota
#include <span>             // for span

/**
 * @brief This tests taking the mean of a tensor.
 * 1. [3,5,2] -> [3,5,1]
 * 2. [3,5,1] -> [3,1]
 * 3. [3,1] -> [1]
 */

// Tensorflow python code to verify results:
// clang-format off
/*
import tensorflow as tf
import numpy as np


def print_grad(label, val, grad):
    print("\n", label, ":\n", "val:\n", val.numpy(), "\ngrad:\n", grad.numpy())


a_elements = np.arange(30).reshape(3, 5, 2).astype(np.float32)
a = tf.Variable(a_elements)

with tf.GradientTape() as tape:

    a_mean = tf.reduce_mean(a, axis=2, keepdims=True)
    a_mean2 = tf.reduce_mean(a_mean, axis=1)
    res = tf.reduce_mean(a_mean2)

grad_a, grad_res = tape.gradient(res, [a, res])


def print_all(label, val, grad):
    print(label + " shape:", val.numpy().shape)
    print(label + " val:", val.numpy().ravel().tolist())
    print(label + " grad:", grad.numpy().ravel().tolist())


print_all("A", a, grad_a)
# A shape: (3, 5, 2)
# A val: [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0]
# A grad: [0.03333333507180214, 0.03333333507180214, 0.03333333507180214, 0.03333333507180214, 0.03333333507180214, 0.03333333507180214, 0.03333333507180214, 0.03333333507180214, 0.03333333507180214, 0.03333333507180214, 0.03333333507180214, 0.03333333507180214, 0.03333333507180214, 0.03333333507180214, 0.03333333507180214, 0.03333333507180214, 0.03333333507180214, 0.03333333507180214, 0.03333333507180214, 0.03333333507180214, 0.03333333507180214, 0.03333333507180214, 0.03333333507180214, 0.03333333507180214, 0.03333333507180214, 0.03333333507180214, 0.03333333507180214, 0.03333333507180214, 0.03333333507180214, 0.03333333507180214]

print_all("Res", res, grad_res)
# Res shape: ()
# Res val: [14.5]
# Res grad: [1.0]
*/
// clang-format on

std::array a_shape{3, 5, 2};
std::array<float, 30> a_val{0.0,  1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,
                            8.0,  9.0,  10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
                            16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0,
                            24.0, 25.0, 26.0, 27.0, 28.0, 29.0};
std::array<float, 30> a_grad{
    0.03333333507180214, 0.03333333507180214, 0.03333333507180214,
    0.03333333507180214, 0.03333333507180214, 0.03333333507180214,
    0.03333333507180214, 0.03333333507180214, 0.03333333507180214,
    0.03333333507180214, 0.03333333507180214, 0.03333333507180214,
    0.03333333507180214, 0.03333333507180214, 0.03333333507180214,
    0.03333333507180214, 0.03333333507180214, 0.03333333507180214,
    0.03333333507180214, 0.03333333507180214, 0.03333333507180214,
    0.03333333507180214, 0.03333333507180214, 0.03333333507180214,
    0.03333333507180214, 0.03333333507180214, 0.03333333507180214,
    0.03333333507180214, 0.03333333507180214, 0.03333333507180214};

std::array res_shape{1}; // actual tensorflow output is 0-rank
std::array<float, 1> res_val{14.5};
std::array<float, 1> res_grad{1.0};

int main() {
    kaad::Computation_graph rec;

    std::span<float> a_vals;
    kaad::Node_handle a = rec.add_input_node(std::array{3, 5, 2}, a_vals);
    std::iota(a_vals.begin(), a_vals.end(), 0);

    kaad::Node_handle a_mean = mean(rec, a, 2, true);
    kaad::Node_handle a_mean2 = mean(rec, a, 1);
    kaad::Node_handle res = mean(rec, a_mean2);

    rec.reset();

    rec.evaluate(std::array{res});

    rec.getGradient(res, std::array{a});

    // Check a
    assert(check_tensor(a.value(), a_shape, a_val));
    assert(check_tensor(a.gradient(), a_shape, a_grad));

    // Check res
    assert(check_tensor(res.value(), res_shape, res_val));
    assert(check_tensor(res.gradient(), res_shape, res_grad));

    return 0;
}
