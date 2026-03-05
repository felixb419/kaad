#include "../include/kaad/kaad.hpp"
#include "check_tensor.hpp" // for check_tensor
#include <array>            // for array
#include <cassert>          // for assert
#include <iostream>         // for operator<<
#include <numeric>          // for iota
#include <span>             // for span

/**
 * @brief This tests transposition wiht these shapes:
 * 1. [3,5,2] -> [2,5,3]
 */

// Tensorflow python code to verify results:
/*
import tensorflow as tf
import numpy as np
from math import prod


def get_data(shape, val_start):
    return np.arange(prod(shape)).reshape(shape).astype(np.float32) + val_start


def print_all(label, val, grad):
    print(label + " shape:", list(val.numpy().shape))
    print(label + " val:", val.numpy().ravel().tolist())
    print(label + " grad:", grad.numpy().ravel().tolist())


a = tf.Variable(get_data([3, 5, 2], 32))

with tf.GradientTape() as tape:

    res = tf.transpose(a)

grad_a, grad_res = tape.gradient(res, [a, res])

print_all("a", a, grad_a)
print_all("res", res, grad_res)
*/

std::array a_shape{3, 5, 2};
std::array<kaad::Scalar, 30> a_val{
    32.0, 33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0, 40.0, 41.0,
    42.0, 43.0, 44.0, 45.0, 46.0, 47.0, 48.0, 49.0, 50.0, 51.0,
    52.0, 53.0, 54.0, 55.0, 56.0, 57.0, 58.0, 59.0, 60.0, 61.0};
std::array<kaad::Scalar, 30> a_grad{
    1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
    1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};

std::array res_shape{2, 5, 3};
std::array<kaad::Scalar, 30> res_val{
    32.0, 42.0, 52.0, 34.0, 44.0, 54.0, 36.0, 46.0, 56.0, 38.0,
    48.0, 58.0, 40.0, 50.0, 60.0, 33.0, 43.0, 53.0, 35.0, 45.0,
    55.0, 37.0, 47.0, 57.0, 39.0, 49.0, 59.0, 41.0, 51.0, 61.0};
std::array<kaad::Scalar, 30> res_grad{
    1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
    1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};

int main() {
    kaad::Computation_graph rec;

    std::span<float> a_vals;
    kaad::Node_handle a = rec.add_input_node(std::array{3, 5, 2}, a_vals);
    std::iota(a_vals.begin(), a_vals.end(), 32);

    kaad::Node_handle res = transpose(rec, a);

    rec.reset();

    rec.evaluate(std::array{res});

    rec.getGradient(res, std::array{a});

    // Check a
    assert(check_tensor("a value", a.value(), a_shape, a_val));
    assert(check_tensor("a gradient", a.gradient(), a_shape, a_grad));

    // Check res
    assert(check_tensor("res value", res.value(), res_shape, res_val));
    assert(check_tensor("res gradient", res.gradient(), res_shape, res_grad));

    return 0;
}
