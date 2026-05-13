#include "check_tensor.hpp"
#include <array>
#include <cassert>
#include <kaad/graph/graph.hpp>
#include <kaad/graph/node_handle.hpp>
#include <kaad/operators/operators.hpp>
#include <kaad/scalar.hpp>
#include <kaad/tensor/internal/tensor_types.hpp>
#include <kaad/tensor/tensor_view.hpp>
#include <numeric>
#include <span>

/**
 * @brief This tests taking the mean of a tensor.
 * 1. [3,5,2] -> [3,5,1]
 * 2. [3,5,1] -> [3,1]
 * 3. [3,1] -> [1]
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


a = tf.Variable(get_data([3, 5, 2], 50))

with tf.GradientTape() as tape:

    a_mean = tf.reduce_mean(a, axis=2, keepdims=True)
    a_mean2 = tf.reduce_mean(a_mean, axis=1)
    res = tf.reduce_mean(a_mean2)

grad_a, grad_res = tape.gradient(res, [a, res])

print_all("a", a, grad_a)
print_all("res", res, grad_res)
*/

// NOLINTBEGIN(readability-magic-numbers)

kaad::Shape a_shape{3, 5, 2};
std::array<kaad::Scalar, 30> a_val{
    50.0, 51.0, 52.0, 53.0, 54.0, 55.0, 56.0, 57.0, 58.0, 59.0,
    60.0, 61.0, 62.0, 63.0, 64.0, 65.0, 66.0, 67.0, 68.0, 69.0,
    70.0, 71.0, 72.0, 73.0, 74.0, 75.0, 76.0, 77.0, 78.0, 79.0};
std::array<kaad::Scalar, 30> a_grad{
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

kaad::Shape res_shape{};
std::array<kaad::Scalar, 1> res_val{64.5};
std::array<kaad::Scalar, 1> res_grad{1.0};

int main() {
    kaad::Graph rec;

    kaad::Node input_a = rec.add_input_node(kaad::Shape{3, 5, 2});
    kaad::TensorViewMut a_view = input_a.value_mut();
    std::iota(a_view.begin(), a_view.end(), 50);

    // NOLINTEND(readability-magic-numbers)

    kaad::Node a_mean = mean(rec, input_a, 2, true);
    kaad::Node a_mean2 = mean(rec, a_mean, 1);
    kaad::Node res = mean(rec, a_mean2);

    rec.reset();

    rec.evaluate(std::array{res});

    rec.get_gradient(res, std::array{input_a});

    // Check a
    assert(check_tensor("a value", input_a.value(), a_shape, a_val));
    assert(check_tensor("a gradient", input_a.gradient(), a_shape, a_grad));

    // Check res
    assert(check_tensor("res value", res.value(), res_shape, res_val));
    assert(check_tensor("res gradient", res.gradient(), res_shape, res_grad));

    return 0;
}
