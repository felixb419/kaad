#include "check_tensor.hpp"
#include "kaad/tensor/internal/tensor_types.hpp"

#include <array>
#include <cassert>
#include <kaad/graph/graph.hpp>
#include <kaad/graph/node_handle.hpp>
#include <kaad/operators/operators.hpp>
#include <kaad/scalar.hpp>
#include <numeric>
#include <span>

/**
 * @brief This tests taking the sum of a tensor.
 * 1. [2,5,3] -> [2,5,1]
 * 2. [2,5,1] -> [2,1]
 * 3. [2,1] -> [1]
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

    a_t = tf.transpose(a)

    a_sum = tf.reduce_sum(a_t, axis=2, keepdims=True)
    a_sum2 = tf.reduce_sum(a_sum, axis=1)
    res = tf.reduce_sum(a_sum2)

grad_a, grad_res = tape.gradient(res, [a, res])

print_all("a", a, grad_a)
print_all("res", res, grad_res)
*/

kaad::Shape a_shape = {3, 5, 2};
std::array<kaad::Scalar, 30> a_val = {
    50.0, 51.0, 52.0, 53.0, 54.0, 55.0, 56.0, 57.0, 58.0, 59.0,
    60.0, 61.0, 62.0, 63.0, 64.0, 65.0, 66.0, 67.0, 68.0, 69.0,
    70.0, 71.0, 72.0, 73.0, 74.0, 75.0, 76.0, 77.0, 78.0, 79.0};
std::array<kaad::Scalar, 30> a_grad = {
    1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
    1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};

kaad::Shape res_shape = {};
std::array<kaad::Scalar, 1> res_val = {1935.0};
std::array<kaad::Scalar, 1> res_grad = {1.0};

int main() {
    kaad::Graph graph;

    kaad::Node input_a = input(graph, kaad::Shape{3, 5, 2});

    kaad::Node a_t = transpose(graph, input_a);

    kaad::Node a_sum = sum(graph, a_t, 2, true);
    kaad::Node a_sum2 = sum(graph, a_sum, 1);
    kaad::Node res = sum(graph, a_sum2);

    graph.init();

    std::iota(input_a.data_mut(), input_a.data_mut() + input_a.size(), 50);

    graph.reset();

    graph.evaluate(std::array{res});

    graph.get_gradient(res, std::array{input_a});

    // Check a
    assert(check_tensor("a value", input_a.value(), a_shape, a_val));
    assert(check_tensor("a gradient", input_a.gradient(), a_shape, a_grad));

    // Check res
    assert(check_tensor("res value", res.value(), res_shape, res_val));
    assert(check_tensor("res gradient", res.gradient(), res_shape, res_grad));

    return 0;
}
