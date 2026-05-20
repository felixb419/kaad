#include "check_tensor.hpp"
#include "kaad/tensor/internal/tensor_types.hpp"

#include <array>
#include <cassert>
#include <kaad/graph/graph.hpp>
#include <kaad/graph/node_handle.hpp>
#include <kaad/scalar.hpp>
#include <numeric>
#include <span>

/**
 * @brief This test tests a slicing operation
 * 1. [5,4,3,2] -> [5,2,2,1]
 */

// Tensorflow python code to verify results:
// clang-format off
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


a = tf.Variable(get_data([2, 3, 4, 5], 20))

with tf.GradientTape() as tape:
    a_t = tf.transpose(a)

    res = tf.slice(a_t, [1, 2, 0, 1], [4, 2, 2, 1])

grad_a, grad_a_t, grad_res = tape.gradient(res, [a, a_t, res])

print_all("a", a, grad_a)
print_all("res", res, grad_res)
*/
// clang-format on

kaad::Shape a_shape = {2, 3, 4, 5};
std::array<kaad::Scalar, 120> a_val = {
    20.0,  21.0,  22.0,  23.0,  24.0,  25.0,  26.0,  27.0,  28.0,  29.0,  30.0,
    31.0,  32.0,  33.0,  34.0,  35.0,  36.0,  37.0,  38.0,  39.0,  40.0,  41.0,
    42.0,  43.0,  44.0,  45.0,  46.0,  47.0,  48.0,  49.0,  50.0,  51.0,  52.0,
    53.0,  54.0,  55.0,  56.0,  57.0,  58.0,  59.0,  60.0,  61.0,  62.0,  63.0,
    64.0,  65.0,  66.0,  67.0,  68.0,  69.0,  70.0,  71.0,  72.0,  73.0,  74.0,
    75.0,  76.0,  77.0,  78.0,  79.0,  80.0,  81.0,  82.0,  83.0,  84.0,  85.0,
    86.0,  87.0,  88.0,  89.0,  90.0,  91.0,  92.0,  93.0,  94.0,  95.0,  96.0,
    97.0,  98.0,  99.0,  100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0,
    108.0, 109.0, 110.0, 111.0, 112.0, 113.0, 114.0, 115.0, 116.0, 117.0, 118.0,
    119.0, 120.0, 121.0, 122.0, 123.0, 124.0, 125.0, 126.0, 127.0, 128.0, 129.0,
    130.0, 131.0, 132.0, 133.0, 134.0, 135.0, 136.0, 137.0, 138.0, 139.0};
std::array<kaad::Scalar, 120> a_grad = {
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0,
    0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

kaad::Shape res_shape = {4, 2, 2, 1};
std::array<kaad::Scalar, 16> res_val = {91.0, 111.0, 96.0, 116.0, 92.0, 112.0,
                                        97.0, 117.0, 93.0, 113.0, 98.0, 118.0,
                                        94.0, 114.0, 99.0, 119.0};
std::array<kaad::Scalar, 16> res_grad = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                                         1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                                         1.0, 1.0, 1.0, 1.0};

int main() {
    kaad::Graph graph;

    kaad::Node input_a = graph.input(kaad::Shape{2, 3, 4, 5});

    kaad::Node a_t = graph.transpose(input_a);
    kaad::Node res = graph.slice(a_t, {4, 2, 2, 1}, {1, 2, 0, 1});

    graph.init();

    std::iota(input_a.data_mut(), input_a.data_mut() + input_a.size(), 20);

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
