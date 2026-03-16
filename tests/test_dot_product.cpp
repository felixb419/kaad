#include "check_tensor.hpp" // for check_tensor
#include <array>            // for array
#include <cassert>          // for assert
#include <kaad/kaad.hpp>
#include <numeric> // for iota
#include <span>    // for span

/**
 * @brief This test tests a taking the dot product with different kinds of
 * broadcasts:
 * 1. [5] * [5] -> [1] (matching shapes)
 * 2. [1] * [5] -> [1] (scalar broadcasting)
 */

// Tensorflow python code to verify results:
// clang-format off
/*
import tensorflow as tf
import numpy as np
from math import prod


def get_data(shape, val_start):
    return (np.arange(prod(shape)).reshape(shape).astype(np.float32)) + val_start


def print_all(label, val, grad):
    print(label + " shape:", list(val.numpy().shape))
    print(label + " val:", val.numpy().ravel().tolist())
    print(label + " grad:", grad.numpy().ravel().tolist())


a = tf.Variable(get_data([5], 40))
b = tf.Variable(get_data([5], 90))
c = tf.Variable(get_data([5], 150))

with tf.GradientTape() as tape:
    ab = tf.tensordot(a, b, axes=1)

    # res = tf.tensordot(ab, c, axes=1)
    res = tf.reduce_sum(ab * c)  # Analog to dot-product with braodcasting

grad_a, grad_b, grad_c, grad_res = tape.gradient(res, [a, b, c, res])

print_all("a", a, grad_a)
print_all("b", b, grad_b)
print_all("c", c, grad_c)
print_all("res", res, grad_res)
*/
// clang-format on

// NOLINTBEGIN(readability-magic-numbers)

std::array a_shape{5};
std::array<kaad::Scalar, 5> a_val{40.0, 41.0, 42.0, 43.0, 44.0};
std::array<kaad::Scalar, 5> a_grad{68400.0, 69160.0, 69920.0, 70680.0, 71440.0};

std::array b_shape{5};
std::array<kaad::Scalar, 5> b_val{90.0, 91.0, 92.0, 93.0, 94.0};
std::array<kaad::Scalar, 5> b_grad{30400.0, 31160.0, 31920.0, 32680.0, 33440.0};

std::array c_shape{5};
std::array<kaad::Scalar, 5> c_val{150.0, 151.0, 152.0, 153.0, 154.0};
std::array<kaad::Scalar, 5> c_grad{19330.0, 19330.0, 19330.0, 19330.0, 19330.0};

std::array<int, 0> res_shape{};
std::array<kaad::Scalar, 1> res_val{14690800.0};
std::array<kaad::Scalar, 1> res_grad{1.0};

int main() {
    kaad::Graph rec;

    kaad::Node input_a = rec.add_input_node(std::array{5});
    std::span<float> a_vals = input_a.value_elements();
    std::iota(a_vals.begin(), a_vals.end(), 40);

    kaad::Node input_b = rec.add_input_node(std::array{5});
    std::span<float> b_vals = input_b.value_elements();
    std::iota(b_vals.begin(), b_vals.end(), 90);

    kaad::Node input_c = rec.add_input_node(std::array{5});
    std::span<float> c_vals = input_c.value_elements();
    std::iota(c_vals.begin(), c_vals.end(), 150);

    kaad::Node dot_ab = dot(rec, input_a, input_b);
    kaad::Node res = dot(rec, dot_ab, input_c);

    // NOLINTEND(readability-magic-numbers)

    rec.reset();

    rec.evaluate(std::array{res});

    rec.getGradient(res, std::array{input_a, input_b, input_c});

    // Check a
    assert(check_tensor("a value", a.value(), a_shape, a_val));
    assert(check_tensor("a gradient", a.gradient(), a_shape, a_grad));

    // Check b
    assert(check_tensor("b value", b.value(), b_shape, b_val));
    assert(check_tensor("b gradient", b.gradient(), b_shape, b_grad));

    // Check c
    assert(check_tensor("c value", c.value(), c_shape, c_val));
    assert(check_tensor("c gradient", c.gradient(), c_shape, c_grad));

    // Check res
    assert(check_tensor("res value", res.value(), res_shape, res_val));
    assert(check_tensor("res gradient", res.gradient(), res_shape, res_grad));

    return 0;
}
