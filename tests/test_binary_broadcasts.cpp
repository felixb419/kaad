#include "../include/kaad/kaad.hpp"
#include "check_tensor.hpp" // for check_tensor
#include <array>            // for array
#include <cassert>          // for assert
#include <span>             // for span

/**
 * @brief This test tests a binary operation (add in this case) with different
 * kinds of broadcasts:
 * 1. [3,2] + [3,2] -> [3,2] (matching shapes)
 * 2. [3,2] + [1] -> [3,2] (scalar broadcasting)
 * 3. [3,2] + [2,3,1] -> [2,3,2] (full broadcasting)
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


a = tf.Variable(get_data([3, 2], 200))
b = tf.Variable(get_data([3, 2], 90))
c = tf.Variable(get_data([1], 30))
d = tf.Variable(get_data([2, 3, 1], 0))

with tf.GradientTape() as tape:
    ab = a + b
    abc = ab + c
    res = abc + d

grad_a, grad_b, grad_c, grad_d, grad_res = tape.gradient(res, [a, b, c, d, res])

print_all("a", a, grad_a)
print_all("b", b, grad_b)
print_all("c", c, grad_c)
print_all("d", d, grad_d)
print_all("res", res, grad_res)
 */

std::array a_shape{3, 2};
std::array<kaad::Scalar, 6> a_val{200.0, 201.0, 202.0, 203.0, 204.0, 205.0};
std::array<kaad::Scalar, 6> a_grad{2.0, 2.0, 2.0, 2.0, 2.0, 2.0};

std::array b_shape{3, 2};
std::array<kaad::Scalar, 6> b_val{90.0, 91.0, 92.0, 93.0, 94.0, 95.0};
std::array<kaad::Scalar, 6> b_grad{2.0, 2.0, 2.0, 2.0, 2.0, 2.0};

std::array c_shape{1};
std::array<kaad::Scalar, 1> c_val{30.0};
std::array<kaad::Scalar, 1> c_grad{12.0};

std::array d_shape{2, 3, 1};
std::array<kaad::Scalar, 6> d_val{0.0, 1.0, 2.0, 3.0, 4.0, 5.0};
std::array<kaad::Scalar, 6> d_grad{2.0, 2.0, 2.0, 2.0, 2.0, 2.0};

std::array res_shape{2, 3, 2};
std::array<kaad::Scalar, 12> res_val{320.0, 322.0, 325.0, 327.0, 330.0, 332.0,
                                     323.0, 325.0, 328.0, 330.0, 333.0, 335.0};
std::array<kaad::Scalar, 12> res_grad{1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                                      1.0, 1.0, 1.0, 1.0, 1.0, 1.0};

int main() {
    kaad::Graph rec;

    kaad::Node a = rec.add_input_node(std::array{3, 2});
    std::span<float> a_vals = a.value_elements();
    std::iota(a_vals.begin(), a_vals.end(), 200);

    kaad::Node b = rec.add_input_node(std::array{3, 2});
    std::span<float> b_vals = b.value_elements();
    std::iota(b_vals.begin(), b_vals.end(), 90);

    kaad::Node c = rec.add_input_node(std::array{1});
    std::span<float> c_vals = c.value_elements();
    std::iota(c_vals.begin(), c_vals.end(), 30);

    kaad::Node d = rec.add_input_node(std::array{2, 3, 1});
    std::span<float> d_vals = d.value_elements();
    std::iota(d_vals.begin(), d_vals.end(), 0);

    kaad::Node ab = add(rec, a, b);
    kaad::Node abc = add(rec, ab, c);
    kaad::Node res = add(rec, abc, d);

    rec.reset();

    rec.evaluate(std::array{res});

    rec.getGradient(res, std::array{a, b, d, res});

    // Check a
    assert(check_tensor("a value", a.value(), a_shape, a_val));
    assert(check_tensor("a gradient", a.gradient(), a_shape, a_grad));

    // Check b
    assert(check_tensor("b value", b.value(), b_shape, b_val));
    assert(check_tensor("b gradient", b.gradient(), b_shape, b_grad));

    // Check c
    assert(check_tensor("c value", c.value(), c_shape, c_val));
    assert(check_tensor("c gradient", c.gradient(), c_shape, c_grad));

    // Check d
    assert(check_tensor("d value", d.value(), d_shape, d_val));
    assert(check_tensor("d gradient", d.gradient(), d_shape, d_grad));

    // Check res
    assert(check_tensor("res value", res.value(), res_shape, res_val));
    assert(check_tensor("res gradient", res.gradient(), res_shape, res_grad));

    return 0;
}
