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


a = tf.Variable(get_data([2, 3], 200))
b = tf.Variable(get_data([3, 2], 90))
c = tf.Variable(get_data([], 30))
d = tf.Variable(get_data([2, 3, 1], 0))

with tf.GradientTape() as tape:
    a_t = tf.transpose(a)

    ab = a_t + b
    abc = ab + c
    res = abc + d

grad_a, grad_b, grad_c, grad_d, grad_res = tape.gradient(res, [a, b, c, d, res])

print_all("a", a, grad_a)
print_all("b", b, grad_b)
print_all("c", c, grad_c)
print_all("d", d, grad_d)
print_all("res", res, grad_res)
 */

// NOLINTBEGIN(readability-magic-numbers)

kaad::Shape a_shape = {2, 3};
std::array<kaad::Scalar, 6> a_val = {200.0, 201.0, 202.0, 203.0, 204.0, 205.0};
std::array<kaad::Scalar, 6> a_grad = {2.0, 2.0, 2.0, 2.0, 2.0, 2.0};

kaad::Shape b_shape = {3, 2};
std::array<kaad::Scalar, 6> b_val = {90.0, 91.0, 92.0, 93.0, 94.0, 95.0};
std::array<kaad::Scalar, 6> b_grad = {2.0, 2.0, 2.0, 2.0, 2.0, 2.0};

kaad::Shape c_shape = {};
std::array<kaad::Scalar, 1> c_val = {30.0};
std::array<kaad::Scalar, 1> c_grad = {12.0};

kaad::Shape d_shape = {2, 3, 1};
std::array<kaad::Scalar, 6> d_val = {0.0, 1.0, 2.0, 3.0, 4.0, 5.0};
std::array<kaad::Scalar, 6> d_grad = {2.0, 2.0, 2.0, 2.0, 2.0, 2.0};

kaad::Shape res_shape = {2, 3, 2};
std::array<kaad::Scalar, 12> res_val = {320.0, 324.0, 324.0, 328.0,
                                        328.0, 332.0, 323.0, 327.0,
                                        327.0, 331.0, 331.0, 335.0};
std::array<kaad::Scalar, 12> res_grad = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                                         1.0, 1.0, 1.0, 1.0, 1.0, 1.0};

int main() {
    kaad::Graph rec;

    kaad::Node input_a = rec.add_input_node(kaad::Shape{2, 3});
    kaad::TensorViewMut a_view = input_a.value_mut();
    std::iota(a_view.begin(), a_view.end(), 200);

    kaad::Node input_b = rec.add_input_node(kaad::Shape{3, 2});
    kaad::TensorViewMut b_view = input_b.value_mut();
    std::iota(b_view.begin(), b_view.end(), 90);

    kaad::Node input_c = rec.add_input_node(kaad::SCALAR_SHAPE);
    kaad::TensorViewMut c_view = input_c.value_mut();
    std::iota(c_view.begin(), c_view.end(), 30);

    kaad::Node input_d = rec.add_input_node(kaad::Shape{2, 3, 1});
    kaad::TensorViewMut d_view = input_d.value_mut();
    std::iota(d_view.begin(), d_view.end(), 0);

    // NOLINTEND(readability-magic-numbers)

    kaad::Node a_t = transpose(rec, input_a);

    kaad::Node a_plus_b = add(rec, a_t, input_b);
    kaad::Node ab_plus_c = add(rec, a_plus_b, input_c);
    kaad::Node res = add(rec, ab_plus_c, input_d);

    rec.allocate();

    rec.reset();

    rec.evaluate(std::array{res});

    rec.get_gradient(res, std::array{input_a, input_b, input_d, res});

    // Check a
    assert(check_tensor("a value", input_a.value(), a_shape, a_val));
    assert(check_tensor("a gradient", input_a.gradient(), a_shape, a_grad));

    // Check b
    assert(check_tensor("b value", input_b.value(), b_shape, b_val));
    assert(check_tensor("b gradient", input_b.gradient(), b_shape, b_grad));

    // Check c
    assert(check_tensor("c value", input_c.value(), c_shape, c_val));
    assert(check_tensor("c gradient", input_c.gradient(), c_shape, c_grad));

    // Check d
    assert(check_tensor("d value", input_d.value(), d_shape, d_val));
    assert(check_tensor("d gradient", input_d.gradient(), d_shape, d_grad));

    // Check res
    assert(check_tensor("res value", res.value(), res_shape, res_val));
    assert(check_tensor("res gradient", res.gradient(), res_shape, res_grad));

    return 0;
}
