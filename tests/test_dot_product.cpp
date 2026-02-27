#include "../include/kaad/kaad.hpp"
#include "check_tensor.hpp" // for check_tensor
#include <algorithm>        // for fill
#include <array>            // for array
#include <cassert>          // for assert
#include <span>             // for span

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

a = tf.Variable(tf.fill([5], 10.0))
b = tf.Variable(tf.fill([5], 50.0))
c = tf.Variable(tf.fill([5], 20.0))

with tf.GradientTape() as tape:
    ab = tf.tensordot(a, b, axes=1)

    # res = tf.tensordot(ab, c, axes=1)
    res = tf.reduce_sum(ab * c)  # Analog to dot-product with braodcasting

grad_a, grad_b, grad_c, grad_res = tape.gradient(res, [a, b, c, res])


def print_all(label, val, grad):
    print(label + " shape:", val.numpy().shape)
    print(label + " val:", val.numpy().ravel().tolist())
    print(label + " grad:", grad.numpy().ravel().tolist())


print_all("A", a, grad_a)
# A shape: (5,)
# A val: [10.0, 10.0, 10.0, 10.0, 10.0]
# A grad: [5000.0, 5000.0, 5000.0, 5000.0, 5000.0]

print_all("B", b, grad_b)
# B shape: (5,)
# B val: [50.0, 50.0, 50.0, 50.0, 50.0]
# B grad: [1000.0, 1000.0, 1000.0, 1000.0, 1000.0]

print_all("C", c, grad_c)
# C shape: (5,)
# C val: [20.0, 20.0, 20.0, 20.0, 20.0]
# C grad: [2500.0, 2500.0, 2500.0, 2500.0, 2500.0]

print_all("Res", res, grad_res)
# Res shape: ()
# Res val: [250000.0]
# Res grad: [1.0]
*/
// clang-format on

static std::array a_shape{5};
static std::array<kaad::Scalar, 5> a_val{10.0, 10.0, 10.0, 10.0, 10.0};
static std::array<kaad::Scalar, 5> a_grad{5000.0, 5000.0, 5000.0, 5000.0,
                                          5000.0};

static std::array b_shape{5};
static std::array<kaad::Scalar, 5> b_val{50.0, 50.0, 50.0, 50.0, 50.0};
static std::array<kaad::Scalar, 5> b_grad{1000.0, 1000.0, 1000.0, 1000.0,
                                          1000.0};

static std::array c_shape{5};
static std::array<kaad::Scalar, 5> c_val{20.0, 20.0, 20.0, 20.0, 20.0};
static std::array<kaad::Scalar, 5> c_grad{2500.0, 2500.0, 2500.0, 2500.0,
                                          2500.0};

static std::array res_shape{1};
static std::array<kaad::Scalar, 1> res_val{250000.0};
static std::array<kaad::Scalar, 1> res_grad{1.0};

int main() {
    kaad::Computation_graph rec;

    std::span<float> a_vals;
    kaad::Node_handle a = rec.add_input_node(std::array{5}, a_vals);
    std::fill(a_vals.begin(), a_vals.end(), 10);

    std::span<float> b_vals;
    kaad::Node_handle b = rec.add_input_node(std::array{5}, b_vals);
    std::fill(b_vals.begin(), b_vals.end(), 50);

    std::span<float> c_vals;
    kaad::Node_handle c = rec.add_input_node(std::array{5}, c_vals);
    std::fill(c_vals.begin(), c_vals.end(), 20);

    kaad::Node_handle ab = dot(rec, a, b);
    kaad::Node_handle res = dot(rec, ab, c);

    rec.reset();

    rec.evaluate(std::array{res});

    rec.getGradient(res, std::array{a, b, c});

    // Check a
    assert(check_tensor(a.value(), a_shape, a_val));
    assert(check_tensor(a.gradient(), a_shape, a_grad));

    // Check b
    assert(check_tensor(b.value(), b_shape, b_val));
    assert(check_tensor(b.gradient(), b_shape, b_grad));

    // Check c
    assert(check_tensor(c.value(), c_shape, c_val));
    assert(check_tensor(c.gradient(), c_shape, c_grad));

    // Check res
    assert(check_tensor(res.value(), res_shape, res_val));
    assert(check_tensor(res.gradient(), res_shape, res_grad));

    return 0;
}
