#include "../include/kaad/kaad.hpp"
#include "check_tensor.hpp" // for check_tensor
#include <algorithm>        // for fill
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
// clang-format off
/*
import tensorflow as tf

a_data = np.full((3, 2), 10.0)
b_data = np.full((3, 2), 30.0)
c_data = np.full((1,), 50.0)
d_data = np.full((2, 3, 1), 20.0)

a = tf.Variable(a_data)
b = tf.Variable(b_data)
c = tf.Variable(c_data)
d = tf.Variable(d_data)

with tf.GradientTape() as tape:
    ab = a + b
    abc = ab + c
    res = abc + d

grad_a, grad_b, grad_c, grad_d, grad_res = tape.gradient(res, [a, b, c, d, res])


def print_all(label, val, grad):
    print(label + " shape:", val.numpy().shape)
    print(label + " val:", val.numpy().ravel().tolist())
    print(label + " grad:", grad.numpy().ravel().tolist())


print_all("A", a, grad_a)
# A shape: (3, 2)
# A val: [10.0, 10.0, 10.0, 10.0, 10.0, 10.0]
# A grad: [2.0, 2.0, 2.0, 2.0, 2.0, 2.0]

print_all("B", b, grad_b)
# B shape: (3, 2)
# B val: [30.0, 30.0, 30.0, 30.0, 30.0, 30.0]
# B grad: [2.0, 2.0, 2.0, 2.0, 2.0, 2.0]

print_all("C", c, grad_c)
# C shape: (1,)
# C val: [50.0]
# C grad: [12.0]

print_all("D", d, grad_d)
# D shape: (2, 3, 1)
# D val: [20.0, 20.0, 20.0, 20.0, 20.0, 20.0]
# D grad: [2.0, 2.0, 2.0, 2.0, 2.0, 2.0]

print_all("Res", res, grad_res)
# Res shape: (2, 3, 2)
# Res val: [110.0, 110.0, 110.0, 110.0, 110.0, 110.0, 110.0, 110.0, 110.0, 110.0, 110.0, 110.0]
# Res grad: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
*/
// clang-format on

static std::array a_shape{3, 2};
static std::array<kaad::Scalar, 6> a_val{10.0, 10.0, 10.0, 10.0, 10.0, 10.0};
static std::array<kaad::Scalar, 6> a_grad{2.0, 2.0, 2.0, 2.0, 2.0, 2.0};

static std::array b_shape{3, 2};
static std::array<kaad::Scalar, 6> b_val{30.0, 30.0, 30.0, 30.0, 30.0, 30.0};
static std::array<kaad::Scalar, 6> b_grad{2.0, 2.0, 2.0, 2.0, 2.0, 2.0};

static std::array c_shape{1};
static std::array<kaad::Scalar, 1> c_val{50.0};
static std::array<kaad::Scalar, 1> c_grad{12.0};

static std::array d_shape{2, 3, 1};
static std::array<kaad::Scalar, 6> d_val{20.0, 20.0, 20.0, 20.0, 20.0, 20.0};
static std::array<kaad::Scalar, 6> d_grad{2.0, 2.0, 2.0, 2.0, 2.0, 2.0};

static std::array res_shape{2, 3, 2};
static std::array<kaad::Scalar, 12> res_val{110.0, 110.0, 110.0, 110.0,
                                            110.0, 110.0, 110.0, 110.0,
                                            110.0, 110.0, 110.0, 110.0};
static std::array<kaad::Scalar, 12> res_grad{1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                                             1.0, 1.0, 1.0, 1.0, 1.0, 1.0};

int main() {
    kaad::Computation_graph rec;

    std::span<float> a_vals;
    kaad::Node_handle a = rec.add_input_node(std::array{3, 2}, a_vals);
    std::fill(a_vals.begin(), a_vals.end(), 10);

    std::span<float> b_vals;
    kaad::Node_handle b = rec.add_input_node(std::array{3, 2}, b_vals);
    std::fill(b_vals.begin(), b_vals.end(), 30);

    std::span<float> c_vals;
    kaad::Node_handle c = rec.add_input_node(std::array{1}, c_vals);
    std::fill(c_vals.begin(), c_vals.end(), 50);

    std::span<float> d_vals;
    kaad::Node_handle d = rec.add_input_node(std::array{2, 3, 1}, d_vals);
    std::fill(d_vals.begin(), d_vals.end(), 20);

    kaad::Node_handle ab = add(rec, a, b);
    kaad::Node_handle abc = add(rec, ab, c);
    kaad::Node_handle res = add(rec, abc, d);

    rec.reset();

    rec.evaluate(std::array{res});

    rec.getGradient(res, std::array{a, b, d, res});

    // Check a
    assert(check_tensor(a.value(), a_shape, a_val));
    assert(check_tensor(a.gradient(), a_shape, a_grad));

    // Check b
    assert(check_tensor(b.value(), b_shape, b_val));
    assert(check_tensor(b.gradient(), b_shape, b_grad));

    // Check c
    assert(check_tensor(c.value(), c_shape, c_val));
    assert(check_tensor(c.gradient(), c_shape, c_grad));

    // Check d
    assert(check_tensor(d.value(), d_shape, d_val));
    assert(check_tensor(d.gradient(), d_shape, d_grad));

    // Check res
    assert(check_tensor(res.value(), res_shape, res_val));
    assert(check_tensor(res.gradient(), res_shape, res_grad));

    return 0;
}
