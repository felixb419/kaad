#include "../include/kaad/kaad.hpp"
#include "check_tensor.hpp" // for check_tensor
#include <array>            // for array
#include <cassert>          // for assert
#include <numeric>          // for iota
#include <span>             // for span

/**
 * @brief This test tests batched and non-batched matrix multiplication with
 * these shapes:
 * 1. [3,5] * [5,8] -> [3,8] (matmul)
 * 2. [3,8] * [2,2,8,2] -> [2,2,3,2] (batch matmul)
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


a = tf.Variable(get_data([3, 5], 200))
b = tf.Variable(get_data([5, 8], 900))
c = tf.Variable(get_data([2, 2, 8, 2], 100))


with tf.GradientTape() as tape:
    ab = tf.matmul(a, b)
    res = tf.matmul(ab, c)


grad_a, grad_b, grad_c, grad_res = tape.gradient(res, [a, b, c, res])

print_all("a", a, grad_a)
print_all("b", b, grad_b)
print_all("c", c, grad_c)
print_all("res", res, grad_res)
*/

// NOLINTBEGIN(readability-magic-numbers)

std::array a_shape{3, 5};
std::array<kaad::Scalar, 15> a_val{200.0, 201.0, 202.0, 203.0, 204.0,
                                   205.0, 206.0, 207.0, 208.0, 209.0,
                                   210.0, 211.0, 212.0, 213.0, 214.0};
std::array<kaad::Scalar, 15> a_grad{7604528.0, 7671856.0, 7739184.0, 7806512.0,
                                    7873840.0, 7604528.0, 7671856.0, 7739184.0,
                                    7806512.0, 7873840.0, 7604528.0, 7671856.0,
                                    7739184.0, 7806512.0, 7873840.0};

std::array b_shape{5, 8};
std::array<kaad::Scalar, 40> b_val{
    900.0, 901.0, 902.0, 903.0, 904.0, 905.0, 906.0, 907.0, 908.0, 909.0,
    910.0, 911.0, 912.0, 913.0, 914.0, 915.0, 916.0, 917.0, 918.0, 919.0,
    920.0, 921.0, 922.0, 923.0, 924.0, 925.0, 926.0, 927.0, 928.0, 929.0,
    930.0, 931.0, 932.0, 933.0, 934.0, 935.0, 936.0, 937.0, 938.0, 939.0};
std::array<kaad::Scalar, 40> b_grad{
    612540.0, 622380.0, 632220.0, 642060.0, 651900.0, 661740.0, 671580.0,
    681420.0, 615528.0, 625416.0, 635304.0, 645192.0, 655080.0, 664968.0,
    674856.0, 684744.0, 618516.0, 628452.0, 638388.0, 648324.0, 658260.0,
    668196.0, 678132.0, 688068.0, 621504.0, 631488.0, 641472.0, 651456.0,
    661440.0, 671424.0, 681408.0, 691392.0, 624492.0, 634524.0, 644556.0,
    654588.0, 664620.0, 674652.0, 684684.0, 694716.0};

std::array c_shape{2, 2, 8, 2};
std::array<kaad::Scalar, 64> c_val{
    100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0, 110.0,
    111.0, 112.0, 113.0, 114.0, 115.0, 116.0, 117.0, 118.0, 119.0, 120.0, 121.0,
    122.0, 123.0, 124.0, 125.0, 126.0, 127.0, 128.0, 129.0, 130.0, 131.0, 132.0,
    133.0, 134.0, 135.0, 136.0, 137.0, 138.0, 139.0, 140.0, 141.0, 142.0, 143.0,
    144.0, 145.0, 146.0, 147.0, 148.0, 149.0, 150.0, 151.0, 152.0, 153.0, 154.0,
    155.0, 156.0, 157.0, 158.0, 159.0, 160.0, 161.0, 162.0, 163.0};
std::array<kaad::Scalar, 64> c_grad{
    2844420.0, 2844420.0, 2847525.0, 2847525.0, 2850630.0, 2850630.0, 2853735.0,
    2853735.0, 2856840.0, 2856840.0, 2859945.0, 2859945.0, 2863050.0, 2863050.0,
    2866155.0, 2866155.0, 2844420.0, 2844420.0, 2847525.0, 2847525.0, 2850630.0,
    2850630.0, 2853735.0, 2853735.0, 2856840.0, 2856840.0, 2859945.0, 2859945.0,
    2863050.0, 2863050.0, 2866155.0, 2866155.0, 2844420.0, 2844420.0, 2847525.0,
    2847525.0, 2850630.0, 2850630.0, 2853735.0, 2853735.0, 2856840.0, 2856840.0,
    2859945.0, 2859945.0, 2863050.0, 2863050.0, 2866155.0, 2866155.0, 2844420.0,
    2844420.0, 2847525.0, 2847525.0, 2850630.0, 2850630.0, 2853735.0, 2853735.0,
    2856840.0, 2856840.0, 2859945.0, 2859945.0, 2863050.0, 2863050.0, 2866155.0,
    2866155.0};

std::array res_shape{2, 2, 3, 2};
std::array<kaad::Scalar, 24> res_val{
    795116224.0,  802546432.0,  814795648.0,  822409728.0,  834475072.0,
    842273024.0,  913999424.0,  921429632.0,  936621248.0,  944235392.0,
    959243008.0,  967041024.0,  1032882624.0, 1040312960.0, 1058446848.0,
    1066060928.0, 1084011008.0, 1091809024.0, 1151765888.0, 1159195904.0,
    1180272512.0, 1187886592.0, 1208779008.0, 1216577024.0};
std::array<kaad::Scalar, 24> res_grad{1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                                      1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                                      1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};

int main() {
    kaad::Graph rec;

    auto a = rec.add_input_node(std::array{3, 5});
    std::span<float> a_vals = a.value_elements();
    std::iota(a_vals.begin(), a_vals.end(), 200);

    auto b = rec.add_input_node(std::array{5, 8});
    std::span<float> b_vals = b.value_elements();
    std::iota(b_vals.begin(), b_vals.end(), 900);

    auto c = rec.add_input_node(std::array{2, 2, 8, 2});
    std::span<float> c_vals = c.value_elements();
    std::iota(c_vals.begin(), c_vals.end(), 100);

    kaad::Node ab = matmul(rec, a, b);
    kaad::Node res = matmul(rec, ab, c);

    // NOLINTEND(readability-magic-numbers)

    rec.reset();

    rec.evaluate(std::array{res});

    rec.getGradient(res, std::array{a, b, c});

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
