#include "check_tensor.hpp"
#include <array>
#include <cassert>
#include <kaad/graph/graph.hpp>
#include <kaad/graph/node_handle.hpp>
#include <kaad/operators/operators.hpp>
#include <kaad/scalar.hpp>
#include <kaad/tensor/internal/tensor_types.hpp>
#include <numeric>
#include <span>

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


a = tf.Variable(get_data([5, 3], 200))
b = tf.Variable(get_data([5, 8], 900))
c = tf.Variable(get_data([2, 2, 8, 2], 100))


with tf.GradientTape() as tape:
    a_t = tf.transpose(a)
    prod_ab = tf.matmul(a_t, b)
    res = tf.matmul(prod_ab, c)


grad_a, grad_b, grad_c, grad_res = tape.gradient(res, [a, b, c, res])

print_all("a", a, grad_a)
print_all("b", b, grad_b)
print_all("c", c, grad_c)
print_all("res", res, grad_res)
 */

kaad::Shape a_shape = {5, 3};
std::array<kaad::Scalar, 15> a_val = {200.0, 201.0, 202.0, 203.0, 204.0,
                                      205.0, 206.0, 207.0, 208.0, 209.0,
                                      210.0, 211.0, 212.0, 213.0, 214.0};
std::array<kaad::Scalar, 15> a_grad = {
    7604528.0, 7604528.0, 7604528.0, 7671856.0, 7671856.0,
    7671856.0, 7739184.0, 7739184.0, 7739184.0, 7806512.0,
    7806512.0, 7806512.0, 7873840.0, 7873840.0, 7873840.0};

kaad::Shape b_shape = {5, 8};
std::array<kaad::Scalar, 40> b_val = {
    900.0, 901.0, 902.0, 903.0, 904.0, 905.0, 906.0, 907.0, 908.0, 909.0,
    910.0, 911.0, 912.0, 913.0, 914.0, 915.0, 916.0, 917.0, 918.0, 919.0,
    920.0, 921.0, 922.0, 923.0, 924.0, 925.0, 926.0, 927.0, 928.0, 929.0,
    930.0, 931.0, 932.0, 933.0, 934.0, 935.0, 936.0, 937.0, 938.0, 939.0};
std::array<kaad::Scalar, 40> b_grad = {
    600588.0, 610236.0, 619884.0, 629532.0, 639180.0, 648828.0, 658476.0,
    668124.0, 609552.0, 619344.0, 629136.0, 638928.0, 648720.0, 658512.0,
    668304.0, 678096.0, 618516.0, 628452.0, 638388.0, 648324.0, 658260.0,
    668196.0, 678132.0, 688068.0, 627480.0, 637560.0, 647640.0, 657720.0,
    667800.0, 677880.0, 687960.0, 698040.0, 636444.0, 646668.0, 656892.0,
    667116.0, 677340.0, 687564.0, 697788.0, 708012.0};

kaad::Shape c_shape = {2, 2, 8, 2};
std::array<kaad::Scalar, 64> c_val = {
    100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0, 110.0,
    111.0, 112.0, 113.0, 114.0, 115.0, 116.0, 117.0, 118.0, 119.0, 120.0, 121.0,
    122.0, 123.0, 124.0, 125.0, 126.0, 127.0, 128.0, 129.0, 130.0, 131.0, 132.0,
    133.0, 134.0, 135.0, 136.0, 137.0, 138.0, 139.0, 140.0, 141.0, 142.0, 143.0,
    144.0, 145.0, 146.0, 147.0, 148.0, 149.0, 150.0, 151.0, 152.0, 153.0, 154.0,
    155.0, 156.0, 157.0, 158.0, 159.0, 160.0, 161.0, 162.0, 163.0};
std::array<kaad::Scalar, 64> c_grad = {
    2844900.0, 2844900.0, 2848005.0, 2848005.0, 2851110.0, 2851110.0, 2854215.0,
    2854215.0, 2857320.0, 2857320.0, 2860425.0, 2860425.0, 2863530.0, 2863530.0,
    2866635.0, 2866635.0, 2844900.0, 2844900.0, 2848005.0, 2848005.0, 2851110.0,
    2851110.0, 2854215.0, 2854215.0, 2857320.0, 2857320.0, 2860425.0, 2860425.0,
    2863530.0, 2863530.0, 2866635.0, 2866635.0, 2844900.0, 2844900.0, 2848005.0,
    2848005.0, 2851110.0, 2851110.0, 2854215.0, 2854215.0, 2857320.0, 2857320.0,
    2860425.0, 2860425.0, 2863530.0, 2863530.0, 2866635.0, 2866635.0, 2844900.0,
    2844900.0, 2848005.0, 2848005.0, 2851110.0, 2851110.0, 2854215.0, 2854215.0,
    2857320.0, 2857320.0, 2860425.0, 2860425.0, 2863530.0, 2863530.0, 2866635.0,
    2866635.0};

kaad::Shape res_shape = {2, 2, 3, 2};
std::array<kaad::Scalar, 24> res_val = {
    810996672.0,  818575232.0,  814932608.0,  822548032.0,  818868480.0,
    826520576.0,  932254336.0,  939832896.0,  936778688.0,  944394048.0,
    941303040.0,  948955136.0,  1053511872.0, 1061090432.0, 1058624704.0,
    1066240128.0, 1063737600.0, 1071389696.0, 1174769536.0, 1182348032.0,
    1180470912.0, 1188086144.0, 1186172160.0, 1193824256.0};
std::array<kaad::Scalar, 24> res_grad = {
    1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
    1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};

int main() {
    kaad::Graph rec;

    kaad::Node input_a = rec.add_input_node(kaad::Shape{5, 3});
    kaad::Node input_b = rec.add_input_node(kaad::Shape{5, 8});
    kaad::Node input_c = rec.add_input_node(kaad::Shape{2, 2, 8, 2});

    kaad::Node a_t = transpose(rec, input_a);

    kaad::Node prod_ab = matmul(rec, a_t, input_b);
    kaad::Node res = matmul(rec, prod_ab, input_c);

    rec.init();

    std::iota(input_a.data_mut(), input_a.data_mut() + input_a.size(), 200);
    std::iota(input_b.data_mut(), input_b.data_mut() + input_b.size(), 900);
    std::iota(input_c.data_mut(), input_c.data_mut() + input_c.size(), 100);

    rec.reset();

    rec.evaluate(std::array{res});

    rec.get_gradient(res, std::array{input_a, input_b, input_c});

    // Check a
    assert(check_tensor("a value", input_a.value(), a_shape, a_val));
    assert(check_tensor("a gradient", input_a.gradient(), a_shape, a_grad));

    // Check b
    assert(check_tensor("b value", input_b.value(), b_shape, b_val));
    assert(check_tensor("b gradient", input_b.gradient(), b_shape, b_grad));

    // Check c
    assert(check_tensor("c value", input_c.value(), c_shape, c_val));
    assert(check_tensor("c gradient", input_c.gradient(), c_shape, c_grad));

    // Check res
    assert(check_tensor("res value", res.value(), res_shape, res_val));
    assert(check_tensor("res gradient", res.gradient(), res_shape, res_grad));

    return 0;
}
