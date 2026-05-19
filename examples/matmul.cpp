#include <algorithm>
#include <array>
#include <iostream>
#include <kaad/kaad.hpp>
#include <span>

int main() {
    // Create computation graph.
    kaad::Graph rec;

    // Add input nodes to the graph.
    kaad::Node input_a = input(rec, kaad::Shape{3, 5});

    kaad::Node input_b = input(rec, kaad::Shape{5, 8});

    kaad::Node input_c = input(rec, kaad::Shape{2, 2, 8, 2});

    // Add computation nodes to graph via operators.
    kaad::Node prod_ab = matmul(rec, input_a, input_b);
    kaad::Node res = matmul(rec, prod_ab, input_c);

    // allocate memory for the tensors
    rec.init();

    // Fill input nodes with values,
    // get mutable pointer to node values with .data_mut().
    std::fill_n(input_a.data_mut(), input_a.size(), 10);
    std::fill_n(input_b.data_mut(), input_a.size(), 50);
    std::fill_n(input_c.data_mut(), input_a.size(), 20);

    // Reset the graph.
    rec.reset();

    // Evaluate 'res'.
    rec.evaluate(std::array{res});

    // Compute the gradient of res w.r.t. to a, b and c.
    rec.get_gradient(res, std::array{input_a, input_b, input_c});

    // Print values of nodes.
    std::cout << "A:\n" << input_a << '\n';
    std::cout << "B:\n" << input_b << '\n';
    std::cout << "C:\n" << input_c << '\n';
    std::cout << "Res:\n" << res << '\n';

    return 0;
}
