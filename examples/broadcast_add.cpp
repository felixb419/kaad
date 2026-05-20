#include <algorithm>
#include <array>
#include <iostream>
#include <kaad/kaad.hpp>
#include <span>

int main() {
    // Create computation graph.
    kaad::Graph graph;

    // Add input nodes to the graph.
    kaad::Node input_a = graph.input(kaad::Shape{3, 2});

    kaad::Node input_b = graph.input(kaad::Shape{3, 2});

    kaad::Node input_c = graph.input(kaad::Shape{});

    kaad::Node input_d = graph.input(kaad::Shape{2, 3, 1});

    // Add computation nodes to graph via operators.
    kaad::Node a_plus_b = graph.add(input_a, input_b); // [3,2] + [3,2] -> [3,2]
    kaad::Node ab_plus_c = graph.add(a_plus_b, input_c); // [3,2] + [1] -> [3,2]
    kaad::Node res =
        graph.add(ab_plus_c, input_d); // [3,2] + [2,3,1] -> [2,3,2]

    // allocate memory for the tensors
    graph.init();

    // Fill input nodes with values,
    // get mutable pointer to node values with .data_mut().
    std::fill_n(input_a.data_mut(), input_a.size(), 10);
    std::fill_n(input_b.data_mut(), input_b.size(), 30);
    std::fill_n(input_c.data_mut(), input_c.size(), 50);
    std::fill_n(input_d.data_mut(), input_d.size(), 20);

    // Reset the graph.
    graph.reset();

    // Evaluate 'res'.
    graph.evaluate(std::array{res});

    // Compute the gradient of res w.r.t. to a, b, c and d.
    graph.get_gradient(res, std::array{input_a, input_b, input_c, input_d});

    // Print values of nodes.
    std::cout << "A:\n" << input_a << '\n';
    std::cout << "B:\n" << input_b << '\n';
    std::cout << "C:\n" << input_c << '\n';
    std::cout << "D:\n" << input_d << '\n';
    std::cout << "Res:\n" << res << '\n';

    return 0;
}
