#include <algorithm>
#include <array>
#include <iostream>
#include <kaad/kaad.hpp>
#include <span>

int main() {
    // Create computation graph.
    kaad::Graph graph;

    // Add input nodes to the graph.
    kaad::Node input_a = input(graph, kaad::Shape{5});
    kaad::Node input_b = input(graph, kaad::Shape{5});
    kaad::Node input_c = input(graph, kaad::Shape{5});

    // Add computation nodes to graph via operators.
    kaad::Node dot_ab = dot(graph, input_a, input_b); // [5] * [5] -> [1]
    kaad::Node res = dot(graph, dot_ab, input_c);     // [1] * [5] -> [1]

    // allocate memory for the tensors
    graph.init();

    // Fill input nodes with values,
    // get mutable pointer to node values with .data_mut().
    std::fill_n(input_a.data_mut(), input_a.size(), 10);
    std::fill_n(input_b.data_mut(), input_a.size(), 50);
    std::fill_n(input_c.data_mut(), input_a.size(), 20);

    // Reset the graph.
    graph.reset();

    // Evaluate 'res'.
    graph.evaluate(std::array{res});

    // Compute the gradient of res w.r.t. to a, b and c.
    graph.get_gradient(res, std::array{input_a, input_b, input_c});

    // Print values of nodes.
    std::cout << "A:\n" << input_a << '\n';
    std::cout << "B:\n" << input_b << '\n';
    std::cout << "C:\n" << input_c << '\n';
    std::cout << "Res:\n" << res << '\n';

    return 0;
}
