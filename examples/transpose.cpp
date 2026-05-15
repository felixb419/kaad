#include <array>
#include <iostream>
#include <kaad/kaad.hpp>
#include <numeric>
#include <span>

int main() {
    // Create computation graph.
    kaad::Graph rec;

    // Add input nodes to the graph.
    kaad::Node input_a = rec.add_input_node(kaad::Shape{3, 5, 2});

    // Add computation nodes to graph via operators.
    kaad::Node res = transpose(rec, input_a); // [3,5,2] -> [2,5,3]

    // allocate memory for the tensors
    rec.allocate();

    // Fill input nodes with values,
    // get mutable pointer to node values with .data_mut().
    std::iota(input_a.data_mut(), input_a.data_mut() + input_a.size(), 0);

    // Reset the graph.
    rec.reset();

    // Evaluate 'res'.
    rec.evaluate(std::array{res});

    // Compute the gradient of res w.r.t. to a.
    rec.get_gradient(res, std::array{input_a});

    // Print values of nodes.
    std::cout << "A:\n" << input_a << '\n';
    std::cout << "C:\n" << res << '\n';

    return 0;
}
