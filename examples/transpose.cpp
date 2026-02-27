#include "../include/kaad/kaad.hpp"
#include <array>    // for array
#include <iostream> // for operator<<
#include <numeric>  // for iota
#include <span>     // for span

int main() {
    // Create computation graph.
    kaad::Computation_graph rec;

    // Add input nodes to the graph.
    std::span<float> a_vals; // span to represent the element array of a
    kaad::Node_handle a = rec.add_input_node(std::array{3, 5, 2}, a_vals);
    std::iota(a_vals.begin(), a_vals.end(), 0);

    // Add computation nodes to graph via operators.
    kaad::Node_handle c = transpose(rec, a); // [3,5,2] -> [2,5,3]

    // Reset the graph.
    rec.reset();

    // Evaluate 'res'.
    rec.evaluate(std::array{c});

    // Compute the gradient of res w.r.t. to a.
    rec.getGradient(c, std::array{a});

    // Print values of nodes.
    std::cout << "A:\n" << a << std::endl;
    std::cout << "C:\n" << c << std::endl;

    return 0;
}
