#include "../include/kaad/kaad.hpp"
#include <algorithm> // for fill
#include <array>     // for array
#include <iostream>  // for operator<<
#include <span>      // for span

int main() {
    // Create computation graph.
    kaad::Graph rec;

    // Add input nodes to the graph.
    auto a = rec.add_input_node(std::array{5});
    std::span<float> a_vals =
        a.value_elements(); // Add input nodes to the graph
    std::ranges::fill(a_vals, 10);

    auto b = rec.add_input_node(std::array{5});
    std::span<float> b_vals = b.value_elements();
    std::ranges::fill(b_vals, 50);

    auto c = rec.add_input_node(std::array{5});
    std::span<float> c_vals = c.value_elements();
    std::ranges::fill(c_vals, 20);

    // Add computation nodes to graph via operators.
    kaad::Node ab = dot(rec, a, b);   // [5] * [5] -> [1]
    kaad::Node res = dot(rec, ab, c); // [1] * [5] -> [1]

    // Reset the graph.
    rec.reset();

    // Evaluate 'res'.
    rec.evaluate(std::array{res});

    // Compute the gradient of res w.r.t. to a, b and c.
    rec.getGradient(res, std::array{a, b, c});

    // Print values of nodes.
    std::cout << "A:\n" << a << '\n';
    std::cout << "B:\n" << b << '\n';
    std::cout << "C:\n" << c << '\n';
    std::cout << "Res:\n" << res << '\n';

    return 0;
}
