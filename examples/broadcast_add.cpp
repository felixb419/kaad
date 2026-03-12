#include "../include/kaad/kaad.hpp"
#include <algorithm> // for __fill_fn, fill
#include <array>     // for array
#include <iostream>  // for basic_ostream, operator<<
#include <span>      // for span

int main() {
    // Create computation graph.
    kaad::Graph rec;

    // NOLINTBEGIN(readability-magic-numbers)

    // Add input nodes to the graph.
    kaad::Node input_a = rec.add_input_node(std::array{3, 2});
    std::span<float> a_vals =
        input_a.value_elements(); // span to represent the element array of a
    std::ranges::fill(a_vals, 10);

    kaad::Node input_b = rec.add_input_node(std::array{3, 2});
    std::span<float> b_vals = input_b.value_elements();
    std::ranges::fill(b_vals, 30);

    kaad::Node input_c = rec.add_input_node(std::array{1});
    std::span<float> c_vals = input_c.value_elements();
    std::ranges::fill(c_vals, 50);

    kaad::Node input_d = rec.add_input_node(std::array{2, 3, 1});
    std::span<float> d_vals = input_d.value_elements();
    std::ranges::fill(d_vals, 20);

    // NOLINTEND(readability-magic-numbers)

    // Add computation nodes to graph via operators.
    kaad::Node a_plus_b = add(rec, input_a, input_b); // [3,2] + [3,2] -> [3,2]
    kaad::Node ab_plus_c = add(rec, a_plus_b, input_c); // [3,2] + [1] -> [3,2]
    kaad::Node res = add(rec, ab_plus_c, input_d); // [3,2] + [2,3,1] -> [2,3,2]

    // Reset the graph.
    rec.reset();

    // Evaluate 'res'.
    rec.evaluate(std::array{res});

    // Compute the gradient of res w.r.t. to a, b, c and d.
    rec.getGradient(res, std::array{input_a, input_b, input_c, input_d});

    // Print values of nodes.
    std::cout << "A:\n" << input_a << '\n';
    std::cout << "B:\n" << input_b << '\n';
    std::cout << "C:\n" << input_c << '\n';
    std::cout << "D:\n" << input_d << '\n';
    std::cout << "Res:\n" << res << '\n';

    return 0;
}
