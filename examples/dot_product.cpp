#include <algorithm> // for __fill_fn, fill
#include <array>     // for array
#include <iostream>  // for basic_ostream, operator<<
#include <kaad/kaad.hpp>
#include <span> // for span

int main() {
    // Create computation graph.
    kaad::Graph rec;

    // NOLINTBEGIN(readability-magic-numbers)

    // Add input nodes to the graph.
    kaad::Node input_a = rec.add_input_node(std::array{5});
    std::span<float> a_vals =
        input_a.value_elements(); // Add input nodes to the graph
    std::ranges::fill(a_vals, 10);

    kaad::Node input_b = rec.add_input_node(std::array{5});
    std::span<float> b_vals = input_b.value_elements();
    std::ranges::fill(b_vals, 50);

    kaad::Node input_c = rec.add_input_node(std::array{5});
    std::span<float> c_vals = input_c.value_elements();
    std::ranges::fill(c_vals, 20);

    // NOLINTEND(readability-magic-numbers)

    // Add computation nodes to graph via operators.
    kaad::Node dot_ab = dot(rec, input_a, input_b); // [5] * [5] -> [1]
    kaad::Node res = dot(rec, dot_ab, input_c);     // [1] * [5] -> [1]

    // Reset the graph.
    rec.reset();

    // Evaluate 'res'.
    rec.evaluate(std::array{res});

    // Compute the gradient of res w.r.t. to a, b and c.
    rec.getGradient(res, std::array{input_a, input_b, input_c});

    // Print values of nodes.
    std::cout << "A:\n" << input_a << '\n';
    std::cout << "B:\n" << input_b << '\n';
    std::cout << "C:\n" << input_c << '\n';
    std::cout << "Res:\n" << res << '\n';

    return 0;
}
