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
    kaad::Node input_a = rec.add_input_node(std::array{3, 5});
    std::span<float> a_vals =
        input_a.value_elements(); // span to represent the element array of a
    std::ranges::fill(a_vals, 10);

    kaad::Node input_b = rec.add_input_node(std::array{5, 8});
    std::span<float> b_vals = input_b.value_elements();
    std::ranges::fill(b_vals, 50);

    kaad::Node input_c = rec.add_input_node(std::array{2, 2, 8, 2});
    std::span<float> c_vals = input_c.value_elements();
    std::ranges::fill(c_vals, 20);

    // Add computation nodes to graph via operators.
    kaad::Node prod_ab = matmul(rec, input_a, input_b);
    kaad::Node res = matmul(rec, prod_ab, input_c);

    // NOLINTEND(readability-magic-numbers)

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
