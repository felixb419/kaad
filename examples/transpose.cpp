#include <array>
#include <iostream>
#include <kaad/kaad.hpp>
#include <numeric>
#include <span>

int main() {
    // Create computation graph.
    kaad::Graph rec;

    // NOLINTBEGIN(readability-magic-numbers)

    // Add input nodes to the graph.
    kaad::Node input_a = rec.add_input_node(kaad::Shape{3, 5, 2});
    std::span<float> a_vals =
        input_a.value_mut()
            .elements; // span to represent the element array of a
    std::iota(a_vals.begin(), a_vals.end(), 0);

    // Add computation nodes to graph via operators.
    kaad::Node res = transpose(rec, input_a); // [3,5,2] -> [2,5,3]

    // NOLINTEND(readability-magic-numbers)

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
