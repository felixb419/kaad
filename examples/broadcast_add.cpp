#include "../include/kaad/kaad.hpp"
#include <algorithm> // for __fill_fn, fill
#include <array>     // for array
#include <iostream>  // for basic_ostream, operator<<
#include <span>      // for span

int main() {
    // Create computation graph.
    kaad::Graph rec;

    // Add input nodes to the graph.
    auto a = rec.add_input_node(std::array{3, 2});
    std::span<float> a_vals =
        a.value_elements(); // span to represent the element array of a
    std::ranges::fill(a_vals, 10);

    auto b = rec.add_input_node(std::array{3, 2});
    std::span<float> b_vals = b.value_elements();
    std::ranges::fill(b_vals, 30);

    auto c = rec.add_input_node(std::array{1});
    std::span<float> c_vals = c.value_elements();
    std::ranges::fill(c_vals, 50);

    auto d = rec.add_input_node(std::array{2, 3, 1});
    std::span<float> d_vals = d.value_elements();
    std::ranges::fill(d_vals, 20);

    // Add computation nodes to graph via operators.
    auto ab = add(rec, a, b);    // [3,2] + [3,2] -> [3,2]
    auto abc = add(rec, ab, c);  // [3,2] + [1] -> [3,2]
    auto res = add(rec, abc, d); // [3,2] + [2,3,1] -> [2,3,2]

    // Reset the graph.
    rec.reset();

    // Evaluate 'res'.
    rec.evaluate(std::array{res});

    // Compute the gradient of res w.r.t. to a, b, c and d.
    rec.getGradient(res, std::array{a, b, c, d});

    // Print values of nodes.
    std::cout << "A:\n" << a << '\n';
    std::cout << "B:\n" << b << '\n';
    std::cout << "C:\n" << c << '\n';
    std::cout << "D:\n" << d << '\n';
    std::cout << "Res:\n" << res << '\n';

    return 0;
}
