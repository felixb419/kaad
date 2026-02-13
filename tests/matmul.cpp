#include "../kaad.hpp"
#include "../scalar.hpp"
#include <iostream>

int main() {
    kaad::Computation_graph rec;

    kaad::Node_handle a = rec.add_input_node(std::array{3, 5});
    kaad::Node_handle b = rec.add_input_node(std::array{5, 8});
    kaad::Node_handle d = rec.add_input_node(std::array{2, 2, 8, 2});

    kaad::Node_handle ab = matmul(rec, a, b);
    kaad::Node_handle c = matmul(rec, ab, d);

    rec.reset();

    auto e = rec.evaluate(c);

    auto g = rec.getGradient(c, a, b, d);

    std::cout << "A:\n" << a << std::endl;
    std::cout << "B:\n" << b << std::endl;
    std::cout << "D:\n" << d << std::endl;
    std::cout << "C:\n" << c << std::endl;

    return 0;
}
