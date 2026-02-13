#include "../kaad.hpp"
#include <iostream>

int main() {
    kaad::Computation_graph rec;

    kaad::Node_handle a = rec.append(std::array{5});
    kaad::Node_handle b = rec.append(std::array{5});
    kaad::Node_handle d = rec.append(std::array{5});

    kaad::Node_handle ab = dot(rec, a, b);
    kaad::Node_handle c = dot(rec, ab, d);

    rec.reset();

    auto e = rec.evaluate(c);

    auto g = rec.getGradient(c, a, b, d);

    std::cout << "A:\n" << a << std::endl;
    std::cout << "B:\n" << b << std::endl;
    std::cout << "D:\n" << d << std::endl;
    std::cout << "C:\n" << c << std::endl;

    return 0;
}
