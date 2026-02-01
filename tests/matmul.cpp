#include "../kaad.hpp"
#include "../scalar.hpp"
#include <iostream>

int main() {
    kaad::Computation_graph rec;

    kaad::Node_handle a = rec.append(std::vector<int>{3, 5}, 10);
    kaad::Node_handle b = rec.append(std::vector<int>{5, 8}, 50);
    kaad::Node_handle d = rec.append(std::vector<int>{2, 2, 8, 2}, 20);

    kaad::Node_handle ab = matmul<kaad::Scalar>(rec, a, b);
    kaad::Node_handle c = matmul<kaad::Scalar>(rec, ab, d);

    rec.reset();

    auto e = rec.evaluate(c);

    auto g = rec.getGradient(c, a, b, d);

    std::cout << "A:\n" << a << std::endl;
    std::cout << "B:\n" << b << std::endl;
    std::cout << "D:\n" << d << std::endl;
    std::cout << "C:\n" << c << std::endl;

    return 0;
}
