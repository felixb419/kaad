#include "../kaad.hpp"
#include <iostream>

int main() {
    kaad::Computation_graph rec;

    auto a = rec.add_input_node(std::array{3, 2});
    auto b = rec.add_input_node(std::array{3, 2});
    auto c = rec.add_input_node(std::array{1});
    auto d = rec.add_input_node(std::array{2, 3, 1});

    auto ab = add(rec, a, b);
    auto abc = add(rec, ab, c);
    auto res = add(rec, abc, d);

    rec.reset();

    rec.evaluate(res);

    rec.getGradient(res, a, b, d, res);

    std::cout << "A:\n" << a << std::endl;
    std::cout << "B:\n" << b << std::endl;
    std::cout << "D:\n" << d << std::endl;
    std::cout << "Res:\n" << res << std::endl;

    return 0;
}
