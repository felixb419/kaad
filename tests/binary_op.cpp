#include "../kaad.hpp"
#include <iostream>

int main() {
    kaad::Computation_graph rec;

    auto a = rec.append(std::vector<int>{3, 2}, 10);
    auto b = rec.append(std::vector<int>{3, 2}, 30);
    auto c = rec.append(std::vector<int>{1}, 50);
    auto d = rec.append(std::vector<int>{2, 3, 1}, 20);

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
