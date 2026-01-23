#include "kaad.hpp"
#include <iostream>

using T = double;

void call_all_operators(kaad::Computation_graph<T> &rec, kaad::Node_handle<T> a,
                        kaad::Node_handle<T> b) {
    add(rec, a, b);
    dot(rec, a, b);
    matmul(rec, a, b);
    outer(rec, a, b);

    exp(rec, a);
    transpose(rec, a, {5, 4, 3, 2, 1});
    sum(rec, a);
    sum(rec, a, 1);
    mean(rec, a);
    mean(rec, a, 1);
    slice(rec, a, {1, 1, 1}, {3, 3, 3});
}

int main() {
    system("clear");
    kaad::Computation_graph<T> rec;

    std::vector<int> a_shape = {8, 8};
    auto a = rec.append(a_shape, 10);

    auto b = rec.append(50);

    auto c = add(rec, a, b);

    std::cout << "A:\n" << a.value() << std::endl;
    std::cout << "B:\n" << b.value() << std::endl;

    rec.reset();

    auto e = rec.evaluate(c);

    std::cout << "C:\n" << c.value() << std::endl;

    auto g = rec.getGradient(c, a, b);

    std::cout << "dA\n" << *g[0] << std::endl;
    std::cout << "dB\n" << *g[1] << std::endl;

    return 0;
}
