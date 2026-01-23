#include "../kaad.hpp"
#include <iostream>

using namespace kaad;
using namespace std;
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
    Computation_graph<T> rec;

    std::vector<int> a_shape = {8, 8};
    auto a = rec.append(a_shape, 10);

    std::vector<int> b_shape = {4, 5};
    auto b = rec.append(b_shape, 50);

    std::vector<int> d_shape = {4, 5};
    auto d = rec.append(d_shape, 20);

    call_all_operators(rec, a, b);

    return 0;
}
