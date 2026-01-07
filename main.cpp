#include "kaad.hpp"
#include <iostream>

using namespace kaad;
using namespace std;
using T = double;

void call_all_operators(kaad::CompGraph<T> &rec, kaad::INode<T> *a,
                        kaad::INode<T> *b) {
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
    CompGraph<T> rec;

    std::vector<int> a_shape = {8, 8};
    auto a = rec.append(a_shape, 10);

    std::vector<int> b_shape = {4, 5};
    auto b = rec.append(b_shape, 50);

    std::vector<int> d_shape = {4, 5};
    auto d = rec.append(d_shape, 20);

    auto sa = slice(rec, a, {4, 1}, {2, 2});
    auto c = add(rec, exp(rec, sa), mul(rec, b, d));

    // call_all_operators(rec, a, b);

    cout << "A:\n" << a->value << endl;
    cout << "B:\n" << b->value << endl;
    cout << "D:\n" << d->value << endl;

    rec.reset();

    auto e = rec.evaluate(c);

    cout << "C:\n" << c->value << endl;

    auto g = rec.getGradient(c, a, b, d);

    cout << "dA\n" << *g[0] << endl;
    cout << "dB\n" << *g[1] << endl;
    cout << "dD\n" << *g[2] << endl;

    return 0;
}
