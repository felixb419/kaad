#include "kaad.hpp"
#include <iostream>

using namespace kaad;
using namespace std;

int main() {
    system("clear");
    CompGraph<double> rec;

    std::vector<int> a_shape = {4, 1};
    auto a = rec.append(a_shape, 10);

    std::vector<int> b_shape = {4, 5};
    auto b = rec.append(b_shape, 50);

    std::vector<int> d_shape = {4, 5};
    auto d = rec.append(d_shape, 20);

    auto c = add(rec, exp(rec, a), mul(rec, b, d));

    auto c2 = dot(rec, a, b);
    // c2 = matmul(rec, a, b);
    c2 = outer(rec, a, b);

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
