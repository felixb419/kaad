#include "../kaad.hpp"
#include <iostream>

using namespace kaad;
using namespace std;
using T = double;

int main() {
    system("clear");
    Computation_graph<T> rec;

    std::vector<int> a_shape = {3, 5};
    auto a = rec.append(a_shape, 10);

    std::vector<int> b_shape = {5, 8};
    auto b = rec.append(b_shape, 50);

    std::vector<int> d_shape = {2, 2, 8, 2};
    auto d = rec.append(d_shape, 20);

    auto ab = matmul(rec, a, b);
    auto c = matmul(rec, ab, d);

    cout << "A:\n" << a.value() << endl;
    cout << "B:\n" << b.value() << endl;
    cout << "D:\n" << d.value() << endl;

    rec.reset();

    auto e = rec.evaluate(c);

    cout << "C:\n" << c.value() << endl;

    auto g = rec.getGradient(c, a, b, d);

    cout << "dA\n" << *g[0] << endl;
    cout << "dB\n" << *g[1] << endl;
    cout << "dD\n" << *g[2] << endl;

    return 0;
}
