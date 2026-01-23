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

    rec.reset();

    auto e = rec.evaluate(c);

    auto g = rec.getGradient(c, a, b, d);

    cout << "A:\n" << a << endl;
    cout << "B:\n" << b << endl;
    cout << "D:\n" << d << endl;
    cout << "C:\n" << c << endl;

    return 0;
}
