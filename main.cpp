#include <iostream>
#include "kaad.h"

using namespace kaad;
using namespace std;

auto append_rec(CompGraph<double>& rec, initializer_list<int> _shape) {
    size_t nDims = _shape.size();
    int* shape = new int[nDims];
    copy(_shape.begin(), _shape.end(), shape);
    size_t len = 1;
    for (int dim : _shape) {
        len *= dim;
    }
    double* val = new double[len];
    for (int i = 0; i < len; i++) {
        val[i] = i + 1;
    }
    return rec.append(shape, nDims, val, len);
}

int main() {
    system("clear");
    CompGraph<double> rec;

    auto a = append_rec(rec, {2,4});

    auto b = append_rec(rec, {2,4,3});

    auto c = matmul(rec, a, b);

    cout << "A:\n" << a->value << endl;
    cout << "B:\n" << b->value << endl;

    rec.reset();

    auto e = rec.evaluate(c);

    cout << "C:\n" << c->value << endl;

    auto g = rec.getGradient(c, a, b);

    cout << "dA\n" << *g[0] << endl;
    cout << "dB\n" << *g[1] << endl;
}

