#include <iostream>
#include "kaad.h"

using namespace kaad;
using namespace std;

int main() {
    system("clear");
    CompGraph<double> rec;

    int* s1 = new int[3] {2,4,1};
    double* v1 = new double[32] {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32};
    auto a = rec.append(s1, 3, v1, 8);

    int* s2 = new int[2] {1,3};
    double* v2 = new double[32] {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32};
    auto b = rec.append(s2, 2, v2, 3);

    int* s3 = new int[3] {2,4,3};
    double* v3 = new double[32] {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32};
    auto n = rec.append(s3, 3, v3, 24);

    auto x = rec.append(10);

    auto y = rec.append(5);

    auto c1 = add(rec, y, add(rec, add(rec, a, add(rec, b, x)), n));
    auto c = exp(rec, c1);


    //cout << "A:\n" << a->value << endl;
    //cout << "B:\n" << b->value << endl;
    //cout << "N:\n" << n->value << endl;

    rec.reset();
    auto e = rec.evaluate(c);

    cout << "C:\n" << c->value << endl;

    auto g = rec.getGradient(c, a, b, n, x, y);

    cout << "dA\n" << *g[0] << endl;
    cout << "dB\n" << *g[1] << endl;
    cout << "dN\n" << *g[2] << endl;
    cout << "dX\n" << *g[3] << endl;
    cout << "dY\n" << *g[4] << endl;
}

