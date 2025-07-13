#include <iostream>
#include "kaad.h"

using namespace kaad;
using namespace std;

int main() {
    system("clear");
    CompGraph<double> rec;

    int* s1 = new int[3] {2,2,1};
    double* v1 = new double[32] {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32};
    auto a = rec.append(s1, 3, v1, 4);

    int* s2 = new int[2] {3};
    double* v2 = new double[32] {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32};
    auto b = rec.append(s2, 1, v2, 3);

    auto c = add(rec, a, b);
    auto c_sum = sum(rec, c);

    cout << "A:\n" << a->value << endl;
    cout << "B:\n" << b->value << endl;

    rec.reset();

    auto e = rec.evaluate(c, c_sum);

    cout << "C:\n" << c->value << endl;
    cout << "Sum:\n" << c_sum->value << endl;

    auto g = rec.getGradient(c, a, b, c_sum);

    cout << "dA\n" << *g[0] << endl;
    cout << "dB\n" << *g[1] << endl;
    cout << "dSum\n" << *g[2] << endl;
}

