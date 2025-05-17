#include "kaad.h"

int main() {
    int* s1 = new int[] {3,2,2,4};
    double* v1 = new double[] {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48};
    Tensor<double> A(s1, 4, v1, 48);
    int* s2 = new int[] {3,2,4};
    double* v2 = new double [] {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24};
    Tensor<double> N(s2, 3, v2, 24);

    Recorder<double> rec;
    int a = rec.append(move(A));
    int n = rec.append(move(N));

    int b = mean(rec, a, 1);
    int c = mul(rec, b, n);

    auto e = rec.evaluate(c);

    int vals[] = {a};
    auto g = rec.getGradient(c, vals);

    cout << *e[0] << endl;
    cout << *g[0] << endl;
}
