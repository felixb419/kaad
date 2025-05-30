#include "kaad.h"

int main() {
    //int* s1 = new int[] {3,2,2,4};
    //double* v1 = new double[] {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48};
    int* s1 = new int[] {2,1,4};
    double* v1 = new double[] {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16};
    Tensor<double> A(s1, 3, v1, 8);

    int* s2 = new int[] {2,4};
    double* v2 = new double [] {1,2,3,4,5,6,7,8};
    Tensor<double> N(s2, 2, v2, 8);

    Tensor<double> X(10);

    Recorder<double> rec;
    int a = rec.append(move(A));
    int n = rec.append(move(N));
    int x = rec.append(move(X));

    int c = div(rec, a, n);

    auto e = rec.evaluate(c);

    auto g = rec.getGradient(c, a, n);

    cout << "A:\n" << rec.nodes[a].value << endl;
    cout << "N:\n" << rec.nodes[n].value << endl;
    cout << "C:\n" << rec.nodes[c].value << endl;
    cout << "dA\n" << *g[0] << endl;
    cout << "dN\n" << *g[1] << endl;
}
