#include "kaad.h"

int main() {
    system("clear");
    int* s1 = new int[] {2,4,1};
    double* v1 = new double[] {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32};
    Tensor<double> A(s1, 3, v1, 8);

    int* s2 = new int[] {3};
    double* v2 = new double[] {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32};
    Tensor<double> B(s2, 1, v2, 3);

    int* s3 = new int[] {2,4,3};
    double* v3 = new double[] {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32};
    Tensor<double> N(s3, 3, v3, 24);

    Tensor<double> X(10);
    Tensor<double> Y(5);

    CompGraph<double> rec;
    int a = rec.append(move(A));
    int b = rec.append(move(B));
    int n = rec.append(move(N));
    int x = rec.append(move(X));
    int y = rec.append(move(Y));

    int c = minimum(rec, minimum(rec, minimum(rec, a, minimum(rec, y, b)), n), x);

    //cout << "A:\n" << rec.nodes[a].value << endl;
    //cout << "B:\n" << rec.nodes[b].value << endl;
    //cout << "N:\n" << rec.nodes[n].value << endl;

    auto e = rec.evaluate(c);

    cout << "C:\n" << rec.nodes[c].value << endl;

    auto g = rec.getGradient(c, a, b, n, x, y);

    cout << "dA\n" << *g[0] << endl;
    cout << "dB\n" << *g[1] << endl;
    cout << "dN\n" << *g[2] << endl;
    cout << "dX\n" << *g[3] << endl;
    cout << "dY\n" << *g[4] << endl;
}
