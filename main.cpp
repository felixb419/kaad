#include "kaad.h"

int main() {
    system("clear");
    //int* s1 = new int[] {3,2,2,4};
    //double* v1 = new double[] {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48};
    int* s1 = new int[] {1,2};
    double* v1 = new double[] {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16};
    Tensor<double> A(s1, 2, v1, 2);

    int* s2 = new int[] {2,2,4,1};
    double* v2 = new double[] {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32};
    Tensor<double> B(s2, 4, v2, 16);

    int* s3 = new int[] {2,2,4,2};
    double* v3 = new double[] {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32};
    Tensor<double> N(s3, 4, v3, 32);

    Tensor<double> X(10);
    Tensor<double> Y(5);

    Recorder<double> rec;
    int a = rec.append(move(A));
    int b = rec.append(move(B));
    int n = rec.append(move(N));
    int x = rec.append(move(X));
    int y = rec.append(move(Y));

    //int c = pow(rec, pow(rec, x, pow(rec, n, pow(rec, a, b))), y);
    int c = abs(rec, b);

    //cout << "A:\n" << rec.nodes[a].value << endl;
    //cout << "N:\n" << rec.nodes[n].value << endl;

    auto e = rec.evaluate(c);

    cout << "C:\n" << rec.nodes[c].value << endl;

    auto g = rec.getGradient(c, a, b, n, x, y);

    //cout << "dA\n" << *g[0] << endl;
    cout << "dB\n" << *g[1] << endl;
    //cout << "dN\n" << *g[2] << endl;
    //cout << "dX\n" << *g[3] << endl;
    //cout << "dY\n" << *g[4] << endl;
}
