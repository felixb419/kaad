#include "kaad.h"

int main() {
    Tensor<int> A({3,4},200);
    Tensor<int> B({4,2},100);

    Recorder<int> rec;
    int a = rec.append(move(A));
    int b = rec.append(move(B));

    int c = matmul(rec, a, b);

    auto e = rec.evaluate(c);

    int vals[] = {a,b};
    auto g = rec.getGradient(c, vals);

    cout << *e[0] << endl;
    cout << *g[0] << endl;
    cout << *g[1] << endl;
}
