#include "../kaad.hpp"
#include <algorithm>
#include <numeric>

using namespace kaad;
using namespace std;
using T = double;

int main() {
    system("clear");
    Computation_graph<T> rec;

    std::vector<int> a_shape = {3, 5, 2};
    std::vector<int> a_elements(30);
    std::iota(a_elements.begin(), a_elements.end(), 0);

    auto a = rec.append(a_shape, a_elements);

    auto c = transpose(rec, a);

    rec.reset();

    auto e = rec.evaluate(c);

    auto g = rec.getGradient(c, a);

    cout << "A:\n" << a << endl;
    cout << "C:\n" << c << endl;

    return 0;
}
