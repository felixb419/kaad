#include "../kaad.hpp"
#include "../scalar.hpp"
#include <algorithm>
#include <numeric>

int main() {
    kaad::Computation_graph rec;

    std::vector<float> a_elements(30);
    std::iota(a_elements.begin(), a_elements.end(), 0);

    kaad::Node_handle a = rec.append(std::vector<int>{3, 5, 2}, a_elements);

    kaad::Node_handle c = transpose(rec, a);

    rec.reset();

    auto e = rec.evaluate(c);

    auto g = rec.getGradient(c, a);

    std::cout << "A:\n" << a << std::endl;
    std::cout << "C:\n" << c << std::endl;

    return 0;
}
