#include "common.hpp" // for kaad::detail::print_tensor
#include <algorithm>  // for std::fill
#include <cstddef>    // for std::size_t
#include <iostream>   // for std::ostream, std::cout
#include <vector>     // for std::vector

namespace kaad {

/**
 * @brief Lightweight view into a tensor's shape, stride, and data.
 *
 * This struct allows for non-owning access to tensor metadata and values,
 * useful for read-only operations or efficient slicing without copying.
 *
 * @tparam T The data type stored in the tensor.
 */
template <typename T> struct tView {
    std::vector<int> &shape; ///< Referehce to the vector containing the
                             ///< size of the tensor in each dimension.
    std::vector<int> &
        stride; ///< Referehce to the vector containing the stride of the tensor
                ///< (steps needed to move one element in each dimension).
    std::vector<T> &
        val; ///< Referehce to the vector containing the elements of the Tensor.

    /**
     * @brief Default constructor.
     */
    tView() {}

    /**
     * @brief Constructs a tensor view.
     * @param shape Reference to shape vector.
     * @param stride Reference to stride vector.
     * @param val Reference to element vector.
     */
    tView(std::vector<int> &shape, std::vector<int> &stride,
          std::vector<T> &val)
        : shape(shape), stride(stride), val(val) {}

    size_t nDims() const { return this->shape.size(); }

    /**
     * @brief Overloads the stream output operator to print the tensor view.
     *
     * @param stream Output stream.
     * @param view The tensor view to print.
     * @return std::ostream& The updated output stream.
     */
    friend std::ostream &operator<<(std::ostream &stream, tView<T> view) {
        if (view.nDims() == 0) {
            std::cout << "[]";
        } else {
            int *cords = new int[view.nDims()];
            std::fill(cords, cords + view.nDims(), 0);
            int indent = 0;

            detail::print_tensor(stream, cords, view.shape, view.stride,
                                 view.nDims(), view.val, 0, indent);

            delete[] cords;
        }
        return stream;
    }
};

} // namespace kaad
