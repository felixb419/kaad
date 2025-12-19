#include "common.hpp" // for kaad::detail::print_tensor
#include <algorithm>  // for std::fill
#include <cstddef>    // for std::size_t
#include <iostream>   // for std::ostream, std::cout

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
    /// Pointer to the shape array, where shape[i] is the size along dimension
    /// i.
    int *shape = nullptr;
    /// Pointer to the stride array, where stride[i] gives the step to the next
    /// element in dimension i.
    int *stride = nullptr;
    /// Number of dimensions in the tensor.
    std::size_t nDims = 0;
    /// Pointer to the flat data array storing the tensor elements.
    T *val = nullptr;
    /// Total number of elements in the tensor.
    std::size_t len = 0;

    /**
     * @brief Default constructor.
     */
    tView() {}

    /**
     * @brief Copy constructor.
     */
    tView(const tView &other)
        : shape(other.shape), stride(other.stride), nDims(other.nDims),
          val(other.val), len(other.len) {}

    /**
     * @brief Constructs a tView from shape, stride, and data pointer.
     *
     * @param shape Pointer to the shape array.
     * @param stride Pointer to the stride array.
     * @param nDims Number of dimensions.
     * @param val Pointer to the tensor values.
     * @param len Total number of elements.
     */
    tView(int *shape, int *stride, std::size_t nDims, T *val, std::size_t len)
        : shape(shape), stride(stride), nDims(nDims), val(val), len(len) {}

    /**
     * @brief Overloads the stream output operator to print the tensor view.
     *
     * @param stream Output stream.
     * @param view The tensor view to print.
     * @return std::ostream& The updated output stream.
     */
    friend std::ostream &operator<<(std::ostream &stream, tView<T> view) {
        if (view.nDims == 0) {
            std::cout << "[]";
        } else {
            int *cords = new int[view.nDims];
            std::fill(cords, cords + view.nDims, 0);
            int indent = 0;

            detail::print_tensor(stream, cords, view.shape, view.stride,
                                 view.nDims, view.val, 0, indent);

            delete[] cords;
        }
        return stream;
    }
};

} // namespace kaad
