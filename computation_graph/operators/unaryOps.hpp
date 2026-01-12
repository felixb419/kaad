#pragma once

#include "../../tensorfuncs/adjoint_ops.hpp" // for tensorfuncs::adjoint
#include "../../tensorfuncs/kernels.hpp"     // for Sum, Null, Null::Op
#include "../../tensorfuncs/primal_ops.hpp"  // for tensorfuncs::primal
#include "../../tensorfuncs/strides.hpp"     // for mean_dim, slice, sum_dim
#include "../../utils.hpp"                   // for print_arr, transp
#include "dispatchers.hpp"  // for KAAD_MAX_NDIMS, get_meanDim, get_meanDim...
#include <algorithm>        // for std::copy, std::fill
#include <cstddef>          // for size_t
#include <initializer_list> // for std::initializer_list
#include <memory>           // for std::make_unique
#include <sstream>   // for std::operator<<, std::basic_ostream::operator<<
#include <stdexcept> // for std::invalid_argument

namespace kaad {

template <typename T, class Kernel> struct Node_unary;
template <typename T> class Tensor;
template <typename T> struct Computation_graph;
template <typename T> struct INode;
template <typename T> struct Node_mean;
template <typename T> struct Node_mean_dim;
template <typename T> struct Node_slice;
template <typename T> struct Node_sum_dim;
template <typename T> struct Node_transp;

/**
 * @brief Contains a collection of unary functions for pointwise version of the
 * operation and gradient of a given unary Kernel.
 *
 * @tparam T Datatype the operations are performed on (e.g. float, double, ...).
 * @tparam Kernel Kernel the functions should be using.
 */
template <typename T, class Kernel> struct UnaryKernels {
    using Op = class Kernel::Op;
    using Grad = class Kernel::Grad;
    tensorfuncs::primal::unary::pointwise_fn<T, Op> op =
        tensorfuncs::primal::unary::pointwise<T, Op>;
    tensorfuncs::adjoint::unary::pointwise_fn<T, Grad> grad =
        tensorfuncs::adjoint::unary::pointwise<T, Grad>;
};

/**
 * @internal
 * @brief Internal helper function not intended for direct user calls.
 *
 * Adds a generalized unary operation node to the computation graph `rec`.
 * Applies the unary operation specified by `kernels` to the input tensor node
 * `A_ptr`.
 *
 * @tparam T The data type of tensor elements.
 * @tparam Kernel The kernel providing forward operation and gradient.
 *
 * @param rec Reference to the computation graph.
 * @param A_ptr Pointer to the input node.
 * @param kernels Unary operation and gradient kernels.
 * @return Pointer to the newly created unary operation node.
 */
template <typename T, class Kernel>
INode<T> *unOperator(Computation_graph<T> &rec, INode<T> *A_ptr,
                     const UnaryKernels<T, Kernel> kernels) {
    Tensor<T> &A = A_ptr->value;

    auto newNode = std::make_unique<Node_unary<T, Kernel>>(
        kernels.op, kernels.grad, A_ptr, A.shape);
    auto raw = newNode.get();
    rec.nodes.push_back(std::move(newNode));
    raw->end = raw->value.data() + raw->value.nElems();

    return raw;
}

/**
 * @brief Adds a unary negation node (-A) to the computation graph.
 *
 * @tparam T The data type of the tensor values.
 *
 * @param rec The computation graph to which the node will be added.
 * @param A_ptr Pointer to the input tensor node A.
 * @return A pointer to the new node representing the negated tensor,
 *         with the same shape as A.
 */
template <typename T>
INode<T> *negative(Computation_graph<T> &rec, INode<T> *A_ptr) {
    static const UnaryKernels<T, class Kernels::Neg<T>> negK;
    return unOperator(rec, A_ptr, negK);
}

/**
 * @brief Adds a unary square node (A²) to the computation graph.
 *
 * @tparam T The data type of the tensor values.
 *
 * @param rec The computation graph to which the node will be added.
 * @param A_ptr Pointer to the input tensor node A.
 * @return A pointer to the new node representing the element-wise square of A,
 *         with the same shape as the input tensor.
 */
template <typename T>
INode<T> *square(Computation_graph<T> &rec, INode<T> *A_ptr) {
    static const UnaryKernels<T, class Kernels::Square<T>> squareK;
    return unOperator(rec, A_ptr, squareK);
}

/**
 * @brief Adds a unary square root node (√A) to the computation graph.
 *
 * @tparam T The data type of the tensor values.
 *
 * @param rec The computation graph to which the node will be added.
 * @param A_ptr Pointer to the input tensor node A.
 * @return A pointer to the new node representing the element-wise square root
 * of A, with the same shape as the input tensor.
 */
template <typename T>
INode<T> *sqrt(Computation_graph<T> &rec, INode<T> *A_ptr) {
    static const UnaryKernels<T, class Kernels::Sqrt<T>> sqrtK;
    return unOperator(rec, A_ptr, sqrtK);
}

/**
 * @brief Adds a unary logarithm node (log(A)) to the computation graph.
 *
 * @tparam T The data type of the tensor values.
 *
 * @param rec The computation graph to which the node will be added.
 * @param A_ptr Pointer to the input tensor node A.
 * @return A pointer to the new node representing the element-wise logarithm
 * of A, with the same shape as the input tensor.
 */
template <typename T>
INode<T> *log(Computation_graph<T> &rec, INode<T> *A_ptr) {
    static const UnaryKernels<T, class Kernels::Log<T>> logK;
    return unOperator(rec, A_ptr, logK);
}

/**
 * @brief Adds a unary exponent node (e^A) to the computation graph.
 *
 * @tparam T The data type of the tensor values.
 *
 * @param rec The computation graph to which the node will be added.
 * @param A_ptr Pointer to the input tensor node A.
 * @return A pointer to the new node representing the element-wise exponent
 * of A, with the same shape as the input tensor.
 */
template <typename T>
INode<T> *exp(Computation_graph<T> &rec, INode<T> *A_ptr) {
    static const UnaryKernels<T, class Kernels::Exp<T>> expK;
    return unOperator(rec, A_ptr, expK);
}

/**
 * @brief Adds a unary absolute value node (|A|) to the computation graph.
 *
 * @tparam T The data type of the tensor values.
 *
 * @param rec The computation graph to which the node will be added.
 * @param A_ptr Pointer to the input tensor node A.
 * @return A pointer to the new node representing the element-wise absolute
 * value of A, with the same shape as the input tensor.
 */
template <typename T>
INode<T> *abs(Computation_graph<T> &rec, INode<T> *A_ptr) {
    static const UnaryKernels<T, class Kernels::Abs<T>> absK;
    return unOperator(rec, A_ptr, absK);
}

/**
 * @brief Adds a unary transpose node to the computation graph.
 *
 * Transposes the input tensor node `A_ptr` according to the permutation `perm`.
 * If `perm` is empty, the tensor is fully transposed by reversing its
 * dimensions.
 *
 * @tparam T The data type of the tensor values.
 *
 * @param rec The computation graph to which the node will be added.
 * @param A_ptr Pointer to the input tensor node A.
 * @param perm An optional initializer list specifying the permutation of axes.
 *             If not provided or empty, a full transpose (reverse of all axes)
 * is performed.
 * @return A pointer to the new node representing the transposed tensor,
 *         with shape adjusted according to `perm` or full transpose.
 */
template <typename T>
INode<T> *transpose(Computation_graph<T> &rec, INode<T> *A_ptr,
                    std::initializer_list<int> perm = {}) {
    int recLen = rec.nodes.size();
    Tensor<T> &A = A_ptr->value;

    if (A.nDims() < 2) {
        std::ostringstream errmsg;
        errmsg << "shape error in node[" << recLen
               << "] (transpose), A.nDims() hast to be > 1 (shape1=";
        print_arr(A.shape.data(), A.shape.data() + A.nDims(), errmsg);
        errmsg << ")";
        throw std::invalid_argument(errmsg.str());
    }

    std::vector<int> shape_T(A.nDims());
    std::vector<int> stride_T(A.nDims());
    if (perm.size() == 0) {
        std::copy(A.shape.begin(), A.shape.end(), shape_T.begin());
        std::copy(A.stride.begin(), A.stride.end(), stride_T.begin());

        transp(A.shape.data(), A.stride.data(), A.nDims(), shape_T.data(),
               stride_T.data());
    } else {
        if (perm.size() != A.nDims()) {
            std::ostringstream errmsg;
            errmsg << "argument error in node[" << recLen
                   << "] (transpose), perm.size() has to be same as A.nDims() "
                      "(perm=";
            print_arr(perm.begin(), perm.end(), errmsg);
            errmsg << ", shape1=";
            print_arr(A.shape.data(), A.shape.data() + A.nDims(), errmsg);
            errmsg << ")";
            throw std::invalid_argument(errmsg.str());
        }

        int *count = new int[A.nDims()];
        std::fill(count, count + A.nDims(), 0);

        int *sh = shape_T.data();
        int *st = stride_T.data();
        for (int idx : perm) {

            count[idx]++;

            *(sh++) = A.shape[idx];
            *(st++) = A.stride[idx];
        }
        for (int *p = count; p != count + A.nDims(); p++) {
            if (*p != 1) {
                std::ostringstream errmsg;
                errmsg
                    << "argument error in node[" << recLen
                    << "] (transpose), invalid permutation, perm has to "
                       "contain index of every dimension exactly once (perm=";
                print_arr(perm.begin(), perm.end(), errmsg);
                errmsg << ")";
                throw std::invalid_argument(errmsg.str());
            }
        }
    }

    auto newNode =
        std::make_unique<Node_transp<T>>(A_ptr, shape_T, stride_T, A.nDims());
    auto raw_ptr = newNode.get();
    newNode->A_end = A.data() + A.nElems();
    newNode->C_end = raw_ptr->value.data() + raw_ptr->value.nElems();
    rec.nodes.push_back(std::move(newNode));

    return raw_ptr;
}

/**
 * @brief Adds a unary sum node to the computation graph.
 *
 * Computes the sum of all elements in the input tensor node `A_ptr`,
 * producing a scalar tensor node containing the total sum.
 *
 * @tparam T The data type of the tensor values.
 *
 * @param rec The computation graph to which the node will be added.
 * @param A_ptr Pointer to the input tensor node A.
 * @return A pointer to the new node representing the scalar sum of all elements
 * of A.
 */
template <typename T>
INode<T> *sum(Computation_graph<T> &rec, INode<T> *A_ptr) {
    int recLen = rec.nodes.size();
    Tensor<T> &A = A_ptr->value;

    using Kernel = class Kernels::Sum<T>;
    using Op = typename Kernel::Op;
    using Grad = typename Kernel::Grad;
    tensorfuncs::primal::unary::pointwise_fn<T, Op> op =
        tensorfuncs::primal::unary::scalarOut<T, Op>;
    tensorfuncs::adjoint::unary::pointwise_fn<T, Grad> grad =
        tensorfuncs::adjoint::unary::scalarOut<T, Grad>;
    auto newNode =
        std::make_unique<Node_unary<T, Kernel>>(op, grad, A_ptr, (T)0);
    newNode->end = A.data() + A.nElems();
    rec.nodes.push_back(std::move(newNode));
    return rec.nodes.back().get();
}

/**
 * @brief Adds a sum node to the computation graph that sums elements along a
 * specified dimension.
 *
 * Computes the sum of elements in the input tensor node `A_ptr` along the given
 * dimension `dim`. The resulting tensor shape depends on the `keepNDims` flag:
 * - If `keepNDims` is false (default), the dimension `dim` is removed from the
 * output shape.
 * - If `keepNDims` is true, the dimension `dim` is retained with size 1.
 *
 * @tparam T The data type of the tensor values.
 *
 * @param rec The computation graph to which the node will be added.
 * @param A_ptr Pointer to the input tensor node A.
 * @param dim The dimension along which to sum.
 * @param keepNDims If true, retains the summed dimension with size 1; if false,
 * removes it.
 * @return A pointer to the new node representing the tensor after summation
 * along the specified dimension.
 */
template <typename T>
INode<T> *sum(Computation_graph<T> &rec, INode<T> *A_ptr, int dim,
              bool keepNDims = false) {
    int recLen = rec.nodes.size();
    Tensor<T> &A = A_ptr->value;

    if (dim < 0 || dim >= A.nDims()) {
        std::ostringstream errmsg;
        errmsg << "argument error in node[" << recLen
               << "] (sum), dim has to be a valid index of A.shape (dim=" << dim
               << ", A.nDims()=" << A.nDims() << ")" << std::endl;
        throw std::invalid_argument(errmsg.str());
    }

    if (A.nDims() == 1) {
        return sum(rec, A_ptr);
    }

    size_t newLen = A.nDims();
    std::vector<int> newShape(newLen);
    if (keepNDims) {
        std::copy(A.shape.begin(), A.shape.end(), newShape.begin());
        newShape[dim] = 1;

    } else {
        newLen--;
        std::copy(A.shape.begin(), A.shape.begin() + dim, newShape.begin());
        std::copy(A.shape.begin() + dim + 1, A.shape.end(),
                  newShape.begin() + dim);
    }

    auto newNode = std::make_unique<Node_sum_dim<T>>(A_ptr, newShape, newLen);
    auto raw_ptr = newNode.get();
    if (A.nDims() <= KAAD_MAX_NDIMS) {
        raw_ptr->val_func = detail::Dispatchers::get_sumDim<T>()[A.nDims()];
        raw_ptr->grad_func =
            detail::Dispatchers::get_sumDim_grad<T>()[A.nDims()];
    }

    Strides::sum_dim<T>(*raw_ptr, dim);
    rec.nodes.push_back(std::move(newNode));
    return rec.nodes.back().get();
}

/**
 * @brief Adds a unary mean node to the computation graph.
 *
 * Computes the mean (average) of all elements in the input tensor node `A_ptr`,
 * producing a scalar tensor node containing the total average.
 *
 * @tparam T The data type of the tensor values.
 *
 * @param rec The computation graph to which the node will be added.
 * @param A_ptr Pointer to the input tensor node A.
 * @return A pointer to the new node representing the scalar mean of all
 * elements of A.
 */
template <typename T>
INode<T> *mean(Computation_graph<T> &rec, INode<T> *A_ptr) {
    int recLen = rec.nodes.size();
    Tensor<T> &A = A_ptr->value;

    auto newNode = std::make_unique<Node_mean<T>>(A_ptr, (T)0);
    newNode->A_end = A.data() + A.nElems();
    newNode->dA_end =
        newNode->A->gradient.data() + newNode->A->gradient.nElems();
    newNode->divisor = A.val.size();
    rec.nodes.push_back(std::move(newNode));
    return rec.nodes.back().get();
}

/**
 * @brief Adds a mean node to the computation graph that computes the mean along
 * a specified dimension.
 *
 * Computes the mean of elements in the input tensor node `A_ptr` along the
 * given dimension `dim`. The resulting tensor shape depends on the `keepNDims`
 * flag:
 * - If `keepNDims` is false (default), the dimension `dim` is removed from the
 * output shape.
 * - If `keepNDims` is true, the dimension `dim` is retained with size 1.
 *
 * @tparam T The data type of the tensor values.
 *
 * @param rec The computation graph to which the node will be added.
 * @param A_ptr Pointer to the input tensor node A.
 * @param dim The dimension along which to compute the mean.
 * @param keepNDims If true, retains the mean-reduced dimension with size 1; if
 * false, removes it.
 * @return A pointer to the new node representing the tensor after mean
 * reduction along the specified dimension.
 */
template <typename T>
INode<T> *mean(Computation_graph<T> &rec, INode<T> *A_ptr, int dim,
               bool keepNDims = 0) {
    int recLen = rec.nodes.size();
    Tensor<T> &A = A_ptr->value;

    if (dim < 0 || dim >= A.nDims()) {
        std::ostringstream errmsg;
        errmsg << "argument error in node[" << recLen
               << "] (mean), dim has to be a valid index of A.shape (dim="
               << dim << ", A.nDims()=" << A.nDims() << ")" << std::endl;
        throw std::invalid_argument(errmsg.str());
    }

    if (A.nDims() == 1) {
        return mean(rec, A_ptr);
    }

    size_t newLen = A.nDims();
    std::vector<int> newShape(newLen);
    if (keepNDims) {
        std::copy(A.shape.begin(), A.shape.end(), newShape.begin());
        newShape[dim] = 1;

    } else {
        newLen--;
        std::copy(A.shape.begin(), A.shape.begin() + dim, newShape.begin());
        std::copy(A.shape.begin() + dim + 1, A.shape.end(),
                  newShape.begin() + dim);
    }

    auto newNode = std::make_unique<Node_mean_dim<T>>(A_ptr, newShape, newLen);
    auto raw_ptr = newNode.get();
    if (A.nDims() <= KAAD_MAX_NDIMS) {
        raw_ptr->val_func = detail::Dispatchers::get_meanDim<T>()[A.nDims()];
        raw_ptr->grad_func =
            detail::Dispatchers::get_meanDim_grad<T>()[A.nDims()];
    }

    Strides::mean_dim<T>(*raw_ptr, dim);
    rec.nodes.push_back(std::move(newNode));
    return rec.nodes.back().get();
}

/**
 * @brief Adds a slice node to the computation graph.
 *
 * Extracts a slice from the input tensor node `A_ptr`, starting at the
 * specified `offset` and extending for the given `size` along each dimension.
 *
 * @tparam T The data type of the tensor values.
 *
 * @param rec The computation graph to which the node will be added.
 * @param A_ptr Pointer to the input tensor node A.
 * @param size An initializer list specifying the size (length) of the slice
 * along each dimension.
 * @param offset An initializer list specifying the starting indices (offset)
 * for the slice along each dimension.
 * @return A pointer to the new node representing the sliced tensor.
 */
template <typename T>
INode<T> *slice(Computation_graph<T> &rec, INode<T> *A_ptr,
                std::initializer_list<int> size,
                std::initializer_list<int> offset) {
    int recLen = rec.nodes.size();
    Tensor<T> &A = A_ptr->value;

    if (size.size() > A.nDims()) {
        std::ostringstream errmsg;
        errmsg << "argument error in node[" << recLen
               << "] (slice), length of size cant be bigger than A.nDims(), "
                  "(size length="
               << size.size() << ", A.nDims()=" << A.nDims() << ")"
               << std::endl;
        throw std::invalid_argument(errmsg.str());
    }
    if (offset.size() > A.nDims()) {
        std::ostringstream errmsg;
        errmsg << "argument error in node[" << recLen
               << "] (slice), length of offset cant be bigger than A.nDims(), "
                  "(offset length="
               << offset.size() << ", A.nDims()=" << A.nDims() << ")"
               << std::endl;
        throw std::invalid_argument(errmsg.str());
    }

    int diff = A.nDims() - offset.size();
    std::vector<int> size_owned(A.nDims());
    std::copy(A.shape.begin(), A.shape.begin() + diff, size_owned.begin());
    std::copy(size.begin(), size.begin() + size.size(),
              size_owned.begin() + diff);

    std::vector<int> offset_owned(A.nDims());
    std::fill(offset_owned.begin(), offset_owned.begin() + diff, 0);
    std::copy(offset.begin(), offset.begin() + offset.size(),
              offset_owned.begin() + diff);

    for (int i = 0; i < A.nDims(); i++) {
        if (offset_owned[i] + size_owned[i] > A.shape[i]) {
            std::ostringstream errmsg;
            errmsg << "argument error in node[" << recLen
                   << "] (slice), offset[" << i << "] with length[" << i
                   << "] would overflow shape of A, (offset=";
            print_arr(offset_owned.data(), offset_owned.data() + A.nDims(),
                      errmsg);
            errmsg << ", length=";
            print_arr(size_owned.data(), size_owned.data() + A.nDims(), errmsg);
            errmsg << ", shape=";
            print_arr(A.shape.data(), A.shape.data() + A.nDims(), errmsg);
            errmsg << ")" << std::endl;
            throw std::invalid_argument(errmsg.str());
        }
    }

    size_t newLen = A.nDims();
    std::vector<int> newShape(newLen);
    std::copy(size_owned.begin(), size_owned.end(), newShape.begin());

    auto newNode = std::make_unique<Node_slice<T>>(A_ptr, newShape, newLen);
    auto raw_ptr = newNode.get();

    if (A.nDims() < KAAD_MAX_NDIMS) {
        raw_ptr->val_func = detail::Dispatchers::get_slice<T>()[A.nDims()];
        raw_ptr->grad_func =
            detail::Dispatchers::get_slice_grad<T>()[A.nDims()];
    }

    Strides::slice(*raw_ptr, offset_owned.data());

    rec.nodes.push_back(std::move(newNode));
    return rec.nodes.back().get();
}
} // namespace kaad
