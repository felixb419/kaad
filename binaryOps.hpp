#pragma once

#include "dispatchers.hpp"   // for KAAD_MAX_NDIMS, get_batch_matmul
#include "tensor/tensor.hpp" // for print_arr, combine_flexible, combine_matrix
#include "tensorfuncs/adjoint_ops.hpp" // for binaryGrad, flexBinaryGrad, flexible, batch...
#include "tensorfuncs/kernels.hpp" // for Kernels::Null
#include "tensorfuncs/primal_ops.hpp" // for binaryOp, flexBinaryOp, flexible, batch_matmul
#include "tensorfuncs/strides.hpp" // for batch_matmul, flexible_binary, matmul, outer
#include "utils.hpp"
#include <cstddef> // for size_t
#include <memory>  // for std::make_unique
#include <sstream> // for std::operator<<, std::basic_ostream, std::char_traits, std::ostringstream
#include <stdexcept> // for std::invalid_argument

namespace kaad {

template <typename T, class Kernel> struct Node_binary;
template <typename T, class Kernel> struct Node_binary_flex;
template <typename T> struct CompGraph;
template <typename T> struct INode;
template <typename T> struct Node_batch_matmul;
template <typename T> struct Node_matmul;

/**
 * @brief Contains a collection of binary functions for multiple versions
 * (sclalarRhs, scalarLhs, pointwise, flexible) of the operation and gradient of
 * a given binary Kernel.
 *
 * @tparam T Datatype the operations are performed on (e.g. float, double, ...).
 * @tparam Kernel Kernel the functions should be using.
 */
template <typename T, class Kernel> struct BinaryKernels {
    using Op = class Kernel::Op;
    using Grad = class Kernel::Grad;

    binaryOp<T, Op> scalarOpRhs = tensorfuncs::primal::binary::scalarRhs<T, Op>;
    binaryOp<T, Op> scalarOpLhs = tensorfuncs::primal::binary::scalarLhs<T, Op>;
    binaryOp<T, Op> pointOp = tensorfuncs::primal::binary::pointwise<T, Op>;
    flexBinaryOp<T, Op> flexOp = tensorfuncs::primal::binary::flexible<T, Op>;

    binaryGrad<T, Grad> scalarGradRhs =
        tensorfuncs::adjoint::binary::scalarRhs<T, Grad>;
    binaryGrad<T, Grad> scalarGradLhs =
        tensorfuncs::adjoint::binary::scalarLhs<T, Grad>;
    binaryGrad<T, Grad> pointGrad =
        tensorfuncs::adjoint::binary::pointwise<T, Grad>;
    flexBinaryGrad<T, Grad> flexGrad =
        tensorfuncs::adjoint::binary::flexible<T, Grad>;
};

/**
 * @internal
 * @brief Internal helper function not intended for direct user calls.
 *
 * Adds a generalized binary operation node to the computation graph `rec`.
 * Applies the binary operation specified by `kernels` to the input tensor nodes
 * `A_ptr` and `B_ptr`.
 *
 * @tparam T The data type of tensor elements.
 * @tparam Kernel The kernel providing forward operation and gradient.
 *
 * @param rec Reference to the computation graph.
 * @param A_ptr Pointer to the first input node.
 * @param B_ptr Pointer to the second input node.
 * @param kernels Binary operation and gradient kernels.
 * @param opName A string identifier for the operation (used for debugging or
 * logging).
 * @return Pointer to the newly created binary operation node.
 */
template <typename T, class Kernel>
INode<T> *binOperator(CompGraph<T> &rec, INode<T> *A_ptr, INode<T> *B_ptr,
                      const BinaryKernels<T, Kernel> kernels,
                      const char *opName) {
    int recLen = rec.nodes.size();
    Tensor<T> &A = A_ptr->value;
    Tensor<T> &B = B_ptr->value;
    bool A_scalar = A.nDims() == 1 && A.shape[0] == 1;
    bool B_scalar = B.nDims() == 1 && B.shape[0] == 1;

    size_t newLen = std::max(A.nDims(), B.nDims());
    std::vector<int> newShape(newLen);

    if (B_scalar) {
        auto newNode = std::make_unique<Node_binary<T, Kernel>>(
            kernels.scalarOpRhs, kernels.scalarGradRhs, A_ptr, B_ptr, A.shape);
        auto raw_ptr = newNode.get();
        raw_ptr->end = raw_ptr->value.val.data() + raw_ptr->value.val.size();
        rec.nodes.push_back(move(newNode));
    } else if (A_scalar) {
        auto newNode = std::make_unique<Node_binary<T, Kernel>>(
            kernels.scalarOpLhs, kernels.scalarGradLhs, A_ptr, B_ptr, B.shape);
        auto raw_ptr = newNode.get();
        raw_ptr->end = raw_ptr->value.val.data() + raw_ptr->value.val.size();
        rec.nodes.push_back(move(newNode));
    } else if (A.nDims() == B.nDims() &&
               std::equal(A.shape.begin(), A.shape.end(), B.shape.data()) &&
               std::equal(A.stride.begin(), A.stride.end(), B.stride.data())) {

        auto newNode = std::make_unique<Node_binary<T, Kernel>>(
            kernels.pointOp, kernels.pointGrad, A_ptr, B_ptr, A.shape);
        auto raw_ptr = newNode.get();
        raw_ptr->end = raw_ptr->value.val.data() + raw_ptr->value.val.size();
        rec.nodes.push_back(move(newNode));
    } else if (combine_flexible(A.shape.data(), A.nDims(), B.shape.data(),
                                B.nDims(), newShape.data(), newLen)) {
        using Op = typename Kernel::Op;
        using Grad = typename Kernel::Grad;
        flexBinaryOp<T, Op> operation = kernels.flexOp;
        flexBinaryGrad<T, Grad> gradient = kernels.flexGrad;
        if (newLen <= KAAD_MAX_NDIMS) {
            operation = Dispatchers::get_flexOp<T, Op>()[newLen];
            gradient = Dispatchers::get_flexGrad<T, Grad>()[newLen];
        }

        auto newNode = std::make_unique<Node_binary_flex<T, Kernel>>(
            A_ptr, B_ptr, newShape);
        Strides::flexible_binary<T>(A, B, *newNode.get());
        rec.nodes.push_back(move(newNode));
    } else {
        std::ostringstream errmsg;
        errmsg << "shape error in node[" << recLen << "] (" << opName
               << "), tensor shapes are not broadcastable (A.shape=";
        print_arr(A.shape.data(), A.shape.data() + A.nDims(), errmsg);
        errmsg << ", B.shape=";
        print_arr(B.shape.data(), B.shape.data() + B.nDims(), errmsg);
        errmsg << ")";
        throw std::invalid_argument(errmsg.str());
    }
    return rec.nodes.back().get();
}

/**
 * @brief Adds a binary addition node (A + B) to the computation graph.
 *
 * Computes the element-wise sum of two input tensor nodes `A_ptr` and `B_ptr`.
 * Both tensors must have the same shape or be broadcast-compatible.
 *
 * @tparam T The data type of the tensor values.
 *
 * @param rec The computation graph to which the node will be added.
 * @param A_ptr Pointer to the first input tensor node A.
 * @param B_ptr Pointer to the second input tensor node B.
 * @return A pointer to the new node representing the element-wise sum of A and
 * B.
 */
template <typename T>
INode<T> *add(CompGraph<T> &rec, INode<T> *A_ptr, INode<T> *B_ptr) {
    static const BinaryKernels<T, class Kernels::Add<T>> addK;
    return binOperator(rec, A_ptr, B_ptr, addK, "add");
}

/**
 * @brief Adds a binary subtratction node (A - B) to the computation graph.
 *
 * Computes the element-wise difference of two input tensor nodes `A_ptr` and
 * `B_ptr`. Both tensors must have the same shape or be broadcast-compatible.
 *
 * @tparam T The data type of the tensor values.
 *
 * @param rec The computation graph to which the node will be added.
 * @param A_ptr Pointer to the first input tensor node A.
 * @param B_ptr Pointer to the second input tensor node B.
 * @return A pointer to the new node representing the element-wise difference of
 * A and B.
 */
template <typename T>
INode<T> *sub(CompGraph<T> &rec, INode<T> *A_ptr, INode<T> *B_ptr) {
    static const BinaryKernels<T, class Kernels::Sub<T>> subK;
    return binOperator(rec, A_ptr, B_ptr, subK, "sub");
}

/**
 * @brief Adds a binary multiplication node (A * B) to the computation
 * graph.
 *
 * Computes the element-wise product of two input tensor nodes `A_ptr` and
 * `B_ptr`. Both tensors must have the same shape or be broadcast-compatible.
 *
 * @tparam T The data type of the tensor values.
 *
 * @param rec The computation graph to which the node will be added.
 * @param A_ptr Pointer to the first input tensor node A.
 * @param B_ptr Pointer to the second input tensor node B.
 * @return A pointer to the new node representing the element-wise product of A
 * and B.
 */
template <typename T>
INode<T> *mul(CompGraph<T> &rec, INode<T> *A_ptr, INode<T> *B_ptr) {
    static const BinaryKernels<T, class Kernels::Mul<T>> mulK;
    return binOperator(rec, A_ptr, B_ptr, mulK, "mul");
}

/**
 * @brief Adds a binary division node (A / B) to the computation graph.
 *
 * Computes the element-wise quotient of two input tensor nodes `A_ptr` and
 * `B_ptr`. Both tensors must have the same shape or be broadcast-compatible.
 *
 * @tparam T The data type of the tensor values.
 *
 * @param rec The computation graph to which the node will be added.
 * @param A_ptr Pointer to the first input tensor node A.
 * @param B_ptr Pointer to the second input tensor node B.
 * @return A pointer to the new node representing the element-wise quotient of A
 * and B.
 */
template <typename T>
INode<T> *div(CompGraph<T> &rec, INode<T> *A_ptr, INode<T> *B_ptr) {
    static const BinaryKernels<T, class Kernels::Div<T>> divK;
    return binOperator(rec, A_ptr, B_ptr, divK, "div");
}

/**
 * @brief Adds a binary power node (A ^ B) to the computation graph.
 *
 * Computes the element-wise power of two input tensor nodes `A_ptr` and
 * `B_ptr`. Both tensors must have the same shape or be broadcast-compatible.
 *
 * @tparam T The data type of the tensor values.
 *
 * @param rec The computation graph to which the node will be added.
 * @param A_ptr Pointer to the first input tensor node A.
 * @param B_ptr Pointer to the second input tensor node B.
 * @return A pointer to the new node representing the element-wise power of A
 * and B.
 */
template <typename T>
INode<T> *pow(CompGraph<T> &rec, INode<T> *A_ptr, INode<T> *B_ptr) {
    static const BinaryKernels<T, class Kernels::Pow<T>> powK;
    return binOperator(rec, A_ptr, B_ptr, powK, "pow");
}

/**
 * @brief Adds a binary dot product node (A ⋅ B) to the computation graph.
 *
 * Computes the element-wise dot product of two input tensor nodes `A_ptr` and
 * `B_ptr`. Both tensors must have the same shape or be broadcast-compatible.
 *
 * @tparam T The data type of the tensor values.
 *
 * @param rec The computation graph to which the node will be added.
 * @param A_ptr Pointer to the first input tensor node A.
 * @param B_ptr Pointer to the second input tensor node B.
 * @return A pointer to the new node representing the element-wise dot product
 * of A and B.
 */
template <typename T>
INode<T> *dot(CompGraph<T> &rec, INode<T> *A_ptr, INode<T> *B_ptr) {
    using Op = class Kernels::Null::Op;
    using Grad = class Kernels::Null::Grad;
    binaryOp<T, Op> scalar = tensorfuncs::primal::binary::scalarDot<T, Op>;
    binaryGrad<T, Grad> scalar_grad =
        tensorfuncs::adjoint::binary::scalarDot<T, Grad>;
    binaryOp<T, Op> dot = tensorfuncs::primal::binary::dot<T, Op>;
    binaryGrad<T, Grad> dot_grad = tensorfuncs::adjoint::binary::dot<T, Grad>;

    int recLen = rec.nodes.size();
    Tensor<T> &A = A_ptr->value;
    Tensor<T> &B = B_ptr->value;

    bool A_scalar = A.nDims() == 1 && A.shape[0] == 1;
    bool B_scalar = B.nDims() == 1 && B.shape[0] == 1;
    if (B_scalar) {
        auto newNode = std::make_unique<Node_binary<T, Kernels::Null>>(
            scalar, scalar_grad, A_ptr, B_ptr, ((T)0));
        auto raw_ptr = newNode.get();
        raw_ptr->end = A.val.data() + A.val.size();
        rec.nodes.push_back(move(newNode));
    } else if (A_scalar) {
        auto newNode = std::make_unique<Node_binary<T, Kernels::Null>>(
            scalar, scalar_grad, B_ptr, A_ptr, ((T)0));
        auto raw_ptr = newNode.get();
        raw_ptr->end = B.val.data() + B.val.size();
        rec.nodes.push_back(move(newNode));
    } else if (A.nDims() == 1 && B.nDims() == 1 &&
               std::equal(A.shape.begin(), A.shape.end(), B.shape.begin())) {
        auto newNode = std::make_unique<Node_binary<T, Kernels::Null>>(
            dot, dot_grad, A_ptr, B_ptr, ((T)0));
        auto raw_ptr = newNode.get();
        raw_ptr->end = A.val.data() + A.val.size();
        rec.nodes.push_back(move(newNode));

    } else {
        std::ostringstream errmsg;
        errmsg << "shape error in node[" << recLen
               << "] (dot), tensor shapes arent valid for dot product (shape1=";
        print_arr(A.shape.data(), A.shape.data() + A.nDims(), errmsg);
        errmsg << ", shape2=";
        print_arr(B.shape.data(), B.shape.data() + B.nDims(), errmsg);
        errmsg << ")";
        throw std::invalid_argument(errmsg.str());
    }

    return rec.nodes.back().get();
}

/**
 * @brief Adds a matrix multiplication node (A × B) to the computation graph.
 *
 * Performs matrix multiplication between two input tensor nodes `A_ptr` and
 * `B_ptr`. Supports both standard 2D matrix multiplication and batched matrix
 * multiplication:
 * - If both tensors are 2D, performs standard matrix multiplication.
 * - If tensors have more than 2 dimensions, performs batched matrix
 * multiplication over the leading dimensions. For example, multiplying tensors
 * of shape (batch, M, K) × (batch, K, N) yields a result of shape (batch, M,
 * N).
 *
 * @tparam T The data type of the tensor values.
 *
 * @param rec The computation graph to which the node will be added.
 * @param A_ptr Pointer to the left-hand-side input tensor node A.
 * @param B_ptr Pointer to the right-hand-side input tensor node B.
 * @return A pointer to the new node representing the matrix (or batched)
 * product of A and B.
 */
template <typename T>
INode<T> *matmul(CompGraph<T> &rec, INode<T> *A_ptr, INode<T> *B_ptr) {
    int recLen = rec.nodes.size();
    Tensor<T> &A = A_ptr->value;
    Tensor<T> &B = B_ptr->value;

    size_t newLen = std::max(A.nDims(), B.nDims());
    std::vector<int> newShape(newLen);

    const char *opName = newLen == 2 ? "matmul" : "batch_matmul";
    if (!combine_matrix(A.shape.data(), A.nDims(), B.shape.data(), B.nDims(),
                        newShape.data(), newLen)) {
        std::ostringstream errmsg;
        errmsg << "shape error in node[" << recLen << "] (" << opName
               << "), tensor shapes arent valid for " << opName << " (shape1=";
        print_arr(A.shape.data(), A.shape.data() + A.nDims(), errmsg);
        errmsg << ", shape2=";
        print_arr(B.shape.data(), B.shape.data() + B.nDims(), errmsg);
        errmsg << ")";
        throw std::invalid_argument(errmsg.str());
    }

    if (newLen == 2) {
        auto newNode = std::make_unique<Node_matmul<T>>(A_ptr, B_ptr, newShape);
        Strides::matmul<T>(A, B, *newNode.get());
        rec.nodes.push_back(move(newNode));
    } else {
        auto newNode =
            std::make_unique<Node_batch_matmul<T>>(A_ptr, B_ptr, newShape);
        auto raw_ptr = newNode.get();
        if (newLen <= KAAD_MAX_NDIMS) {
            raw_ptr->val_func = Dispatchers::get_batch_matmul<T>()[newLen];
            raw_ptr->grad_func =
                Dispatchers::get_batch_matmul_grad<T>()[newLen];
        }

        Strides::batch_matmul<T>(A, B, *raw_ptr);
        rec.nodes.push_back(move(newNode));
    }

    return rec.nodes.back().get();
}

/**
 * @brief Adds a generalized outer product node to the computation graph.
 *
 * Computes the outer product of two input tensor nodes `A_ptr` and `B_ptr`.
 * The result is a tensor whose shape is the concatenation of the shapes of A
 * and B. For example:
 * - If A has shape (m,) and B has shape (n,), the result has shape (m, n).
 * - If A has shape (m, k) and B has shape (n, p), the result has shape (m, k,
 * n, p).
 *
 * Each element of the output is computed as the product of an element from A
 * and an element from B, preserving the full structure of both input tensors.
 *
 * @tparam T The data type of the tensor values.
 *
 * @param rec The computation graph to which the node will be added.
 * @param A_ptr Pointer to the first input tensor node A.
 * @param B_ptr Pointer to the second input tensor node B.
 * @return A pointer to the new node representing the generalized outer product
 * of A and B.
 */
template <typename T>
INode<T> *outer(CompGraph<T> &rec, INode<T> *A_ptr, INode<T> *B_ptr) {
    int recLen = rec.nodes.size();
    Tensor<T> &A = A_ptr->value;
    Tensor<T> &B = B_ptr->value;

    size_t newLen = A.nDims() + B.nDims();
    std::vector<int> newShape(newLen);
    std::copy(A.shape.begin(), A.shape.end(), newShape.begin());
    std::copy(B.shape.begin(), B.shape.end(), newShape.begin() + A.nDims());

    using Kernel = typename Kernels::Mul<T>;
    using Op = typename Kernel::Op;
    using Grad = typename Kernel::Grad;

    auto newNode = std::make_unique<Node_binary_flex<T, Kernel>>(
        A_ptr, B_ptr, newShape, newLen);
    auto raw = newNode.get();
    Strides::outer<T>(A, B, *raw);
    rec.nodes.push_back(move(newNode));

    return raw;
}

/**
 * @brief Adds a binary minimum node (A __SYMBOL__ B) to the computation graph.
 *
 * Computes the element-wise minimum of two input tensor nodes `A_ptr` and
 * `B_ptr`. Both tensors must have the same shape or be broadcast-compatible.
 *
 * @tparam T The data type of the tensor values.
 *
 * @param rec The computation graph to which the node will be added.
 * @param A_ptr Pointer to the first input tensor node A.
 * @param B_ptr Pointer to the second input tensor node B.
 * @return A pointer to the new node representing the element-wise minimum of A
 * and B.
 */
template <typename T>
INode<T> *min(CompGraph<T> &rec, INode<T> *A_ptr, INode<T> *B_ptr) {
    static const BinaryKernels<T, class Kernels::Min<T>> minK;
    return binOperator(rec, A_ptr, B_ptr, minK, "minimum");
}

/**
 * @brief Adds a binary maximum node (A __SYMBOL__ B) to the computation graph.
 *
 * Computes the element-wise maximum of two input tensor nodes `A_ptr` and
 * `B_ptr`. Both tensors must have the same shape or be broadcast-compatible.
 *
 * @tparam T The data type of the tensor values.
 *
 * @param rec The computation graph to which the node will be added.
 * @param A_ptr Pointer to the first input tensor node A.
 * @param B_ptr Pointer to the second input tensor node B.
 * @return A pointer to the new node representing the element-wise maximum of A
 * and B.
 */
template <typename T>
INode<T> *max(CompGraph<T> &rec, INode<T> *A_ptr, INode<T> *B_ptr) {
    static const BinaryKernels<T, class Kernels::Max<T>> maxK;
    return binOperator(rec, A_ptr, B_ptr, maxK, "minimum");
}
} // namespace kaad
