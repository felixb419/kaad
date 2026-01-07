#pragma once

#include "tensor/tensor.hpp" // for Tensor
#include "tensorfuncs/adjoint_ops.hpp" // for unaryGrad, binaryGrad, batch_matmul, batchma...
#include "tensorfuncs/kernels.hpp" // for Null, Null::Op, Sum
#include "tensorfuncs/primal_ops.hpp" // for unaryOp, binaryOp, batch_matmul, batchmatmulOp
#include <cstddef>                    // for size_t
#include <utility>                    // for std::forward

namespace kaad {

/**
 * @brief Abstract base class representing a node in a computation graph.
 *
 * Each node holds a value tensor and a corresponding gradient tensor.
 * Nodes can be evaluated to compute their output, and gradients can be
 * backpropagated through them. Derived classes must implement the `eval` and
 * `getGrad` methods. Traversal-related members (e.g., strides, offsets) are not
 * initialized in the node constructors but are instead set externally by the
 * corresponding helper functions in the Strides namespace.
 *
 * @tparam T The scalar type (e.g., float or double).
 */
template <typename T> struct INode {

    INode<T> *A =
        nullptr; ///< Pointer to the first input Node (nullptr if leaf node).

    Tensor<T> value;    ///< Value computed by this node.
    Tensor<T> gradient; ///< Gradient associated with this node.

    bool evaluated = false; ///< Whether this node is currently evaluated.
    bool hasInputs = false; ///< Whether this node depends on any input nodes.

    /**
     * @brief Constructs a leaf node with a pre-evaluated tensor value.
     *
     * @param tensor The tensor value to assign to this node.
     */
    INode(Tensor<T> &&tensor)
        : value(std::move(tensor)), gradient(value), evaluated(true),
          hasInputs(false) {
        std::fill(gradient.val.begin(), gradient.val.end(), 0);
    }

    /**
     * @brief Constructs an unevaluated node with a dependency on an input node.
     *
     * @param A_ptr Pointer to the input node.
     * @param args Arguments to construct the value tensor.
     */
    template <typename... Args>
    INode(INode<T> *A_ptr, Args &&...args)
        : A(A_ptr), evaluated(false), value(std::forward<Args>(args)...),
          hasInputs(true), gradient(value) {
        std::fill(value.val.begin(), value.val.end(), 0);
        std::fill(gradient.val.begin(), gradient.val.end(), 0);
    }

    /// Virtual destructor for polymorphic deletion
    virtual ~INode() = default;

    /**
     * @brief Resets the value and gradient of the node.
     *
     * Clears the value tensor and sets the `evaluated` flag to false if the
     * node has inputs. Clears the gradient tensor in all cases.
     */
    inline void reset() {
        if (hasInputs) {
            std::fill(value.val.begin(), value.val.end(), 0);
            evaluated = false;
        }
        std::fill(gradient.val.begin(), gradient.val.end(), 0);
    }

    /**
     * @brief Evaluates the node's value. Must be implemented by derived
     * classes.
     */
    inline virtual void eval() = 0;
    /**
     * @brief Computes the gradient for this node. Must be implemented by
     * derived classes.
     */
    inline virtual void getGrad() = 0;
};

/**
 * @brief A leaf node that holds a fixed tensor value with no computation.
 *
 * This node is considered already evaluated and has no effect during
 * backpropagation except for holding a gradient.
 *
 * @tparam T The scalar type.
 */
template <typename T> struct Node_valued : INode<T> {

    /**
     * @brief Constructs a leaf node holding a constant value.
     *
     * @param tensor The tensor value to store.
     */
    Node_valued(Tensor<T> &&tensor) : INode<T>(std::move(tensor)) {}

    /// No evaluation needed for fixed value nodes.
    void eval() override {}
    /// No gradient computation needed for fixed value nodes.
    void getGrad() override {}
};

/**
 * @brief A unary operation node in a computation graph.
 *
 * Applies a unary operation during forward evaluation and its corresponding
 * gradient function during backpropagation.
 *
 * @tparam T The scalar type.
 * @tparam Kernel A kernel struct providing `Op` and `Grad` types for the
 * operation.
 */
template <typename T, class Kernel> struct Node_unary : INode<T> {
    using Op = class Kernel::Op; ///< Type alias for the operation kernel.
    Op op;

    unaryOp<T, Op> val_func =
        nullptr; ///< Function pointer to the value operation.

    using Grad = class Kernel::Grad; ///< Type alias for the gradient kernel.
    Grad grad;

    unaryGrad<T, Grad> grad_func =
        nullptr; ///< Function pointer to the gradient operation.

    T *end = nullptr; ///< Pointer to the end of the value buffer (used for
                      ///< iteration)

    /**
     * @brief Constructs a unary node with the given operation and gradient.
     *
     * @param operation  Function pointer to the value operation.
     * @param derivative Function pointer to the gradient operation.
     * @param A_ptr    Pointer to the input node.
     * @param args       Arguments to construct the output tensor.
     */
    template <typename... Args>
    Node_unary(unaryOp<T, Op> operation, unaryGrad<T, Grad> derivative,
               INode<T> *A_ptr, Args &&...args)
        : val_func(operation), grad_func(derivative), INode<T>(A_ptr, args...) {
    }

    /**
     * @brief Evaluates the unary operation if not already evaluated.
     *
     * Calls eval on the input node and applies `val_func` to compute this
     * node's value.
     */
    inline void eval() override {
        if (!this->evaluated) {
            this->A->eval();

            val_func(this->A->value.val.data(), this->value.val.data(), end,
                     op);
            this->evaluated = true;
        }
    }

    /**
     * @brief Propagates gradients back through the unary operation.
     *
     * Applies `grad_func` to compute input gradients and recursively calls/
     * `getGrad` on the input node if it has further dependencies.
     */
    inline void getGrad() override {
        grad_func(this->A->value.val.data(), this->A->gradient.val.data(),
                  this->value.val.data(), this->gradient.val.data(), end, grad);

        if (this->A->hasInputs) {
            this->A->getGrad();
        }
    }
};

/**
 * @brief A binary operation node in a computation graph.
 *
 * Applies a binary operation to two tensors with matching shapes during forward
 * evaluation and its corresponding gradient function during backpropagation.
 *
 * @tparam T The scalar type.
 * @tparam Kernel A kernel struct providing `Op` and `Grad` types for the
 * operation.
 */
template <typename T, class Kernel> struct Node_binary : INode<T> {
    INode<T> *B = nullptr; ///< Pointer to the second input Node.

    using Op = class Kernel::Op; ///< Type alias for the operation kernel.
    Op op;

    binaryOp<T, Op> val_func =
        nullptr; ///< Function pointer to the value operation.

    using Grad = class Kernel::Grad; ///< Type alias for the gradient Kernel.
    Grad grad;

    binaryGrad<T, Grad> grad_func =
        nullptr; ///< Function pointer to the gradient operation.

    T *end = nullptr; ///< Pointer to the end of the value buffer (used for
                      ///< iteration).

    /**
     * @brief Constructs a binary operation node with the given operation and
     * gradient.
     *
     * @param operation Function pointer to the value operation.
     * @param derivative Function pointer to the gradient operation.
     * @param A_ptr Pointer to the first input node.
     * @param B_ptr Pointer to the second input node.
     * @param args Arguments to construct the output tensor.
     */
    template <typename... Args>
    Node_binary(binaryOp<T, Op> operation, binaryGrad<T, Grad> derivative,
                INode<T> *A_ptr, INode<T> *B_ptr, Args &&...args)
        : B(B_ptr), val_func(operation), grad_func(derivative),
          INode<T>(A_ptr, args...) {}

    /**
     * @brief Evaluates the binary operation if not already evaluated.
     *
     * Calls eval on the input nodes and applies `val_func` to compute this
     * node's value.
     */
    inline void eval() override {
        if (!this->evaluated) {
            this->A->eval();
            this->B->eval();

            val_func(this->A->value.val.data(), B->value.val.data(),
                     this->value.val.data(), end, op);
            this->evaluated = true;
        }
    }

    /**
     * @brief Propagates gradients back through the binary operation.
     *
     * Applies `grad_func` to compute input gradients and recursively calls
     * `getGrad` on the input nodes if they have further dependencies.
     */
    inline void getGrad() override {
        grad_func(this->A->value.val.data(), this->A->gradient.val.data(),
                  B->value.val.data(), B->gradient.val.data(),
                  this->value.val.data(), this->gradient.val.data(), end, grad);

        if (this->A->hasInputs) {
            this->A->getGrad();
        }
        if (this->B->hasInputs) {
            this->B->getGrad();
        }
    }
};

/**
 * @brief A binary_flex operation node in a computation graph.
 *
 * Applies a binary_flex operation to two tensors with broadcastable shapes
 * during forward evaluation and its corresponding gradient function during
 * backpropagation.
 *
 * @tparam T The scalar type.
 * @tparam Kernel A kernel struct providing `Op` and `Grad` types for the
 * operation.
 */
template <typename T, class Kernel> struct Node_binary_flex : INode<T> {
    INode<T> *B = nullptr; ///< Pointer to the second input Node.

    using Op = class Kernel::Op; ///< Type alias for the operation kernel.
    Op op;

    flexBinaryOp<T, Op> val_func =
        tensorfuncs::primal::binary::flexible<T, Op>; ///< Function pointer to
                                                      ///< the value operation.

    using Grad = class Kernel::Grad; ///< Type alias for the gradient kernel.
    Grad grad;

    flexBinaryGrad<T, Grad> grad_func =
        tensorfuncs::adjoint::binary::flexible<T, Grad>; ///< Function pointer
                                                         ///< to the gradient
                                                         ///< operation.

    int *strideA = nullptr;     ///< stride Array for A.
    int *strideB = nullptr;     ///< stride Array for B.
    int *strideC = nullptr;     ///< stride Array for C.
    size_t *C_offset = nullptr; ///< Per-dim offset to the end of C buffer.
    size_t D = 0;               ///< Number of the dimensions of the C tensor.

    /**
     * @brief Constructs a binary_flex operation node with binary_flex operation
     * and gradient.
     *
     * @param A_ptr Pointer to the first input node.
     * @param B_ptr Pointer to the second input node.
     * @param args Arguments to construct the output tensor.
     */
    template <typename... Args>
    Node_binary_flex(INode<T> *A_ptr, INode<T> *B_ptr, Args &&...args)
        : B(B_ptr), INode<T>(A_ptr, args...) {}

    /**
     * @brief Destructor for Node_binary_flex.
     *
     * Frees dynamically allocated memory for stride and offset arrays.
     */
    ~Node_binary_flex() {
        delete[] strideA;
        delete[] strideB;
        delete[] strideC;
        delete[] C_offset;
    }

    /**
     * @brief Evaluates the binary_flex operation if not already evaluated.
     *
     * Calls eval on the input nodes and applies `val_func` to compute this
     * node's value.
     */
    inline void eval() override {
        if (!this->evaluated) {
            this->A->eval();
            this->B->eval();

            val_func(this->A->value.val.data(), this->B->value.val.data(),
                     this->value.val.data(), strideA, strideB, strideC,
                     C_offset, D, op);
            this->evaluated = true;
        }
    }

    /**
     * @brief Propagates gradients back through the binary_flex operation.
     *
     * Applies `grad_func` to compute input gradients and recursively calls
     * `getGrad` on the input nodes if they have further dependencies.
     */
    inline void getGrad() override {
        grad_func(this->A->value.val.data(), this->A->gradient.val.data(),
                  this->B->value.val.data(), this->B->gradient.val.data(),
                  this->value.val.data(), this->gradient.val.data(), strideA,
                  strideB, strideC, C_offset, D, grad);

        if (this->A->hasInputs) {
            this->A->getGrad();
        }
        if (this->B->hasInputs) {
            this->B->getGrad();
        }
    }
};

/**
 * @brief A matmul node in a computation graph.
 *
 * Applies a matmul operation to two tensors with compatible shapes during
 * forward evaluation and its corresponding gradient function during
 * backpropagation.
 *
 * @tparam T The scalar type.
 */
template <typename T> struct Node_matmul : INode<T> {
    INode<T> *B = nullptr; ///< Pointer to the second input Node.

    matmulOp<T> val_func =
        tensorfuncs::primal::binary::matmul; ///< Function pointer to the matmul
                                             ///< operation.
    matmulGrad<T> grad_func =
        tensorfuncs::adjoint::binary::matmul; ///< Function pointer to the
                                              ///< matmul gradient.

    /**
     * @brief Stride arrays for tensors A, B, and C for all computation stages.
     *
     * Each stride array is a flattened array of size 6, containing strides for
     * all 3 passes (forward and both backward gradients).
     *
     * Index layout for each stride array:
     * - [0..1] Forward pass (C = A * B)
     * - [2..3] Gradient w.r.t. A (dA = dC * Bᵗ)
     * - [4..5] Gradient w.r.t. B (dB = Aᵗ * dC)
     *
     * Each pair of entries represents strides for rows and columns, or outer
     * and inner dimensions, depending on layout.
     */
    int a_dim[3]; ///< Number of rows of tensor A for each computation stage.
    int b_dim[3]; ///< Number of columns of tensor B for each computation stage.
    int k[3];     ///< Shared inner dimension for each computation stage.
    int strideA[6]; ///< Flattened stride pairs for tensor A (2 per stage × 3
                    ///< stages).
    int strideB[6]; ///< Flattened stride pairs for tensor B (2 per stage × 3
                    ///< stages).
    int strideC[6]; ///< Flattened stride pairs for tensor C (2 per stage × 3
                    ///< stages).

    /**
     * @brief Constructs a matmul node.
     *
     * @param A_ptr Pointer to the first input node.
     * @param B_ptr Pointer to the second input node.
     * @param args Arguments to construct the output tensor.
     */
    template <typename... Args>
    Node_matmul(INode<T> *A_ptr, INode<T> *B_ptr, Args &&...args)
        : B(B_ptr), INode<T>(A_ptr, args...) {}

    /**
     * @brief Evaluates the matmul operation if not already evaluated.
     *
     * Calls eval on the input nodes and applies `val_func` to compute this
     * node's value.
     */
    inline void eval() override {
        if (!this->evaluated) {
            this->A->eval();
            this->B->eval();

            val_func(this->A->value.val, this->B->value.val, this->value.val,
                     a_dim[0], b_dim[0], k[0], strideA, strideB, strideC);
            this->evaluated = true;
        }
    }

    /**
     * @brief Propagates gradients back through the matmul operation.
     *
     * Applies `grad_func` to compute input gradients and recursively calls
     * `getGrad` on the input nodes if they have further dependencies.
     */
    inline void getGrad() override {
        grad_func(this->A->value.val, this->A->gradient.val, this->B->value.val,
                  this->B->gradient.val, this->value.val, this->gradient.val,
                  a_dim + 1, b_dim + 1, k + 1, strideA + 2, strideB + 2,
                  strideC + 2);

        if (this->A->hasInputs) {
            this->A->getGrad();
        }
        if (this->B->hasInputs) {
            this->B->getGrad();
        }
    }
};

/**
 * @brief A batch_matmul operation node in a computation graph.
 *
 * Applies a batch_matmul operation to two tensors with matching shapes during
 * forward evaluation and its corresponding gradient function during
 * backpropagation.
 *
 * @tparam T The scalar type.
 * @tparam Kernel A kernel struct providing `Op` and `Grad` types for the
 * operation.
 */
template <typename T> struct Node_batch_matmul : INode<T> {
    INode<T> *B = nullptr; ///< Pointer to the second input Node.

    batchmatmulOp<T> val_func =
        tensorfuncs::primal::binary::batch_matmul; ///< Function pointer to the
                                                   ///< batch_matmul operation.
    batchmatmulGrad<T> grad_func =
        tensorfuncs::adjoint::binary::batch_matmul; ///< Function pointer to the
                                                    ///< batch_matmul gradient.

    /**
     * @brief Stride arrays for A, B, and C for each stage of computation.
     *
     * Index convention:
     * - [0] Forward pass (C = A * B)
     * - [1] Gradient w.r.t. A (dA = dC * Bᵗ)
     * - [2] Gradient w.r.t. B (dB = Aᵗ * dC)
     */
    int *strideA[3]; ///< Stride array for tensor A.
    int *strideB[3]; ///< Stride array for tensor B.
    int *strideC[3]; ///< Stride array for tensor C.
    int *c_shape[3]; ///< shape of C (used for iteration).
    int A_offset[3]; ///< Gap between columns of the A matrix.
    int B_offset[3]; ///< Gap between rows of the B matrix.
    int k[3];        ///< shared inner dimension of the tensors.
    size_t D = 0;    ///< Number of the dimensions of the C tensor.

    /**
     * @brief Constructs a batch_matmul node.
     *
     * @param A_ptr Pointer to the first input node.
     * @param B_ptr Pointer to the second input node.
     * @param args Arguments to construct the output tensor.
     */
    template <typename... Args>
    Node_batch_matmul(INode<T> *A_ptr, INode<T> *B_ptr, Args &&...args)
        : B(B_ptr), INode<T>(A_ptr, args...) {}

    /**
     * @brief Destructor for Node_batch_matmul.
     *
     * Frees dynamically allocated memory for stride and shape arrays.
     */
    ~Node_batch_matmul() {
        for (int i = 0; i < 3; i++) {
            delete[] strideA[i];
            delete[] strideB[i];
            delete[] strideC[i];
            delete[] c_shape[i];
        }
    }

    /**
     * @brief Evaluates the batch_matmul operation if not already
     * evaluated.
     *
     * Calls eval on the input nodes and applies `val_func` to compute this
     * node's value.
     */
    inline void eval() override {
        if (!this->evaluated) {
            this->A->eval();
            this->B->eval();

            val_func(this->A->value.val, this->B->value.val, this->value.val,
                     strideA[0], strideB[0], strideC[0], c_shape[0],
                     A_offset[0], B_offset[0], k[0], D);
            this->evaluated = true;
        }
    }

    /**
     * @brief Propagates gradients back through batch batch_matmul operation.
     *
     * Applies `grad_func` to compute input gradients and recursively calls
     * `getGrad` on the input nodes if they have further dependencies.
     */
    inline void getGrad() override {
        grad_func(this->A->value.val, this->A->gradient.val, this->B->value.val,
                  this->B->gradient.val, this->value.val, this->gradient.val,
                  strideA + 1, strideB + 1, strideC + 1, c_shape + 1,
                  A_offset + 1, B_offset + 1, k + 1, D);

        if (this->A->hasInputs) {
            this->A->getGrad();
        }
        if (this->B->hasInputs) {
            this->B->getGrad();
        }
    }
};

/**
 * @brief A transpose operation node in a computation graph.
 *
 * Applies a transpose operation during forward evaluation and its corresponding
 * gradient function during backpropagation.
 *
 * @tparam T The scalar type.
 */
template <typename T> struct Node_transp : INode<T> {
    using Op = typename Kernels::Null::Op;
    Op op;
    unaryOp<T, Op> val_func =
        tensorfuncs::primal::unary::transpose<T>; ///< Function pointer to the
                                                  ///< value operation.

    using Grad = typename Kernels::Sum<T>::Grad;
    Grad grad;
    unaryGrad<T, Grad> grad_func =
        tensorfuncs::adjoint::unary::pointwise<T, Grad>; ///< Function pointer
                                                         ///< to the gradient
                                                         ///< operation.

    T *A_end = nullptr; ///< Pointer to the end of the A buffer.
    T *C_end = nullptr; ///< Pointer to the end of the C buffer.

    /**
     * @brief Constructs a transpose node with the given operation and gradient.
     *
     * @param A_ptr    Pointer to the input node.
     * @param args       Arguments to construct the output tensor.
     */
    template <typename... Args>
    Node_transp(INode<T> *A_ptr, Args &&...args) : INode<T>(A_ptr, args...) {}

    /**
     * @brief Evaluates the transpose operation if not already evaluated.
     *
     * Calls eval on the input node and applies `val_func` to compute this
     * node's value.
     */
    inline void eval() override {
        if (!this->evaluated) {
            this->A->eval();

            val_func(this->A->value.val.data(), this->value.val.data(), A_end,
                     op);
            this->evaluated = true;
        }
    }

    /**
     * @brief Propagates gradients back through the transpose operation.
     *
     * Applies `grad_func` to compute input gradients and recursively calls/
     * `getGrad` on the input node if it has further dependencies.
     */
    inline void getGrad() override {
        grad_func(this->A->value.val.data(), this->A->gradient.val.data(),
                  this->value.val.data(), this->gradient.val.data(), C_end,
                  grad);

        if (this->A->hasInputs) {
            this->A->getGrad();
        }
    }
};

/**
 * @brief A sum_dim operation node in a computation graph.
 *
 * Applies a sum_dim operation during forward evaluation and its
 * corresponding gradient function during backpropagation.
 *
 * @tparam T The scalar type.
 */
template <typename T> struct Node_sum_dim : INode<T> {
    sumDimOp<T> val_func =
        tensorfuncs::primal::unary::sum_dim; ///< Function pointer to the
                                             ///< sum_dim operation.
    sumDimGrad<T> grad_func =
        tensorfuncs::adjoint::unary::sum_dim; ///< Function pointer to the
                                              ///< sum_dim gradient.

    int *strideA = nullptr;     ///< stride Array for A.
    int *strideC = nullptr;     ///< stride Array for C.
    size_t *A_offset = nullptr; ///< Per-dim offset to the end of A buffer.
    size_t D = 0;               ///< Number of dimensions of A

    /**
     * @brief Constructs a sum_dim node.
     *
     * @param A_ptr    Pointer to the input node.
     * @param args       Arguments to construct the output tensor.
     */
    template <typename... Args>
    Node_sum_dim(INode<T> *A_ptr, Args &&...args) : INode<T>(A_ptr, args...) {}

    /**
     * @brief Destructor for Node_sum_dim.
     *
     * Frees dynamically allocated memory for stride and offset arrays.
     */
    ~Node_sum_dim() {
        delete[] strideA;
        delete[] strideC;
        delete[] A_offset;
    }

    /**
     * @brief Evaluates the sum_dim operation if not already evaluated.
     *
     * Calls eval on the input node and applies `val_func` to compute this
     * node's value.
     */
    inline void eval() override {
        if (!this->evaluated) {
            this->A->eval();

            val_func(this->A->value.val.data(), this->value.val.data(), strideA,
                     strideC, A_offset, D);
            this->evaluated = true;
        }
    }

    /**
     * @brief Propagates gradients back through the sum_dim operation.
     *
     * Applies `grad_func` to compute input gradients and recursively calls/
     * `getGrad` on the input node if it has further dependencies.
     */
    inline void getGrad() override {
        grad_func(this->A->gradient.val.data(), this->gradient.val.data(),
                  strideA, strideC, A_offset, D);

        if (this->A->hasInputs) {
            this->A->getGrad();
        }
    }
};

/**
 * @brief A mean operation node in a computation graph.
 *
 * Applies the mean operation during forward evaluation and computes its
 * corresponding gradient during backpropagation.
 *
 * @tparam T The scalar type.
 */
template <typename T> struct Node_mean : INode<T> {
    meanOp<T> val_func = tensorfuncs::primal::unary::mean;
    meanGrad<T> grad_func = tensorfuncs::adjoint::unary::mean;

    T *A_end =
        nullptr; ///< Pointer to the end of the A buffer (used for iteration).
    T *dA_end =
        nullptr; ///< Pointer to the end of the dA buffer (used for iteration).
    T divisor = 0; ///< Divisor to compute the mean of the A tensor (length of A
                   ///< buffer).

    /**
     * @brief Constructs a mean node.
     *
     * @param A_ptr Pointer to the input node.
     * @param args Arguments to construct the output tensor.
     */
    template <typename... Args>
    Node_mean(INode<T> *A_ptr, Args &&...args) : INode<T>(A_ptr, args...) {}

    /**
     * @brief Evaluates the mean operation if not already evaluated.
     *
     * Calls eval on the input node and applies `val_func` to compute this
     * node's value.
     */
    inline void eval() override {
        if (!this->evaluated) {
            this->A->eval();

            val_func(this->A->value.val.data(), this->value.val.data(), A_end,
                     divisor);
            this->evaluated = true;
        }
    }

    /**
     * @brief Propagates gradients back through the mean operation.
     *
     * Applies `grad_func` to compute input gradients and recursively calls/
     * `getGrad` on the input node if it has further dependencies.
     */
    inline void getGrad() override {
        grad_func(this->A->gradient.val.data(), this->gradient.val.data(),
                  dA_end, divisor);

        if (this->A->hasInputs) {
            this->A->getGrad();
        }
    }
};

/**
 * @brief A mean_dim operation node in a computation graph.
 *
 * Applies a mean_dim operation during forward evaluation and its corresponding
 * gradient function during backpropagation.
 *
 * @tparam T The scalar type.
 */
template <typename T> struct Node_mean_dim : INode<T> {
    meanDimOp<T> val_func = tensorfuncs::primal::unary::mean_dim;
    meanDimGrad<T> grad_func = tensorfuncs::adjoint::unary::mean_dim;

    int *strideA = nullptr; ///< stride Array for A.
    int *strideC = nullptr; ///< stride Array for C.
    size_t *A_offset =
        nullptr; ///< Total number of elements and per-dim offsets of A.
    T *C_end =
        nullptr; ///< Pointer to the end of the C buffer (used for iteration).
    T *dA_end =
        nullptr;  ///< Pointer to the end of the dA buffer (used for iteration).
    size_t D = 0; ///< Number of the dimensions of the A tensor.
    T divisor = 0; ///< Divisor to compute the mean of the A tensor (length of A
                   ///< buffer).

    /**
     * @brief Constructs a mean_dim node with the given operation and gradient.
     *
     * @param A_ptr    Pointer to the input node.
     * @param args       Arguments to construct the output tensor.
     */
    template <typename... Args>
    Node_mean_dim(INode<T> *A_ptr, Args &&...args) : INode<T>(A_ptr, args...) {}

    /**
     * @brief Destructor for Node_mean_dim.
     *
     * Frees dynamically allocated memory for stride and offset arrays.
     */
    ~Node_mean_dim() {
        delete[] strideA;
        delete[] strideC;
        delete[] A_offset;
    }

    /**
     * @brief Evaluates the mean_dim operation if not already evaluated.
     *
     * Calls eval on the input node and applies `val_func` to compute this
     * node's value.
     */
    inline void eval() override {
        if (!this->evaluated) {
            this->A->eval();

            val_func(this->A->value.val.data(), this->value.val.data(), strideA,
                     strideC, A_offset, D, divisor, C_end);
            this->evaluated = true;
        }
    }

    /**
     * @brief Propagates gradients back through the mean_dim operation.
     *
     * Applies `grad_func` to compute input gradients and recursively calls/
     * `getGrad` on the input node if it has further dependencies.
     */
    inline void getGrad() override {
        grad_func(this->A->value.val.data(), this->A->gradient.val.data(),
                  this->value.val.data(), this->gradient.val.data(), strideA,
                  strideC, A_offset, D, divisor, dA_end);

        if (this->A->hasInputs) {
            this->A->getGrad();
        }
    }
};

/**
 * @brief A slice operation node in a computation graph.
 *
 * Applies a slice operation during forward evaluation and its corresponding
 * gradient function during backpropagation.
 *
 * @tparam T The scalar type.
 */
template <typename T> struct Node_slice : INode<T> {
    sliceOp<T> val_func = tensorfuncs::primal::unary::slice<T>;
    sliceGrad<T> grad_func = tensorfuncs::adjoint::unary::slice<T>;

    int *strideA = nullptr;           ///< Stride array for tensor A.
    int *strideB = nullptr;           ///< Stride array for tensor B.
    int *strideC = nullptr;           ///< Stride array for tensor C.
    size_t *start_offset_a = nullptr; ///< Offset for the start of A.
    size_t *C_offset = nullptr; ///< Per-dim offset to the end of C buffer.
    size_t D = 0;               ///< Number of the dimensions of the C tensor.

    /**
     * @brief Constructs a slice node.
     *
     * @param A_ptr    Pointer to the input node.
     * @param args       Arguments to construct the output tensor.
     */
    template <typename... Args>
    Node_slice(INode<T> *A_ptr, Args &&...args) : INode<T>(A_ptr, args...) {}

    /**
     * @brief Destructor for Node_slice.
     *
     * Frees dynamically allocated memory for stride and offset arrays.
     */
    ~Node_slice() {
        delete[] strideA;
        delete[] strideB;
        delete[] strideC;
        delete[] C_offset;
    }

    /**
     * @brief Evaluates the slice operation if not already evaluated.
     *
     * Calls eval on the input node and applies `val_func` to compute this
     * node's value.
     */
    inline void eval() override {
        if (!this->evaluated) {
            this->A->eval();

            val_func(this->A->value.val.data(), this->value.val.data(), strideA,
                     strideC, start_offset_a, C_offset, D);
            this->evaluated = true;
        }
    }

    /**
     * @brief Propagates gradients back through the slice operation.
     *
     * Applies `grad_func` to compute input gradients and recursively calls/
     * `getGrad` on the input node if it has further dependencies.
     */
    inline void getGrad() override {
        grad_func(this->A->gradient.val.data(), this->gradient.val.data(),
                  strideA, strideC, start_offset_a, C_offset, D);

        if (this->A->hasInputs) {
            this->A->getGrad();
        }
    }
};

} // namespace kaad
