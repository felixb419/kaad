#include <stddef.h>      // for size_t
#include <algorithm>     // for equal, max
#include <memory>        // for make_unique
#include <ostream>       // for operator<<, basic_ostream, char_traits, ostr...
#include <stdexcept>     // for invalid_argument
#include "gradients.h"   // for Gradients, binaryGrad, flexBinaryGrad
#include "kernels.h"     // for Kernels, NullOp
#include "operations.h"  // for Operations, binaryOp, flexBinaryOp
#include "strides.h"     // for Strides
#include "tensor.h"      // for Tensor (ptr only), print_arr, combine_flexible
template <typename T, class Kernel> struct Node_binary;
template <typename T, class Kernel> struct Node_binary_flex;
template <typename T> struct CompGraph;
template <typename T> struct INode;
template <typename T> struct Node_batch_matmul;
template <typename T> struct Node_matmul;

#pragma once

template <typename T, class Kernel>
struct BinaryKernels {
    using Op = class Kernel::Op;
    using Grad = class Kernel::Grad;

    binaryOp<T,Op> scalarOpRhs = Operations<T,Op>::scalarRhs;
    binaryOp<T,Op> scalarOpLhs = Operations<T,Op>::scalarLhs;
    binaryOp<T,Op> pointOp = Operations<T,Op>::pointwise;
    flexBinaryOp<T,Op> flexOp = Operations<T,Op>::flexible;
    
    binaryGrad<T,Grad> scalarGradRhs = Gradients<T,Grad>::scalarRhs;
    binaryGrad<T,Grad> scalarGradLhs = Gradients<T,Grad>::scalarLhs;
    binaryGrad<T,Grad> pointGrad = Gradients<T,Grad>::pointwise;
    flexBinaryGrad<T,Grad> flexGrad = Gradients<T,Grad>::flexible;
};

template <typename T, class Kernel>
INode<T>* binOperator(CompGraph<T>& rec, INode<T>* A_ptr, INode<T>* B_ptr, const BinaryKernels<T,Kernel> kernels, const char* opName) {
    int recLen = rec.nodes.size();
    Tensor<T>& A = A_ptr->value;
    Tensor<T>& B = B_ptr->value;
    bool A_scalar = A.nDims == 1 && A.shape[0] == 1;
    bool B_scalar = B.nDims == 1 && B.shape[0] == 1;

    size_t newLen = max(A.nDims, B.nDims);
    int* newShape = new int[newLen];

    if (B_scalar) {
        copy(A.shape, A.shape + A.nDims, newShape);

        auto newNode = std::make_unique<Node_binary<T,Kernel>>(kernels.scalarOpRhs, kernels.scalarGradRhs, A_ptr, B_ptr, newShape, A.nDims);
        newNode->len = newNode->value.len;
        rec.nodes.push_back(move(newNode));
    }
    else if (A_scalar) {
        copy(B.shape, B.shape + B.nDims, newShape);

        auto newNode = std::make_unique<Node_binary<T,Kernel>>(kernels.scalarOpLhs, kernels.scalarGradLhs, A_ptr, B_ptr, newShape, B.nDims);
        newNode->len = newNode->value.len;
        rec.nodes.push_back(move(newNode));
    }
    else if (A.nDims == B.nDims && equal(A.shape, A.shape + A.nDims, B.shape)) {
        copy(A.shape, A.shape + A.nDims, newShape);

        auto newNode = std::make_unique<Node_binary<T,Kernel>>(kernels.pointOp, kernels.pointGrad, A_ptr, B_ptr, newShape, A.nDims);
        newNode->len = newNode->value.len;
        rec.nodes.push_back(move(newNode));
    }
    else if (combine_flexible(A.shape, A.nDims, B.shape, B.nDims, newShape, newLen)) {
        auto newNode = std::make_unique<Node_binary_flex<T,Kernel>>(kernels.flexOp, kernels.flexGrad, A_ptr, B_ptr, newShape, newLen);
        Strides<T>::flexible_binary(A, B, *newNode.get());
        rec.nodes.push_back(move(newNode));
    }
    else {
        ostringstream errmsg;
        errmsg << "shape error in node[" << recLen << "] (" << opName << "), tensor shapes are not broadcastable (A.shape=";
        print_arr(A.shape, A.shape + A.nDims, errmsg);
        errmsg << ", B.shape=";
        print_arr(B.shape, B.shape + B.nDims, errmsg);
        errmsg << ")";
        throw invalid_argument(errmsg.str());
    }
    return rec.nodes[recLen].get();
}

// add A and B
// where A and B are Tensors with Broadcastable shapes
template <typename T>
INode<T>* add(CompGraph<T>& rec, INode<T>* A_ptr, INode<T>* B_ptr) {
    static const BinaryKernels<T, class Kernels<T>::Add> addK;
    return binOperator(rec, A_ptr, B_ptr, addK, "add");
}

// subtract B from A
// where A and B are Tensors with Broadcastable shapes
template <typename T>
INode<T>* sub(CompGraph<T>& rec, INode<T>* A_ptr, INode<T>* B_ptr) {
    static const BinaryKernels<T, class Kernels<T>::Sub> subK;
    return binOperator(rec, A_ptr, B_ptr, subK, "sub");
}

// multiply A and B
// where A and B are Tensors with Broadcastable shapes
template <typename T>
INode<T>* mul(CompGraph<T>& rec, INode<T>* A_ptr, INode<T>* B_ptr) {
    static const BinaryKernels<T, class Kernels<T>::Mul> mulK;
    return binOperator(rec, A_ptr, B_ptr, mulK, "mul");
}

// divide A by B
// where A and B are Tensors with Broadcastable shapes
template <typename T>
INode<T>* div(CompGraph<T>& rec, INode<T>* A_ptr, INode<T>* B_ptr) {
    static const BinaryKernels<T, class Kernels<T>::Div> divK;
    return binOperator(rec, A_ptr, B_ptr, divK, "div");
}

// raise A to the power of B
// where A and B are Tensors with Broadcastable shapes
template <typename T>
INode<T>* pow(CompGraph<T>& rec, INode<T>* A_ptr, INode<T>* B_ptr) {
    static const BinaryKernels<T, class Kernels<T>::Pow> powK;
    return binOperator(rec, A_ptr, B_ptr, powK, "pow");
}

// compute dot prodcut of A and B
// where A and B are scalars or vectors with the same length
template <typename T>
INode<T>* dot(CompGraph<T>& rec, INode<T>* A_ptr, INode<T>* B_ptr) {
    using Op = class NullOp::Op;
    using Grad = class NullOp::Grad;
    binaryOp<T,Op> scalar = Operations<T,Op>::scalarDot;
    binaryGrad<T,Grad> scalar_grad = Gradients<T,Grad>::scalarDot_grad;
    binaryOp<T,Op> dot = Operations<T,Op>::dot;
    binaryGrad<T,Grad> dot_grad = Gradients<T,Grad>::dot_grad;

    int recLen = rec.nodes.size();
    Tensor<T>& A = A_ptr->value;
    Tensor<T>& B = B_ptr->value;

    bool A_scalar = A.nDims == 1 && A.shape[0] == 1;
    bool B_scalar = B.nDims == 1 && B.shape[0] == 1;
    if (B_scalar) {
        auto newNode = make_unique<Node_binary<T,NullOp>>(scalar, scalar_grad, A_ptr, B_ptr, ((T)0));

        newNode->len = A.len;
        rec.nodes.push_back(move(newNode));
    }
    else if (A_scalar) {
        auto newNode = make_unique<Node_binary<T,NullOp>>(scalar, scalar_grad, B_ptr, A_ptr, ((T)0));
        newNode->len = B.len;
        rec.nodes.push_back(move(newNode));
    }
    else if (A.nDims == B.nDims && equal(A.shape, A.shape + A.nDims, B.shape)) {
        auto newNode = make_unique<Node_binary<T,NullOp>>(dot, dot_grad, A_ptr, B_ptr, ((T)0));
        newNode->len = A.len;
        rec.nodes.push_back(move(newNode));

    }
    else {
        ostringstream errmsg;
        errmsg << "shape error in node[" << recLen << "] (dot), tensor shapes arent valid for dot product (shape1=";
        print_arr(A.shape, A.shape + A.nDims, errmsg);
        errmsg << ", shape2=";
        print_arr(B.shape, B.shape + B.nDims, errmsg);
        errmsg << ")";
        throw invalid_argument(errmsg.str());
    }

    return rec.nodes.back().get();
}

// matrix multiply A and B
// where A and B are Tensors with valid dimensions
template <typename T>
INode<T>* matmul(CompGraph<T>& rec, INode<T>* A_ptr, INode<T>* B_ptr) {
    int recLen = rec.nodes.size();
    Tensor<T>& A = A_ptr->value;
    Tensor<T>& B = B_ptr->value;

    size_t newLen = max(A.nDims, B.nDims);
    int* newShape = new int[newLen];

    const char* opName = newLen == 2 ? "matmul" : "batch_matmul";
    if (!combine_matrix(A.shape, A.nDims, B.shape, B.nDims, newShape, newLen)) {
        ostringstream errmsg;
        errmsg << "shape error in node[" << recLen << "] (" << opName << "), tensor shapes arent valid for " << opName << " (shape1=";
        print_arr(A.shape, A.shape + A.nDims, errmsg);
        errmsg << ", shape2=";
        print_arr(B.shape, B.shape + B.nDims, errmsg);
        errmsg << ")";
        throw invalid_argument(errmsg.str());
    }

    if (newLen == 2) {
        auto newNode = make_unique<Node_matmul<T>>(Operations<T,nullptr_t>::matmul, Gradients<T,nullptr_t>::matmul_grad, A_ptr, B_ptr, newShape, newLen);
        Strides<T>::matmul(A, B, *newNode.get());
        rec.nodes.push_back(move(newNode));
    }
    else {
        auto newNode = make_unique<Node_batch_matmul<T>>(Operations<T,nullptr_t>::batch_matmul, Gradients<T,nullptr_t>::batch_matmul_grad, A_ptr, B_ptr, newShape, newLen);
        Strides<T>::batch_matmul(A, B, *newNode.get());
        rec.nodes.push_back(move(newNode));
    }
    
    return rec.nodes.back().get();
}

// compute outer product of A and B
// where A and B are Tensors
template <typename T>
INode<T>* outer(CompGraph<T>& rec, INode<T>* A_ptr, INode<T>* B_ptr) {
    int recLen = rec.nodes.size();
    Tensor<T>& A = A_ptr->value;
    Tensor<T>& B = B_ptr->value;

    size_t newLen = A.nDims + B.nDims;
    int* newShape = new int[newLen];
    copy(A.shape, A.shape + A.nDims, newShape);
    copy(B.shape, B.shape + B.nDims, newShape + A.nDims);

    using Kernel = typename Kernels<T>::Mul;
    using Op = typename Kernel::Op;
    using Grad = typename Kernel::Grad;

    auto newNode = make_unique<Node_binary_flex<T,Kernel>>(Operations<T,Op>::flexible, Gradients<T,Grad>::flexible, A_ptr, B_ptr, newShape, newLen);
    auto raw = newNode.get();
    Strides<T>::outer(A, B, *raw);
    rec.nodes.push_back(move(newNode));

    return raw;
}

// compute pointwise minimum of A and B
// where A and B are Tensors with broadcastable shapes
template <typename T>
INode<T>* min(CompGraph<T>& rec, INode<T>* A_ptr, INode<T>* B_ptr) {
    static const BinaryKernels<T, class Kernels<T>::Min> minK;
    return binOperator(rec, A_ptr, B_ptr, minK, "minimum");
}

// compute pointwise maximum of A and B
// where A and B are Tensors with broadcastable shapes
template <typename T>
INode<T>* max(CompGraph<T>& rec, INode<T>* A_ptr, INode<T>* B_ptr) {
    static const BinaryKernels<T, class Kernels<T>::Max> maxK;
    return binOperator(rec, A_ptr, B_ptr, maxK, "minimum");
}
