#pragma once

#include <stddef.h>          // for size_t
#include <initializer_list>  // for initializer_list
#include <memory>            // for std::make_unique
#include <sstream>           // for operator<<, basic_ostream::operator<<
#include <stdexcept>         // for std::invalid_argument
#include "kernels.h"
#include "gradients.h"       // for Gradients, unaryGrad, flexUnaryGrad, mea...
#include "kernels.h"         // for Kernels, NullOp
#include "operations.h"      // for Operations, unaryOp, flexUnaryOp, meanDimOp
#include "strides.h"         // for Strides
#include "tensor.h"          // for print_arr, Tensor (ptr only), transp

namespace kaad {
    template <typename T, class Kernel> struct Node_unary;
    template <typename T, class Kernel> struct Node_unary_flex;
    template <typename T> struct CompGraph;
    template <typename T> struct INode;
    template <typename T> struct Node_mean_dim;

    template <typename T, class Kernel>
    struct UnaryKernels {
        using Op = class Kernel::Op;
        using Grad = class Kernel::Grad;
        unaryOp<T,Op> op = Operations::unary_pointwise<T,Op>;
        unaryGrad<T,Grad> grad = Gradients::unary_pointwise<T,Grad>;
    };

    template <typename T, class Kernel>
    INode<T>* unOperator(CompGraph<T>& rec, INode<T>* A_ptr , const UnaryKernels<T,Kernel> kernels) {
        Tensor<T>& A = A_ptr->value;

        int* newShape = new int[A.nDims];
        std::copy(A.shape, A.shape + A.nDims, newShape);

        auto newNode = std::make_unique<Node_unary<T,Kernel>>(kernels.op, kernels.grad, A_ptr, newShape, A.nDims);
        auto raw = newNode.get();
        rec.nodes.push_back(move(newNode));
        raw->len = A.len;

        return raw;
    }

    // negate A
    // where A is a tensor
    template <typename T>
    INode<T>* negative(CompGraph<T>& rec, INode<T>* A_ptr) {
        static const UnaryKernels<T, class Kernels<T>::Neg> negK;
        return unOperator(rec, A_ptr, negK);
    }

    // square A
    // where A is a tensor
    template <typename T>
    INode<T>* square(CompGraph<T>& rec, INode<T>* A_ptr) {
        static const UnaryKernels<T, class Kernels<T>::Square> squareK;
        return unOperator(rec, A_ptr, squareK);
    }

    // compte squareroot of A
    // where A is a tensor
    template <typename T>
    INode<T>* sqrt(CompGraph<T>& rec, INode<T>* A_ptr) {
        static const UnaryKernels<T, class Kernels<T>::Sqrt> sqrtK;
        return unOperator(rec, A_ptr, sqrtK);
    }

    // compute logarithm base e of A
    // where A is a tensor
    template <typename T>
    INode<T>* log(CompGraph<T>& rec, INode<T>* A_ptr) {
        static const UnaryKernels<T, class Kernels<T>::Log> logK;
        return unOperator(rec, A_ptr, logK);
    }

    // raise A to the power of e
    // where A is a tensor
    template <typename T>
    INode<T>* exp(CompGraph<T>& rec, INode<T>* A_ptr) {
        static const UnaryKernels<T, class Kernels<T>::Exp> expK;
        return unOperator(rec, A_ptr, expK);
    }

    // compute the absolute value of A
    // where A is a tensor
    template <typename T>
    INode<T>* abs(CompGraph<T>& rec, INode<T>* A_ptr) {
        static const UnaryKernels<T, class Kernels<T>::Abs> absK;
        return unOperator(rec, A_ptr, absK);
    }

    // transpose A
    // if given A is transposed according to perm
    // where A is a tensor
    template <typename T>
    INode<T>* transpose(CompGraph<T>& rec, INode<T>* A_ptr, std::initializer_list<int> perm={}) {
        int recLen = rec.nodes.size();
        Tensor<T>& A = A_ptr->value;

        if (A.nDims < 2) {
            std::ostringstream errmsg;
            errmsg << "shape error in node[" << recLen << "] (transpose), A.nDims hast to be > 1 (shape1=";
            print_arr(A.shape, A.shape + A.nDims, errmsg);
            errmsg << ")";
            throw std::invalid_argument(errmsg.str());
        }

        int *shape_T, *stride_T;
        if (perm.size() == 0) {
            shape_T = new int[A.nDims];
            stride_T = new int[A.nDims];
            std::copy(A.shape, A.shape + A.nDims, shape_T);
            std::copy(A.stride, A.stride + A.nDims, stride_T);
            
            transp(A.shape, A.stride, A.nDims, shape_T, stride_T);
        }
        else {
            if (perm.size() != A.nDims) {
                std::ostringstream errmsg;
                errmsg << "argument error in node[" << recLen << "] (transpose), perm.size() has to be same as A.nDims (perm=";
                print_arr(perm.begin(), perm.end(), errmsg);
                errmsg << ", shape1=";
                print_arr(A.shape, A.shape + A.nDims, errmsg);
                errmsg << ")";
                throw std::invalid_argument(errmsg.str());
            }

            shape_T = new int[A.nDims];
            stride_T = new int[A.nDims];
            int* count = new int[A.nDims];
            fill(count, count + A.nDims, 0);

            int* sh = shape_T;
            int* st = stride_T;
            for (int idx : perm) {

                count[idx]++;

                *(sh++) = A.shape[idx];
                *(st++) = A.stride[idx];
            } 
            for (int* p = count; p != count + A.nDims; p++) {
                if (*p != 1) {
                    std::ostringstream errmsg;
                    errmsg << "argument error in node[" << recLen << "] (transpose), invalid permutation, perm has to contain index of every dimension exactly once (perm=";
                    print_arr(perm.begin(), perm.end(), errmsg);
                    errmsg << ")";
                    throw std::invalid_argument(errmsg.str());
                }
            }
        }

        using Kernel = class Kernels<T>::Transp;
        using Op = class Kernel::Op;
        using Grad = class Kernel::Grad;
        unaryOp<T,Op> op = Operations::transpose<T,Op>;
        unaryGrad<T,Grad> grad = Gradients::unary_pointwise<T,Grad>;

        auto newNode = std::make_unique<Node_unary<T,Kernel>>(op, grad, A_ptr, shape_T, stride_T, A.nDims);
        newNode->len = A.len;
        rec.nodes.push_back(move(newNode));

        return rec.nodes.back().get();
    }

    template <typename T>
    INode<T>* sum(CompGraph<T>& rec, INode<T>* A_ptr, int dim=-1) {
        int recLen = rec.nodes.size();
        Tensor<T>& A = A_ptr->value;

        if (dim == -1) {
            using Kernel = class Kernels<T>::Sum;
            using Op = typename Kernel::Op;
            using Grad = typename Kernel::Grad;
            unaryOp<T,Op> op = Operations::unary_scalarRhs<T,Op>;
            unaryGrad<T,Grad> grad = Gradients::unary_scalarRhs<T,Grad>;
            auto newNode = std::make_unique<Node_unary<T,Kernel>>(op, grad, A_ptr, (T)0);
            auto debug = *static_cast<Node_unary<T,Kernel>*>(newNode.get());
            newNode->len = A_ptr->value.len;
            rec.nodes.push_back(move(newNode));
        }
        else {
            if (dim < 0 || dim >= A.nDims) {
                std::ostringstream errmsg;
                errmsg << "argument error in node[" << recLen << "] (sum), dim has to be a valid index of A.shape (dim=" << dim << ", A.nDims=" << A.nDims << ")" << std::endl;
                throw std::invalid_argument(errmsg.str());
            }

            size_t newLen = A.nDims - 1;
            int* newShape = new int[newLen];

            std::copy(A.shape, A.shape + dim, newShape);
            std::copy(A.shape + dim + 1, A.shape + A.nDims, newShape + dim);

            using Kernel = class Kernels<T>::Sum;
            using Op = class Kernel::Op;
            using Grad = class Kernel::Grad;
            flexUnaryOp<T,Op> op = Operations::unary_flexible<T,Op>;
            flexUnaryGrad<T,Grad> grad = Gradients::unary_flexible<T,Grad>;

            auto newNode = std::make_unique<Node_unary_flex<T,Kernel>>(op, grad, A_ptr, newShape, newLen);
            Strides<T>::along_dim(A, *newNode.get(), dim);
            rec.nodes.push_back(move(newNode));
        }

        return rec.nodes.back().get();
    }

    template <typename T>
    INode<T>* mean(CompGraph<T>& rec, INode<T>* A_ptr, int dim=-1) {
        int recLen = rec.nodes.size();
        Tensor<T>& A = A_ptr->value;

        if (dim == -1) {
            int* newShape = new int[] { 1 };

            using Op = typename NullOp::Op;
            using Grad = typename NullOp::Grad;
            unaryOp<T,Op> op = Operations::mean<T,Op>;
            unaryGrad<T,Grad> grad = Gradients::mean_grad<T,Grad>;
            auto newNode = std::make_unique<Node_unary<T,NullOp>>(op, grad, A_ptr, newShape, 1);
            newNode->len = A_ptr->value.len;
            rec.nodes.push_back(move(newNode));
        }
        else {
            if (dim < 0 || dim >= A.nDims) {
                std::ostringstream errmsg;
                errmsg << "argument error in node[" << recLen << "] (mean), dim has to be a valid index of A.shape (dim=" << dim << ", A.nDims=" << A.nDims << ")" << std::endl;
                throw std::invalid_argument(errmsg.str());
            }

            size_t newLen = A.nDims - 1;
            int* newShape = new int[newLen];

            std::copy(A.shape, A.shape + dim, newShape);
            std::copy(A.shape + dim + 1, A.shape + A.nDims, newShape + dim);

            using Op = typename NullOp::Op;
            using Grad = typename NullOp::Grad;
            meanDimOp<T> op = Operations::mean_dim<T,Op>;
            meanDimGrad<T> grad = Gradients::mean_dim_grad<T,Grad>;
            auto newNode = std::make_unique<Node_mean_dim<T>>(op, grad, A_ptr, newShape, newLen);
            Strides<T>::mean_along_dim(A, *newNode.get(), dim);
            rec.nodes.push_back(move(newNode));
        }

        return rec.nodes.back().get();
    }
}    
