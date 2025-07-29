#pragma once

#include "gradients.h"  // for batchmatmulGrad, binaryGrad, flexBinaryGrad
#include "operations.h" // for batchmatmulOp, binaryOp, flexBinaryOp, matmulOp
#include "tensor.h"     // for Tensor
#include <stddef.h>     // for size_t
#include <utility>      // for forward

namespace kaad {
template <typename T> struct INode {

    INode<T> *in1 = nullptr;

    Tensor<T> value;
    Tensor<T> gradient;

    bool evaluated = false;
    bool hasInputs = false;

    // construct as evaluated
    INode(Tensor<T> &&tensor)
        : value(std::move(tensor)), gradient(value), evaluated(true),
          hasInputs(false) {
        std::fill(gradient.val, gradient.val + gradient.len, 0);
    }

    // construct as not evaluated
    template <typename... Args>
    INode(INode<T> *in1_ptr, Args &&...args)
        : in1(in1_ptr), evaluated(false), value(std::forward<Args>(args)...),
          hasInputs(true), gradient(value) {
        std::fill(value.val, value.val + value.len, 0);
        std::fill(gradient.val, gradient.val + gradient.len, 0);
    }

    virtual ~INode() = default;

    inline void reset() {
        if (hasInputs) {
            std::fill(value.val, value.val + value.len, 0);
            evaluated = false;
        }
        std::fill(gradient.val, gradient.val + gradient.len, 0);
    }

    inline virtual void eval() = 0;
    inline virtual void getGrad() = 0;
};

template <typename T> struct Node_valued : INode<T> {

    Node_valued(Tensor<T> &&tensor) : INode<T>(std::move(tensor)) {}

    void eval() override {}
    void getGrad() override {}
};

template <typename T, class Kernel> struct Node_unary : INode<T> {
    using Op = class Kernel::Op;
    Op op;
    unaryOp<T, Op> val_func = nullptr;
    using Grad = class Kernel::Grad;
    Grad grad;
    unaryGrad<T, Grad> grad_func = nullptr;

    size_t len = 0;

    template <typename... Args>
    Node_unary(unaryOp<T, Op> operation, unaryGrad<T, Grad> derivative,
               INode<T> *in1_ptr, Args &&...args)
        : val_func(operation), grad_func(derivative),
          INode<T>(in1_ptr, args...) {}

    inline void eval() override {
        if (!this->evaluated) {
            this->in1->eval();

            val_func(this->in1->value.val, this->value.val, len, op);
            this->evaluated = true;
        }
    }

    inline void getGrad() override {
        grad_func(this->in1->value.val, this->in1->gradient.val,
                  this->value.val, this->gradient.val, len, grad);

        if (this->in1->hasInputs) {
            this->in1->getGrad();
        }
    }
};

template <typename T, class Kernel> struct Node_binary : INode<T> {
    INode<T> *in2 = nullptr;

    using Op = class Kernel::Op;
    Op op;
    binaryOp<T, Op> val_func = nullptr;
    using Grad = class Kernel::Grad;
    Grad grad;
    binaryGrad<T, Grad> grad_func = nullptr;

    size_t len = 0;

    template <typename... Args>
    Node_binary(binaryOp<T, Op> operation, binaryGrad<T, Grad> derivative,
                INode<T> *in1_ptr, INode<T> *in2_ptr, Args &&...args)
        : in2(in2_ptr), val_func(operation), grad_func(derivative),
          INode<T>(in1_ptr, args...) {}

    inline void eval() override {
        if (!this->evaluated) {
            this->in1->eval();
            this->in2->eval();

            val_func(this->in1->value.val, in2->value.val, this->value.val, len,
                     op);
            this->evaluated = true;
        }
    }

    inline void getGrad() override {
        grad_func(this->in1->value.val, this->in1->gradient.val, in2->value.val,
                  in2->gradient.val, this->value.val, this->gradient.val, len,
                  grad);

        if (this->in1->hasInputs) {
            this->in1->getGrad();
        }
        if (this->in2->hasInputs) {
            this->in2->getGrad();
        }
    }
};

template <typename T, class Kernel> struct Node_binary_flex : INode<T> {
    INode<T> *in2 = nullptr;

    using Op = class Kernel::Op;
    Op op;
    flexBinaryOp<T, Op> val_func = Operations::flexible<T, Op>;
    using Grad = class Kernel::Grad;
    Grad grad;
    flexBinaryGrad<T, Grad> grad_func = Gradients::flexible<T, Grad>;

    int *strideA = nullptr;
    int *strideB = nullptr;
    int *strideC = nullptr;
    size_t *c_offset = nullptr;
    size_t D;

    template <typename... Args>
    Node_binary_flex(INode<T> *in1_ptr, INode<T> *in2_ptr, Args &&...args)
        : in2(in2_ptr), INode<T>(in1_ptr, args...) {}

    ~Node_binary_flex() {
        delete[] strideA;
        delete[] strideB;
        delete[] strideC;
        delete[] c_offset;
    }

    inline void eval() override {
        if (!this->evaluated) {
            this->in1->eval();
            this->in2->eval();

            val_func(this->in1->value.val, this->in2->value.val,
                     this->value.val, strideA, strideB, strideC, c_offset, D,
                     op);
            this->evaluated = true;
        }
    }

    inline void getGrad() override {
        grad_func(this->in1->value.val, this->in1->gradient.val,
                  this->in2->value.val, this->in2->gradient.val,
                  this->value.val, this->gradient.val, strideA, strideB,
                  strideC, c_offset, D, grad);

        if (this->in1->hasInputs) {
            this->in1->getGrad();
        }
        if (this->in2->hasInputs) {
            this->in2->getGrad();
        }
    }
};

template <typename T> struct Node_matmul : INode<T> {
    INode<T> *in2 = nullptr;

    matmulOp<T> op = Operations::matmul;
    matmulGrad<T> grad_op = Gradients::matmul;

    int a_dim[3];
    int b_dim[3];
    int k[3];
    int strideA[6];
    int strideB[6];
    int strideC[6];

    template <typename... Args>
    Node_matmul(INode<T> *in1_ptr, INode<T> *in2_ptr, Args &&...args)
        : in2(in2_ptr), INode<T>(in1_ptr, args...) {}

    inline void eval() override {
        if (!this->evaluated) {
            this->in1->eval();
            this->in2->eval();

            op(this->in1->value.val, this->in2->value.val, this->value.val,
               a_dim[0], b_dim[0], k[0], strideA, strideB, strideC);
            this->evaluated = true;
        }
    }

    inline void getGrad() override {
        grad_op(this->in1->value.val, this->in1->gradient.val,
                this->in2->value.val, this->in2->gradient.val, this->value.val,
                this->gradient.val, a_dim + 1, b_dim + 1, k + 1, strideA + 2,
                strideB + 2, strideC + 2);

        if (this->in1->hasInputs) {
            this->in1->getGrad();
        }
        if (this->in2->hasInputs) {
            this->in2->getGrad();
        }
    }
};

template <typename T> struct Node_batch_matmul : INode<T> {
    INode<T> *in2 = nullptr;

    batchmatmulOp<T> val_func = Operations::batch_matmul;
    batchmatmulGrad<T> grad_func = Gradients::batch_matmul;

    int *strideA[3];
    int *strideB[3];
    int *strideC[3];
    int *c_shape[3];
    int a_offset[3];
    int b_offset[3];
    int k[3];
    size_t D;

    template <typename... Args>
    Node_batch_matmul(INode<T> *in1_ptr, INode<T> *in2_ptr, Args &&...args)
        : in2(in2_ptr), INode<T>(in1_ptr, args...) {}

    ~Node_batch_matmul() {
        for (int i = 0; i < 3; i++) {
            delete[] strideA[i];
            delete[] strideB[i];
            delete[] strideC[i];
        }
    }

    inline void eval() override {
        if (!this->evaluated) {
            this->in1->eval();
            this->in2->eval();

            val_func(this->in1->value.val, this->in2->value.val,
                     this->value.val, strideA[0], strideB[0], strideC[0],
                     c_shape[0], a_offset[0], b_offset[0], k[0], D);
            this->evaluated = true;
        }
    }

    inline void getGrad() override {
        grad_func(this->in1->value.val, this->in1->gradient.val,
                  this->in2->value.val, this->in2->gradient.val,
                  this->value.val, this->gradient.val, strideA + 1, strideB + 1,
                  strideC + 1, c_shape + 1, a_offset + 1, b_offset + 1, k + 1,
                  D);

        if (this->in1->hasInputs) {
            this->in1->getGrad();
        }
        if (this->in2->hasInputs) {
            this->in2->getGrad();
        }
    }
};

template <typename T> struct Node_sum_dim : INode<T> {
    sumDimOp<T> val_func = Operations::sum_dim;
    sumDimGrad<T> grad_func = Gradients::sum_dim;

    int *strideA = nullptr;
    int *strideC = nullptr;
    size_t *a_offset = nullptr;
    size_t D = 0;

    template <typename... Args>
    Node_sum_dim(INode<T> *in1_ptr, Args &&...args)
        : INode<T>(in1_ptr, args...) {}

    ~Node_sum_dim() {
        delete[] strideA;
        delete[] strideC;
        delete[] a_offset;
    }

    inline void eval() override {
        if (!this->evaluated) {
            this->in1->eval();

            val_func(this->in1->value.val, this->value.val, strideA, strideC,
                     a_offset, D);
            this->evaluated = true;
        }
    }

    inline void getGrad() override {
        grad_func(this->in1->gradient.val, this->gradient.val, strideA, strideC,
                  a_offset, D);

        if (this->in1->hasInputs) {
            this->in1->getGrad();
        }
    }
};

template <typename T> struct Node_mean_dim : INode<T> {
    meanDimOp<T> val_func = Operations::mean_dim;
    meanDimGrad<T> grad_func = Gradients::mean_dim;

    int *strideA = nullptr;
    int *strideC = nullptr;
    size_t *a_offset = nullptr;
    T *c_end = nullptr;
    T *dA_end = nullptr;
    size_t D = 0;
    T divisor = 0;

    template <typename... Args>
    Node_mean_dim(INode<T> *in1_ptr, Args &&...args)
        : INode<T>(in1_ptr, args...) {}

    ~Node_mean_dim() {
        delete[] strideA;
        delete[] strideC;
        delete[] a_offset;
    }

    inline void eval() override {
        if (!this->evaluated) {
            this->in1->eval();

            val_func(this->in1->value.val, this->value.val, strideA, strideC,
                     a_offset, D, divisor, c_end);
            this->evaluated = true;
        }
    }

    inline void getGrad() override {
        grad_func(this->in1->value.val, this->in1->gradient.val,
                  this->value.val, this->gradient.val, strideA, strideC,
                  a_offset, D, divisor, dA_end);

        if (this->in1->hasInputs) {
            this->in1->getGrad();
        }
    }
};

template <typename T> struct Node_slice : INode<T> {
    sliceOp<T> val_func = Operations::slice<T>;
    sliceGrad<T> grad_func = Gradients::slice<T>;

    int *strideA = nullptr;
    int *strideB = nullptr;
    int *strideC = nullptr;
    size_t *start_offset = nullptr;
    size_t *c_offset = nullptr;
    size_t D;

    template <typename... Args>
    Node_slice(INode<T> *in1_ptr, Args &&...args)
        : INode<T>(in1_ptr, args...) {}

    ~Node_slice() {
        delete[] strideA;
        delete[] strideB;
        delete[] strideC;
        delete[] c_offset;
    }

    inline void eval() override {
        if (!this->evaluated) {
            this->in1->eval();

            val_func(this->in1->value.val, this->value.val, strideA, strideC,
                     start_offset, c_offset, D);
            this->evaluated = true;
        }
    }

    inline void getGrad() override {
        grad_func(this->in1->gradient.val, this->gradient.val, strideA, strideC,
                  start_offset, c_offset, D);

        if (this->in1->hasInputs) {
            this->in1->getGrad();
        }
    }
};

} // namespace kaad
