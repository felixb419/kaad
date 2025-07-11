#pragma once

#include <stddef.h>      // for size_t
#include <algorithm>     // for fill
#include <utility>       // for forward
#include "gradients.h"   // for batchmatmulGrad, binaryGrad, flexBinaryGrad
#include "operations.h"  // for batchmatmulOp, binaryOp, flexBinaryOp, flexU...
#include "tensor.h"      // for Tensor

namespace kaad {
    template <typename T>
    struct INode {

        INode<T>* in1 = nullptr;

        Tensor<T> value;
        Tensor<T> gradient;

        bool evaluated = false;
        bool hasInputs = false;

        // construct as evaluated
        INode(Tensor<T>&& tensor)
        : value(std::move(tensor)), gradient(value), evaluated(true), hasInputs(false) {
            std::fill(gradient.val, gradient.val + gradient.len, 0);
        }

        // construct as not evaluated
        template <typename... Args>
        INode(INode<T>* in1_ptr, Args&&... args)
        : in1(in1_ptr), evaluated(false), value(std::forward<Args>(args)...), hasInputs(true), gradient(value) {
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

    template <typename T>
    struct Node_valued : INode<T> {
        
        Node_valued(Tensor<T>&& tensor) : INode<T>(std::move(tensor)) {}

        void eval() override {}
        void getGrad() override {}
    };

    template <typename T, class Kernel>
    struct Node_unary : INode<T> {
        using Op = class Kernel::Op;
        Op op;
        unaryOp<T,Op> val_func = nullptr;
        using Grad = class Kernel::Grad;
        Grad grad;
        unaryGrad<T,Grad> grad_func = nullptr;

        size_t len = 0;

        template <typename... Args>
        Node_unary(unaryOp<T,Op> operation, unaryGrad<T,Grad> derivative, INode<T>* in1_ptr, Args&&... args)
        : val_func(operation), grad_func(derivative), INode<T>(in1_ptr, args...) {}

        inline void eval() override {
            if (!this->evaluated) {
                this->in1->eval();

                val_func(this->in1->value.val, this->value.val, len, op);
                this->evaluated = true;
            }
        }

        inline void getGrad() override {
            grad_func(this->in1->value.val, this->in1->gradient.val, this->value.val, this->gradient.val, len, grad);

            if (this->in1->hasInputs) {
                this->in1->getGrad();
            }
        }
    };

    template <typename T, class Kernel>
    struct Node_unary_flex : INode<T> {
        using Op = class Kernel::Op;
        Op op;
        flexUnaryOp<T,Op> val_func = nullptr;
        using Grad = class Kernel::Grad;
        Grad grad;
        flexUnaryGrad<T,Grad> grad_func = nullptr;

        int* strideA = nullptr;
        int* strideC = nullptr;
        int* reps = nullptr;
        int* count = nullptr;
        size_t D = 0;

        template <typename... Args>
        Node_unary_flex(flexUnaryOp<T,Op> operation, flexUnaryGrad<T,Grad> derivative, INode<T>* in1_ptr, Args&&... args)
        : val_func(operation), grad_func(derivative), INode<T>(in1_ptr, args...) {}

        ~Node_unary_flex() {
            delete[] strideA;
            delete[] strideC;
            delete[] reps;
            delete[] count;
        }

        inline void eval() override {
            if (!this->evaluated) {
                this->in1->eval();

                val_func(this->in1->value.val, this->value.val, strideA, strideC, reps, count, D, op);
                this->evaluated = true;
            }
        }

        inline void getGrad() override {
            grad_func(this->in1->value.val, this->in1->gradient.val, this->value.val,  this->gradient.val, strideA, strideC, reps, count, D, grad);

            if (this->in1->hasInputs) {
                this->in1->getGrad();
            }
        }
    };

    template <typename T, class Kernel>
    struct Node_binary : INode<T> {
        INode<T>* in2 = nullptr;

        using Op = class Kernel::Op;
        Op op;
        binaryOp<T,Op> val_func = nullptr;
        using Grad = class Kernel::Grad;
        Grad grad;
        binaryGrad<T,Grad> grad_func = nullptr;

        size_t len = 0;

        template <typename... Args>
        Node_binary(binaryOp<T,Op> operation, binaryGrad<T,Grad> derivative, INode<T>* in1_ptr, INode<T>* in2_ptr, Args&&... args)
        : in2(in2_ptr), val_func(operation), grad_func(derivative), INode<T>(in1_ptr, args...) {}

        inline void eval() override {
            if (!this->evaluated) {
                this->in1->eval();
                this->in2->eval();

                val_func(this->in1->value.val, in2->value.val, this->value.val, len, op);
                this->evaluated = true;
            }
        }

        inline void getGrad() override {
            grad_func(this->in1->value.val, this->in1->gradient.val, in2->value.val, in2->gradient.val, this->value.val, this->gradient.val, len, grad);

            if (this->in1->hasInputs) {
                this->in1->getGrad();
            }
            if (this->in2->hasInputs) {
                this->in2->getGrad();
            }
        }
    };

    template <typename T, class Kernel>
    struct Node_binary_flex : INode<T> {
        INode<T>* in2 = nullptr;

        using Op = class Kernel::Op;
        Op op;
        flexBinaryOp<T,Op> val_func = nullptr;
        using Grad = class Kernel::Grad;
        Grad grad;
        flexBinaryGrad<T,Grad> grad_func = nullptr;

        int* strideA = nullptr;
        int* strideB = nullptr;
        int* strideC = nullptr;
        int* c_shape = nullptr;
        size_t D;

        template <typename... Args>
        Node_binary_flex(flexBinaryOp<T,Op> operation, flexBinaryGrad<T,Grad> derivative, INode<T>* in1_ptr, INode<T>* in2_ptr, Args&&... args)
        : in2(in2_ptr), val_func(operation), grad_func(derivative), INode<T>(in1_ptr, args...) {}

        ~Node_binary_flex() {
            delete[] strideA;
            delete[] strideB;
            delete[] strideC;
            delete[] c_shape;
        }

        inline void eval() override {
            if (!this->evaluated) {
                this->in1->eval();
                this->in2->eval();

                val_func(this->in1->value.val, this->in2->value.val, this->value.val, strideA, strideB, strideC, c_shape, D-1, op);
                this->evaluated = true;
            }
        }

        inline void getGrad() override {
            grad_func(this->in1->value.val, this->in1->gradient.val, this->in2->value.val, this->in2->gradient.val, this->value.val, this->gradient.val,
                strideA, strideB, strideC, c_shape, D-1, grad);

            if (this->in1->hasInputs) {
                this->in1->getGrad();
            }
            if (this->in2->hasInputs) {
                this->in2->getGrad();
            }
        }
    };

    template <typename T>
    struct Node_matmul : INode<T> {
        INode<T>* in2 = nullptr;

        matmulOp<T> op = nullptr;
        matmulGrad<T> grad_op = nullptr;

        int a_dim[3];
        int b_dim[3];
        int k[3];
        int strideA[6];
        int strideB[6];
        int strideC[6];

        template <typename... Args>
        Node_matmul(matmulOp<T> operation, matmulGrad<T> derivative, INode<T>* in1_ptr, INode<T>* in2_ptr, Args&&... args)
        : in2(in2_ptr), op(operation), grad_op(derivative), INode<T>(in1_ptr, args...) {}
        
        inline void eval() override {
            if (!this->evaluated) {
                this->in1->eval();
                this->in2->eval();

                op(this->in1->value.val, this->in2->value.val, this->value.val, a_dim[0], b_dim[0], k[0], strideA, strideB, strideC);
                this->evaluated = true;
            }
        }

        inline void getGrad() override {
            grad_op(this->in1->value.val, this->in1->gradient.val, this->in2->value.val, this->in2->gradient.val, this->value.val, this->gradient.val, a_dim+1, b_dim+1, k+1, strideA+2, strideB+2, strideC+2);

            if (this->in1->hasInputs) {
                this->in1->getGrad();
            }
            if (this->in2->hasInputs) {
                this->in2->getGrad();
            }
        }
    };

    template <typename T>
    struct Node_batch_matmul : INode<T> {
        INode<T>* in2 = nullptr;

        batchmatmulOp<T> op = nullptr;
        batchmatmulGrad<T> grad_op = nullptr;

        int a_offset[3];
        int b_offset[3];
        int k[3];
        int* strideA[3];
        int* strideB[3];
        int* strideC[3];
        int* reps[3];
        int* count[3];
        size_t D[3];

        template <typename... Args>
        Node_batch_matmul(batchmatmulOp<T> operation, batchmatmulGrad<T> derivative, INode<T>* in1_ptr, INode<T>* in2_ptr, Args&&... args)
        : in2(in2_ptr), op(operation), grad_op(derivative), INode<T>(in1_ptr, args...) {}

        ~Node_batch_matmul() {
            for (int i = 0; i < 3; i++) {
                delete[] strideA[i];
                delete[] strideB[i];
                delete[] strideC[i];
                delete[] reps[i];
                delete[] count[i];
            }
        }

        inline void eval() override {
            if (!this->evaluated) {
                this->in1->eval();
                this->in2->eval();

                op(this->in1->value.val, this->in2->value.val, this->value.val,
                    a_offset[0], b_offset[0], k[0], strideA[0], strideB[0], strideC[0], reps[0], count[0], D[0]);
                this->evaluated = true;
            }
        }

        inline void getGrad() override {
            grad_op(this->in1->value.val, this->in1->gradient.val, this->in2->value.val, this->in2->gradient.val, this->value.val, this->gradient.val,
                a_offset+1, b_offset+1, k+1, strideA+1, strideB+1, strideC+1, reps+1, count+1, D+1);

            if (this->in1->hasInputs) {
                this->in1->getGrad();
            }
            if (this->in2->hasInputs) {
                this->in2->getGrad();
            }
        }
    };

    template <typename T>
    struct Node_mean_dim : INode<T> {
        meanDimOp<T> op = nullptr;
        meanDimGrad<T> grad_op = nullptr;

        T divisor = 0;
        size_t c_len[2];
        int* strideA = nullptr;
        int* strideC = nullptr;
        int* reps = nullptr;
        int* count = nullptr;
        size_t D = 0;

        template <typename... Args>
        Node_mean_dim(meanDimOp<T> operation, meanDimGrad<T> derivative, INode<T>* in1_ptr, Args&&... args)
        : op(operation), grad_op(derivative), INode<T>(in1_ptr, args...) {}

        ~Node_mean_dim() {
            delete[] strideA;
            delete[] strideC;
            delete[] reps;
            delete[] count;
        }

        inline void eval() override {
            if (!this->evaluated) {
                this->in1->eval();

                op(this->in1->value.val, this->value.val, divisor, c_len[0], strideA, strideC, reps, count, D);
                this->evaluated = true;
            }
        }

        inline void getGrad() override {
            grad_op(this->in1->value.val, this->in1->gradient.val, this->value.val,  this->gradient.val, divisor, c_len[1], strideA, strideC, reps, count, D);

            if (this->in1->hasInputs) {
                this->in1->getGrad();
            }
        }
    };
}    
