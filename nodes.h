#include "tensor.h"
#include "gradients.h"

#include <array>
#include <utility>
#include <memory>

template <typename T>
struct INode {

    INode<T>* in1 = nullptr;

    Tensor<T> value;
    Tensor<T> gradient;

    bool evaluated = false;
    bool hasInputs = false;

    // construct as evaluated
    INode(Tensor<T>&& tensor)
    : value(move(tensor)), gradient(value), evaluated(true), hasInputs(false) {
        fill(gradient.val, gradient.val + gradient.len, 0.0);
    }

    // construct as not evaluated
    template <typename... Args>
    INode(INode<T>* in1_ptr, Args&&... args)
    : in1(in1_ptr), evaluated(false), value(forward<Args>(args)...), hasInputs(true), gradient(value) {
        fill(value.val, value.val + value.len, 0);
        fill(gradient.val, gradient.val + gradient.len, 0.0);
    }

    virtual ~INode() = default;
    
    inline void reset() {
        fill(value.val, value.val + value.len, 0);
        fill(gradient.val, gradient.val + gradient.len, 0);
    }

    inline virtual void eval() = 0;
    inline virtual void grad() = 0;
};

template <typename T>
struct Node_valued : INode<T> {
    
    Node_valued(Tensor<T>&& tensor) : INode<T>(move(tensor)) {}

    void eval() override {}
    void grad() override {}
};

template <typename T>
struct Node_unary : INode<T> {
    unaryOp<T> op = nullptr;
    unaryGrad<T> grad_op = nullptr;

    size_t len[2] = { 0, 0 };

    template <typename... Args>
    Node_unary(unaryOp<T> operation, unaryGrad<T> derivative, INode<T>* in1_ptr, Args&&... args)
    : op(operation), grad_op(derivative), INode<T>(in1_ptr, args...) {}

    inline void eval() override {
        if (!this->evaluated) {
            this->in1.eval();
            this->in2.eval();

            op(this->in1.value.val, this->value.val, len[0]);
            this->evaluated = true;
        }
    }

    inline void grad() override {
        grad_op(this->in1->value.val, this->in1->gradient.val, this->value.val, this->gradient->val, len[1]);

        if (this->in1.hasInputs) {
            this->in1.grad();
        }
        if (this->in2.hasInputs) {
            this->in2.grad();
        }
    }
};

template <typename T>
struct Node_unary_flex : INode<T> {
    flexUnaryOp<T> op = nullptr;
    flexUnaryGrad<T> grad_op = nullptr;

    int* strideA = nullptr;
    int* strideC = nullptr;
    int* reps = nullptr;
    int* count = nullptr;
    size_t* strideLen = nullptr;

    template <typename... Args>
    Node_unary_flex(flexUnaryOp<T> operation, flexUnaryGrad<T> derivative, INode<T>* in1_ptr, Args&&... args)
    : op(operation), grad_op(derivative), INode<T>(in1_ptr, args...) {}

    ~Node_unary_flex() {
        delete[] strideA;
        delete[] strideC;
        delete[] reps;
        delete[] count;
        delete[] strideLen;
    }

    inline void eval() override {
        if (!this->evaluated) {
            this->in1.eval();
            this->in2.eval();

            op(this->in1.value.val, this->value.val, strideA, strideC, reps, count, strideLen);
            this->evaluated = true;
        }
    }

    inline void grad() override {
        grad_op(this->in1->value.val, this->in1->gradient.val, this->value.val,  this->gradient.val, strideA, strideC, reps, count, strideLen);

        if (this->in1->hasInputs) {
            this->in1->grad();
        }
        if (this->in2->hasInputs) {
            this->in2->grad();
        }
    }
};

template <typename T>
struct Node_binary : INode<T> {
    INode<T>* in2 = nullptr;

    binaryOp<T> op = nullptr;
    binaryGrad<T> grad_op = nullptr;

    size_t len[2] = { 0, 0 };

    template <typename... Args>
    Node_binary(binaryOp<T> operation, binaryGrad<T> derivative, INode<T>* in1_ptr, INode<T>* in2_ptr, Args&&... args)
    : in2(in2_ptr), op(operation), grad_op(derivative), INode<T>(in1_ptr, args...) {}

    inline void eval() override {
        if (!this->evaluated) {
            this->in1->eval();
            this->in2->eval();

            op(this->in1->value.val, in2->value.val, this->value.val, len[0]);
            this->evaluated = true;
        }
    }

    inline void grad() override {
        grad_op(this->in1->value.val, this->in1->gradient.val, in2->value.val, in2->gradient.val, this->value.val, this->gradient.val, len[1]);

        if (this->in1->hasInputs) {
            this->in1->grad();
        }
        if (this->in2->hasInputs) {
            this->in2->grad();
        }
    }
};

template <typename T>
struct Node_binary_flex : INode<T> {
    INode<T>* in2 = nullptr;

    flexBinaryOp<T> op = nullptr;
    flexBinaryGrad<T> grad_op = nullptr;

    int* strideA = nullptr;
    int* strideB = nullptr;
    int* strideC = nullptr;
    int* reps = nullptr;
    int* count = nullptr;
    size_t strideLen;

    template <typename... Args>
    Node_binary_flex(flexBinaryOp<T> operation, flexBinaryGrad<T> derivative, INode<T>* in1_ptr, INode<T>* in2_ptr, Args&&... args)
    : in2(in2_ptr), op(operation), grad_op(derivative), INode<T>(in1_ptr, args...) {}

    ~Node_binary_flex() {
        delete[] strideA;
        delete[] strideB;
        delete[] strideC;
        delete[] reps;
        delete[] count;
    }

    inline void eval() override {
        if (!this->evaluated) {
            this->in1->eval();
            this->in2->eval();

            op(this->in1->value.val, this->in2->value.val, this->value.val, strideA, strideB, strideC, reps, count, strideLen);
            this->evaluated = true;
        }
    }

    inline void grad() override {
        grad_op(this->in1->value.val, this->in1->gradient.val, this->in2->value.val, this->in2->gradient.val, this->value.val, this->gradient.val, strideA, strideB, strideC, reps, count, strideLen);

        if (this->in1->hasInputs) {
            this->in1->grad();
        }
        if (this->in2->hasInputs) {
            this->in2->grad();
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

    inline void grad() override {
        grad_op(this->in1->value.val, this->in1->gradient.val, this->in2->value.val, this->in2->gradient.val, this->value.val, this->gradient.val, a_dim+1, b_dim+1, k+1, strideA+2, strideB+2, strideC+2);

        if (this->in1->hasInputs) {
            this->in1->grad();
        }
        if (this->in2->hasInputs) {
            this->in2->grad();
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
    size_t strideLen[3];

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
                a_offset[0], b_offset[0], k[0], strideA[0], strideB[0], strideC[0], reps[0], count[0], strideLen[0]);
            this->evaluated = true;
        }
    }

    inline void grad() override {
        grad_op(this->in1->value.val, this->in1->gradient.val, this->in2->value.val, this->in2->gradient.val, this->value.val, this->gradient.val,
            a_offset+1, b_offset+1, k+1, strideA+1, strideB+1, strideC+1, reps+1, count+1, strideLen+1);

        if (this->in1->hasInputs) {
            this->in1->grad();
        }
        if (this->in2->hasInputs) {
            this->in2->grad();
        }
    }
};