#include "gradients.h"
#include "recorder.h"
#include "strides.h"

template <typename T>
struct UnaryKernels {
    tensorOp<T> Op;
    gradientOp<T> Grad;
};

template <typename T>
int binaryOp(Recorder<T>& rec, int indA, const UnaryKernels<T> kernels) {
    int recLen = rec.nodes.size();
    Tensor<T>& A = rec.nodes[indA].value;

    int* newShape = new int[A.shapeLen];
    copy(A.shape, A.shape + A.shapeLen, newShape);

    rec.nodes.emplace_back(kernels.Op, kernels.Grad, indA, -1, newShape, A.shapeLen);
    Strides<T>::pointwise(rec.nodes[recLen]);

    return recLen;
}

template <typename T>
int negative(Recorder<T>& rec, int indA) {
    static const UnaryKernels negK = {
        Operations<T>::negate,
        Gradients<T>::negate_grad
    };

    return binaryOp(rec, indA, negK);
}

template <typename T>
int square(Recorder<T>& rec, int indA) {
    static const UnaryKernels squareK = {
        Operations<T>::square,
        Gradients<T>::square_grad
    };

    return binaryOp(rec, indA, squareK);
}

template <typename T>
int sqrt(Recorder<T>& rec, int indA) {
    static const UnaryKernels sqrtK = {
        Operations<T>::sqrt,
        Gradients<T>::sqrt_grad
    };

    return binaryOp(rec, indA, sqrtK);
}

template <typename T>
int log(Recorder<T>& rec, int indA) {
    static const UnaryKernels logK = {
        Operations<T>::log,
        Gradients<T>::log_grad
    };

    return binaryOp(rec, indA, logK);
}

template <typename T>
int exp(Recorder<T>& rec, int indA) {
    static const UnaryKernels expK = {
        Operations<T>::exp,
        Gradients<T>::exp_grad
    };

    return binaryOp(rec, indA, expK);
}

template <typename T>
int abs(Recorder<T>& rec, int indA) {
    static const UnaryKernels absK = {
        Operations<T>::abs,
        Gradients<T>::abs_grad
    };

    return binaryOp(rec, indA, absK);
}