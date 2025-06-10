#include "gradients.h"
#include "recorder.h"
#include "strides.h"

template <typename T>
using tensorOp = void(*)(const T* A, const T* B, T* C, int* strideA, int* strideB, int* strideC, int* reps, int* count, size_t strideLen);
template <typename T>
using gradientOp = void(*)(const T* A, const T* B, const T* C, T* dA, T* dB, const T* dC, int** strideA, int** strideB, int** strideC, int** reps, int** count, size_t* stridelen);

template <typename T>
struct BinaryKernels {
    tensorOp<T> scalarOpRt;
    tensorOp<T> scalarOpLt;
    tensorOp<T> pointOp;
    tensorOp<T> flexOp;
    
    gradientOp<T> scalarGradRt;
    gradientOp<T> scalarGradLt;
    gradientOp<T> pointGrad;
    gradientOp<T> flexGrad;
};

template <typename T>
int binaryOp(Recorder<T>& rec, int indA, int indB, const BinaryKernels<T> kernels) {
    int recLen = rec.nodes.size();
    Tensor<T>& A = rec.nodes[indA].value;
    Tensor<T>& B = rec.nodes[indB].value;
    bool A_scalar = A.shapeLen == 1 && A.shape[0] == 1;
    bool B_scalar = B.shapeLen == 1 && B.shape[0] == 1;

    if (B_scalar) {
        int* newShape = new int[A.shapeLen];
        copy(A.shape, A.shape + A.shapeLen, newShape);

        rec.nodes.emplace_back(kernels.scalarOpRt, kernels.scalarGradRt, indA, indB, newShape, A.shapeLen);
        Strides<T>::pointwise(rec.nodes[recLen]);
    }
    else if (A_scalar) {
        int* newShape = new int[B.shapeLen];
        copy(B.shape, B.shape + B.shapeLen, newShape);

        rec.nodes.emplace_back(kernels.scalarOpLt, kernels.scalarGradLt, indA, indB, newShape, B.shapeLen);
        Strides<T>::pointwise(rec.nodes[recLen]);
    }
    else if (A.shapeLen == B.shapeLen && equal(A.shape, A.shape + A.shapeLen, B.shape)) {
        int* newShape = new int[A.shapeLen];
        copy(A.shape, A.shape + A.shapeLen, newShape);

        rec.nodes.emplace_back(kernels.pointOp, kernels.pointGrad ,indA, indB, newShape, A.shapeLen);
        Strides<T>::pointwise(rec.nodes[recLen]);
    }
    else {
        size_t newLen = max(A.shapeLen, B.shapeLen);
        int* newShape = new int[newLen];
        combine_flexible(A.shape, A.shapeLen, B.shape, B.shapeLen, newShape, newLen);

        rec.nodes.emplace_back(kernels.flexOp, kernels.flexGrad, indA, indB, newShape, newLen);
        Strides<T>::flexible(rec.nodes[indA].value, rec.nodes[indB].value, rec.nodes[recLen]); 
    }
    return recLen;
}

// add A and B
// where A and B are Tensors with Broadcastable shapes
template <typename T>
int add(Recorder<T>& rec, int indA, int indB) {
    
    static const BinaryKernels<T> addK = {
        Operations<T>::scalarAddRt,
        Operations<T>::scalarAddLt,
        Operations<T>::pointAdd,
        Operations<T>::flexAdd,
        Gradients<T>::scalarAddRt_grad,
        Gradients<T>::scalarAddLt_grad,
        Gradients<T>::pointAdd_grad,
        Gradients<T>::flexAdd_grad
    };

    return binaryOp(rec, indA, indB, addK);
}

// subtract B from A
// where A and B are Tensors with Broadcastable shapes
template <typename T>
int sub(Recorder<T>& rec, int indA, int indB) {
    
    static const BinaryKernels<T> subK = {
        Operations<T>::scalarSubRt,
        Operations<T>::scalarSubLt,
        Operations<T>::pointSub,
        Operations<T>::flexSub,
        Gradients<T>::scalarSubRt_grad,
        Gradients<T>::scalarSubLt_grad,
        Gradients<T>::pointSub_grad,
        Gradients<T>::flexSub_grad
    };

    return binaryOp(rec, indA, indB, subK);
}

// multiply A and B
// where A and B are Tensors with Broadcastable shapes
template <typename T>
int mul(Recorder<T>& rec, int indA, int indB) {
    
    static const  BinaryKernels<T> mulK = {
        Operations<T>::scalarMulRt,
        Operations<T>::scalarMulLt,
        Operations<T>::pointMul,
        Operations<T>::flexMul,
        Gradients<T>::scalarMulRt_grad,
        Gradients<T>::scalarMulLt_grad,
        Gradients<T>::pointMul_grad,
        Gradients<T>::flexMul_grad
    };

    return binaryOp(rec, indA, indB, mulK);
}

// divide A by B
// where A and B are Tensors with Broadcastable shapes
template <typename T>
int div(Recorder<T>& rec, int indA, int indB) {
    
    static const  BinaryKernels<T> divK = {
        Operations<T>::scalarDivRt,
        Operations<T>::scalarDivLt,
        Operations<T>::pointDiv,
        Operations<T>::flexDiv,
        Gradients<T>::scalarDivRt_grad,
        Gradients<T>::scalarDivLt_grad,
        Gradients<T>::pointDiv_grad,
        Gradients<T>::flexDiv_grad
    };

    return binaryOp(rec, indA, indB, divK);
}

// raise A to the power of B
// where A and B are Tensors with Broadcastable shapes
template <typename T>
int pow(Recorder<T>& rec, int indA, int indB) {
    
    static const  BinaryKernels<T> powK = {
        Operations<T>::scalarPowRt,
        Operations<T>::scalarPowLt,
        Operations<T>::pointPow,
        Operations<T>::flexPow,
        Gradients<T>::scalarPowRt_grad,
        Gradients<T>::scalarPowLt_grad,
        Gradients<T>::pointPow_grad,
        Gradients<T>::flexPow_grad
    };

    return binaryOp(rec, indA, indB, powK);
}