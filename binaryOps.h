#include "gradients.h"
#include "compGraph.h"
#include "strides.h"

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
int binaryOp(CompGraph<T>& rec, int indA, int indB, const BinaryKernels<T> kernels, const char* opName) {
    int recLen = rec.nodes.size();
    Tensor<T>& A = rec.nodes[indA]->value;
    Tensor<T>& B = rec.nodes[indB]->value;
    bool A_scalar = A.shapeLen == 1 && A.shape[0] == 1;
    bool B_scalar = B.shapeLen == 1 && B.shape[0] == 1;

    size_t newLen = max(A.shapeLen, B.shapeLen);
    int* newShape = new int[newLen];

    if (B_scalar) {
        copy(A.shape, A.shape + A.shapeLen, newShape);

        auto newNode = std::make_unique<Node_binary<T>>(kernels.scalarOpRt, kernels.scalarGradRt, indA, indB, newShape, A.shapeLen);
        newNode->len[0] = newNode->value.len;
        newNode->len[1] = newNode->value.len;
        rec.nodes.push_back(move(newNode));
    }
    else if (A_scalar) {
        copy(B.shape, B.shape + B.shapeLen, newShape);

        auto newNode = std::make_unique<Node_binary<T>>(kernels.scalarOpLt, kernels.scalarGradLt, indA, indB, newShape, B.shapeLen);
        newNode->len[0] = newNode->value.len;
        newNode->len[1] = newNode->value.len;
        rec.nodes.push_back(move(newNode));
    }
    else if (A.shapeLen == B.shapeLen && equal(A.shape, A.shape + A.shapeLen, B.shape)) {
        copy(A.shape, A.shape + A.shapeLen, newShape);

        auto newNode = std::make_unique<Node_binary<T>>(kernels.pointOp, kernels.pointGrad, indA, indB, newShape, A.shapeLen);
        newNode->len[0] = newNode->value.len;
        newNode->len[1] = newNode->value.len;
        rec.nodes.push_back(move(newNode));
    }
    else if (combine_flexible(A.shape, A.shapeLen, B.shape, B.shapeLen, newShape, newLen)) {
        auto newNode = std::make_unique<Node_binary_flex<T>>(kernels.flexOp, kernels.flexGrad, indA, indB, newShape, newLen);
        ///
        rec.nodes.push_back(move(newNode));
    }
    else {
        ostringstream errmsg;
        errmsg << "shape error in node[" << recLen << "] (" << opName << "), tensor shapes are not broadcastable: shape1 ";
        print_arr(A.shape, A.shape + A.shapeLen, errmsg);
        errmsg << ", shape2 ";
        print_arr(B.shape, B.shape + B.shapeLen, errmsg);
        throw invalid_argument(errmsg.str());
    }
    return recLen;
}

// add A and B
// where A and B are Tensors with Broadcastable shapes
template <typename T>
int add(CompGraph<T>& rec, int indA, int indB) {
    
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

    return binaryOp(rec, indA, indB, addK, "add");
}

// subtract B from A
// where A and B are Tensors with Broadcastable shapes
template <typename T>
int sub(CompGraph<T>& rec, int indA, int indB) {
    
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

    return binaryOp(rec, indA, indB, subK, "sub");
}

// multiply A and B
// where A and B are Tensors with Broadcastable shapes
template <typename T>
int mul(CompGraph<T>& rec, int indA, int indB) {
    
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

    return binaryOp(rec, indA, indB, mulK, "mul");
}

// divide A by B
// where A and B are Tensors with Broadcastable shapes
template <typename T>
int div(CompGraph<T>& rec, int indA, int indB) {
    
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

    return binaryOp(rec, indA, indB, divK, "div");
}

// raise A to the power of B
// where A and B are Tensors with Broadcastable shapes
template <typename T>
int pow(CompGraph<T>& rec, int indA, int indB) {
    
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

    return binaryOp(rec, indA, indB, powK, "pow");
}

// compute dot prodcut of A and B
// where A and B are scalars or vectors with the same length
template <typename T>
int dot(CompGraph<T>& rec, int indA, int indB) {
    int recLen = rec.nodes.size();
    Tensor<T>& A = rec.nodes[indA].value;
    Tensor<T>& B = rec.nodes[indB].value;

    bool A_scalar = A.shapeLen == 1 && A.shape[0] == 1;
    bool B_scalar = B.shapeLen == 1 && B.shape[0] == 1;

    if (B_scalar) {
        rec.nodes.emplace_back(Operations<T>::scalarDot, Gradients<T>::scalarDot_grad, indA, indB, ((T)0));
        Strides<T>::iterOverInp(rec.nodes[indA].value, rec.nodes[indB].value, rec.nodes[recLen]);
    }
    else if (A_scalar) {
        rec.nodes.emplace_back(Operations<T>::scalarDot, Gradients<T>::scalarDot_grad, indB, indA, ((T)0));
        Strides<T>::iterOverInp(rec.nodes[indA].value, rec.nodes[indB].value, rec.nodes[recLen]);
    }
    else if (A.shapeLen == B.shapeLen && equal(A.shape, A.shape + A.shapeLen, B.shape)) {
        rec.nodes.emplace_back(Operations<T>::dot, Gradients<T>::dot_grad, indA, indB, ((T)0));
        Strides<T>::iterOverInp(rec.nodes[indA].value, rec.nodes[indB].value, rec.nodes[recLen]);
    }
    else {
        ostringstream errmsg;
        errmsg << "shape error in node[" << recLen << "] (dot), tensor shapes arent valid for dot product: shape1 ";
        print_arr(A.shape, A.shape + A.shapeLen, errmsg);
        errmsg << ", shape2 ";
        print_arr(B.shape, B.shape + B.shapeLen, errmsg);
        throw invalid_argument(errmsg.str());
    }

    return recLen;
}

// matrix multiply A and B
// where A and B are Tensors with valid dimensions
template <typename T>
int matmul(CompGraph<T>& rec, int indA, int indB) {
    int recLen = rec.nodes.size();
    Tensor<T>& A = rec.nodes[indA].value;
    Tensor<T>& B = rec.nodes[indB].value;

    size_t newLen = max(A.shapeLen, B.shapeLen);
    int* newShape = new int[newLen];

    const char* opName = newLen == 2 ? "matmul" : "batch_matmul";
    if (!combine_matrix(A.shape, A.shapeLen, B.shape, B.shapeLen, newShape, newLen)) {
        ostringstream errmsg;
        errmsg << "shape error in node[" << recLen << "] (" << opName << "), tensor shapes arent valid for " << opName << ": shape1 ";
        print_arr(A.shape, A.shape + A.shapeLen, errmsg);
        errmsg << ", shape2 ";
        print_arr(B.shape, B.shape + B.shapeLen, errmsg);
        throw invalid_argument(errmsg.str());
    }

    if (newLen == 2) {
        rec.nodes.emplace_back(Operations<T>::matmul, Gradients<T>::matmul_grad, indA, indB, newShape, newLen);
        Strides<T>::matmul(rec.nodes[indA].value, rec.nodes[indB].value, rec.nodes[recLen]);
    }
    else {
        rec.nodes.emplace_back(Operations<T>::batch_matmul, Gradients<T>::batch_matmul_grad, indA, indB, newShape, newLen);
        Strides<T>::batch_matmul(rec.nodes[indA].value, rec.nodes[indB].value, rec.nodes[recLen]);
    }
    
    return recLen;
}

// compute outer product of A and B
// where A and B are Tensors
template <typename T>
int outer(CompGraph<T>& rec, int indA, int indB) {
    int recLen = rec.nodes.size();
    Tensor<T>& A = rec.nodes[indA].value;
    Tensor<T>& B = rec.nodes[indB].value;

    size_t newLen = A.shapeLen + B.shapeLen;
    int* newShape = new int[newLen];
    copy(A.shape, A.shape + A.shapeLen, newShape);
    copy(B.shape, B.shape + B.shapeLen, newShape + A.shapeLen);

    rec.nodes.emplace_back(Operations<T>::flexMul, Gradients<T>::flexMul_grad, indA, indB, newShape, newLen);
    Strides<T>::outer(rec.nodes[indA].value, rec.nodes[indB].value, rec.nodes[recLen]);

    return recLen;
}

// compute pointwise minimum of A and B
// where A and B are Tensors with broadcastable shapes
template <typename T>
int minimum(CompGraph<T>& rec, int indA, int indB) {
    
    static const  BinaryKernels<T> minK = {
        Operations<T>::scalarMinRt,
        Operations<T>::scalarMinLt,
        Operations<T>::pointMin,
        Operations<T>::flexMin,
        Gradients<T>::scalarMinRt_grad,
        Gradients<T>::scalarMinLt_grad,
        Gradients<T>::pointMin_grad,
        Gradients<T>::flexMin_grad
    };

    return binaryOp(rec, indA, indB, minK, "minimum");
}

// compute pointwise maximum of A and B
// where A and B are Tensors with broadcastable shapes
template <typename T>
int maximum(CompGraph<T>& rec, int indA, int indB) {
    
    static const  BinaryKernels<T> maxK = {
        Operations<T>::scalarMaxRt,
        Operations<T>::scalarMaxLt,
        Operations<T>::pointMax,
        Operations<T>::flexMax,
        Gradients<T>::scalarMaxRt_grad,
        Gradients<T>::scalarMaxLt_grad,
        Gradients<T>::pointMax_grad,
        Gradients<T>::flexMax_grad
    };

    return binaryOp(rec, indA, indB, maxK, "maximum");
}
