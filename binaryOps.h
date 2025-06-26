#include "gradients.h"
#include "compGraph.h"
#include "strides.h"

template <typename T>
struct BinaryKernels {
    binaryOp<T> scalarOpRt;
    binaryOp<T> scalarOpLt;
    binaryOp<T> pointOp;
    flexBinaryOp<T> flexOp;
    
    binaryGrad<T> scalarGradRt;
    binaryGrad<T> scalarGradLt;
    binaryGrad<T> pointGrad;
    flexBinaryGrad<T> flexGrad;
};

template <typename T>
INode<T>* binOperator(CompGraph<T>& rec, INode<T>* A_ptr, INode<T>* B_ptr, const BinaryKernels<T> kernels, const char* opName) {
    int recLen = rec.nodes.size();
    Tensor<T>& A = A_ptr->value;
    Tensor<T>& B = B_ptr->value;
    bool A_scalar = A.shapeLen == 1 && A.shape[0] == 1;
    bool B_scalar = B.shapeLen == 1 && B.shape[0] == 1;

    size_t newLen = max(A.shapeLen, B.shapeLen);
    int* newShape = new int[newLen];

    if (B_scalar) {
        copy(A.shape, A.shape + A.shapeLen, newShape);

        auto newNode = std::make_unique<Node_binary<T>>(kernels.scalarOpRt, kernels.scalarGradRt, A_ptr, B_ptr, newShape, A.shapeLen);
        newNode->len[0] = newNode->value.len;
        newNode->len[1] = newNode->value.len;
        rec.nodes.push_back(move(newNode));
    }
    else if (A_scalar) {
        copy(B.shape, B.shape + B.shapeLen, newShape);

        auto newNode = std::make_unique<Node_binary<T>>(kernels.scalarOpLt, kernels.scalarGradLt, A_ptr, B_ptr, newShape, B.shapeLen);
        newNode->len[0] = newNode->value.len;
        newNode->len[1] = newNode->value.len;
        rec.nodes.push_back(move(newNode));
    }
    else if (A.shapeLen == B.shapeLen && equal(A.shape, A.shape + A.shapeLen, B.shape)) {
        copy(A.shape, A.shape + A.shapeLen, newShape);

        auto newNode = std::make_unique<Node_binary<T>>(kernels.pointOp, kernels.pointGrad, A_ptr, B_ptr, newShape, A.shapeLen);
        newNode->len[0] = newNode->value.len;
        newNode->len[1] = newNode->value.len;
        rec.nodes.push_back(move(newNode));
    }
    else if (combine_flexible(A.shape, A.shapeLen, B.shape, B.shapeLen, newShape, newLen)) {
        auto newNode = std::make_unique<Node_binary_flex<T>>(kernels.flexOp, kernels.flexGrad, A_ptr, B_ptr, newShape, newLen);
        Strides<T>::flexible(A, B, *newNode.get());
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
    return rec.nodes[recLen].get();
}

// add A and B
// where A and B are Tensors with Broadcastable shapes
template <typename T>
INode<T>* add(CompGraph<T>& rec, INode<T>* A_ptr, INode<T>* B_ptr) {
    
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

    return binOperator(rec, A_ptr, B_ptr, addK, "add");
}

// subtract B from A
// where A and B are Tensors with Broadcastable shapes
template <typename T>
INode<T>* sub(CompGraph<T>& rec, INode<T>* A_ptr, INode<T>* B_ptr) {
    
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

    return binOperator(rec, A_ptr, B_ptr, subK, "sub");
}

// multiply A and B
// where A and B are Tensors with Broadcastable shapes
template <typename T>
INode<T>* mul(CompGraph<T>& rec, INode<T>* A_ptr, INode<T>* B_ptr) {
    
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

    return binOperator(rec, A_ptr, B_ptr, mulK, "mul");
}

// divide A by B
// where A and B are Tensors with Broadcastable shapes
template <typename T>
INode<T>* div(CompGraph<T>& rec, INode<T>* A_ptr, INode<T>* B_ptr) {
    
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

    return binOperator(rec, A_ptr, B_ptr, divK, "div");
}

// raise A to the power of B
// where A and B are Tensors with Broadcastable shapes
template <typename T>
INode<T>* pow(CompGraph<T>& rec, INode<T>* A_ptr, INode<T>* B_ptr) {
    
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

    return binOperator(rec, A_ptr, B_ptr, powK, "pow");
}

// compute dot prodcut of A and B
// where A and B are scalars or vectors with the same length
template <typename T>
INode<T>* dot(CompGraph<T>& rec, INode<T>* A_ptr, INode<T>* B_ptr) {
    int recLen = rec.nodes.size();
    Tensor<T>& A = A_ptr->value;
    Tensor<T>& B = B_ptr->value;

    bool A_scalar = A.shapeLen == 1 && A.shape[0] == 1;
    bool B_scalar = B.shapeLen == 1 && B.shape[0] == 1;

    if (B_scalar) {
        auto newNode = make_unique<Node_binary<T>>(Operations<T>::scalarDot, Gradients<T>::scalarDot_grad, A_ptr, B_ptr, ((T)0));

        newNode->len[0] = A.len;
        newNode->len[1] = A.len;
        rec.nodes.push_back(move(newNode));
    }
    else if (A_scalar) {
        auto newNode = make_unique<Node_binary<T>>(Operations<T>::scalarDot, Gradients<T>::scalarDot_grad, B_ptr, A_ptr, ((T)0));
        newNode->len[0] = B.len;
        newNode->len[1] = B.len;
        rec.nodes.push_back(move(newNode));
    }
    else if (A.shapeLen == B.shapeLen && equal(A.shape, A.shape + A.shapeLen, B.shape)) {
        auto newNode = make_unique<Node_binary<T>>(Operations<T>::dot, Gradients<T>::dot_grad, A_ptr, B_ptr, ((T)0));
        newNode->len[0] = A.len;
        newNode->len[1] = A.len;
        rec.nodes.push_back(move(newNode));

    }
    else {
        ostringstream errmsg;
        errmsg << "shape error in node[" << recLen << "] (dot), tensor shapes arent valid for dot product: shape1 ";
        print_arr(A.shape, A.shape + A.shapeLen, errmsg);
        errmsg << ", shape2 ";
        print_arr(B.shape, B.shape + B.shapeLen, errmsg);
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
        auto newNode = make_unique<Node_matmul<T>>(Operations<T>::matmul, Gradients<T>::matmul_grad, A_ptr, B_ptr, newShape, newLen);
        Strides<T>::matmul(A, B, *newNode.get());
        rec.nodes.push_back(move(newNode));
    }
    else {
        auto newNode = make_unique<Node_batch_matmul<T>>(Operations<T>::batch_matmul, Gradients<T>::batch_matmul_grad, A_ptr, B_ptr, newShape, newLen);
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

    size_t newLen = A.shapeLen + B.shapeLen;
    int* newShape = new int[newLen];
    copy(A.shape, A.shape + A.shapeLen, newShape);
    copy(B.shape, B.shape + B.shapeLen, newShape + A.shapeLen);

    auto newNode = make_unique<Node_binary_flex<T>>(Operations<T>::flexMul, Gradients<T>::flexMul_grad, A_ptr, B_ptr, newShape, newLen);
    auto raw = newNode.get();
    Strides<T>::outer(A, B, *raw);
    rec.nodes.push_back(move(newNode));

    return raw;
}

// compute pointwise minimum of A and B
// where A and B are Tensors with broadcastable shapes
template <typename T>
INode<T>* minimum(CompGraph<T>& rec, INode<T>* A_ptr, INode<T>* B_ptr) {
    
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

    return binOperator(rec, A_ptr, B_ptr, minK, "minimum");
}

// compute pointwise maximum of A and B
// where A and B are Tensors with broadcastable shapes
template <typename T>
INode<T>* maximum(CompGraph<T>& rec, INode<T>* A_ptr, INode<T>* B_ptr) {
    
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

    return binOperator(rec, A_ptr, B_ptr, maxK, "maximum");
}
