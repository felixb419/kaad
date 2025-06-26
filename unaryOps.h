#include "gradients.h"
#include "compGraph.h"
#include "strides.h"

template <typename T>
struct UnaryKernels {
    unaryOp<T> Op;
    unaryGrad<T> Grad;
};

template <typename T>
INode<T>* unOperator(CompGraph<T>& rec, INode<T>* A_ptr , const UnaryKernels<T> kernels) {
    Tensor<T>& A = A_ptr->value;

    int* newShape = new int[A.shapeLen];
    copy(A.shape, A.shape + A.shapeLen, newShape);

    auto newNode = make_unique<Node_unary<T>>(kernels.Op, kernels.Grad, A_ptr, newShape, A.shapeLen);
    auto raw = newNode.get();
    rec.nodes.push_back(move(newNode));
    raw->len = A.len;

    return raw;
}

// negate A
// where A is a tensor
template <typename T>
INode<T>* negative(CompGraph<T>& rec, INode<T>* A_ptr) {
    static const UnaryKernels<T> negK = {
        Operations<T>::negate,
        Gradients<T>::negate_grad
    };

    return unOperator(rec, A_ptr, negK);
}

// square A
// where A is a tensor
template <typename T>
INode<T>* square(CompGraph<T>& rec, INode<T>* A_ptr) {
    static const UnaryKernels<T> squareK = {
        Operations<T>::square,
        Gradients<T>::square_grad
    };

    return unOperator(rec, A_ptr, squareK);
}

// compte squareroot of A
// where A is a tensor
template <typename T>
INode<T>* sqrt(CompGraph<T>& rec, INode<T>* A_ptr) {
    static const UnaryKernels<T> sqrtK = {
        Operations<T>::sqrt,
        Gradients<T>::sqrt_grad
    };

    return unOperator(rec, A_ptr, sqrtK);
}

// compute logarithm base e of A
// where A is a tensor
template <typename T>
INode<T>* log(CompGraph<T>& rec, INode<T>* A_ptr) {
    static const UnaryKernels<T> logK = {
        Operations<T>::log,
        Gradients<T>::log_grad
    };

    return unOperator(rec, A_ptr, logK);
}

// raise A to the power of e
// where A is a tensor
template <typename T>
INode<T>* exp(CompGraph<T>& rec, INode<T>* A_ptr) {
    static const UnaryKernels<T> expK = {
        Operations<T>::exp,
        Gradients<T>::exp_grad
    };

    return unOperator(rec, A_ptr, expK);
}

// compute the absolute value of A
// where A is a tensor
template <typename T>
INode<T>* abs(CompGraph<T>& rec, INode<T>* A_ptr) {
    static const UnaryKernels<T> absK = {
        Operations<T>::abs,
        Gradients<T>::abs_grad
    };

    return unOperator(rec, A_ptr, absK);
}

// transpose A
// if given A is transposed according to perm
// where A is a tensor
template <typename T>
INode<T>* transpose(CompGraph<T>& rec, INode<T>* A_ptr, initializer_list<int> perm={}) {
    int recLen = rec.nodes.size();
    Tensor<T>& A = A_ptr->value;

    if (A.shapeLen < 2) {
        ostringstream errmsg;
        errmsg << "shape error in node[" << recLen << "] (transpose), In1 shapeLen hast to be > 1, shape1 ";
        print_arr(A.shape, A.shape + A.shapeLen, errmsg);
        throw invalid_argument(errmsg.str());
    }

    int *shape_T, *stride_T;
    if (perm.size() == 0) {
        shape_T = new int[A.shapeLen];
        stride_T = new int[A.shapeLen];
        copy(A.shape, A.shape + A.shapeLen, shape_T);
        copy(A.stride, A.stride + A.shapeLen, stride_T);
        
        transp(A.shape, A.stride, A.shapeLen, shape_T, stride_T);
    }
    else {
        if (perm.size() != A.shapeLen) {
            ostringstream errmsg;
            errmsg << "shape error in node[" << recLen << "] (transpose), length of perm has to be same as shapeLength of In1, perm ";
            print_arr(perm.begin(), perm.end(), errmsg);
            errmsg << ", shape1";
            print_arr(A.shape, A.shape + A.shapeLen, errmsg);
            throw invalid_argument(errmsg.str());
        }

        shape_T = new int[A.shapeLen];
        stride_T = new int[A.shapeLen];
        int* count = new int[A.shapeLen];
        fill(count, count + A.shapeLen, 0);

        int* sh = shape_T;
        int* st = stride_T;
        for (int idx : perm) {

            count[idx]++;

            *(sh++) = A.shape[idx];
            *(st++) = A.stride[idx];
        } 
        for (int* p = count; p != count + A.shapeLen; p++) {
            if (*p != 1) {
                ostringstream errmsg;
                errmsg << "argument error in node[" << recLen << "] (transpose), invalid permutation, perm has to contain index of every dimension exactly once, perm ";
                print_arr(perm.begin(), perm.end(), errmsg);
                throw invalid_argument(errmsg.str());
            }
        }
    }
    auto newNode = make_unique<Node_unary<T>>(Operations<T>::transpose, Gradients<T>::transp_grad, A_ptr, shape_T, stride_T, A.shapeLen);
    newNode->len = A.len;
    rec.nodes.push_back(move(newNode));

    return rec.nodes.back().get();
}

template <typename T>
INode<T>* sum(CompGraph<T>& rec, INode<T>* A_ptr, int dim=-1) {
    int recLen = rec.nodes.size();
    Tensor<T>& A = A_ptr->value;

    if (dim == -1) {
        int* newShape = new int[] { 1 };

        auto newNode = make_unique<Node_unary<T>>(Operations<T>::sum, Gradients<T>::sum_grad, A_ptr, newShape, 1);
        newNode->len = A_ptr->value.len;
        rec.nodes.push_back(move(newNode));
    }
    else {
        if (dim < 0 || dim >= A.shapeLen) {
            ostringstream errmsg;
            errmsg << "shape error in node[" << recLen << "] (sum), dim has to be a valid shape index, dim=" << dim << ", shapeLen=" << A.shapeLen << endl;
            throw invalid_argument(errmsg.str());
        }

        size_t newLen = A.shapeLen - 1;
        int* newShape = new int[newLen];

        copy(A.shape, A.shape + dim, newShape);
        copy(A.shape + dim + 1, A.shape + A.shapeLen, newShape + dim);

        auto newNode = make_unique<Node_unary_flex<T>>(Operations<T>::sum_dim, Gradients<T>::sum_dim_grad, A_ptr, newShape, newLen);
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

        auto newNode = make_unique<Node_unary<T>>(Operations<T>::mean, Gradients<T>::mean_grad, A_ptr, newShape, 1);
        newNode->len = A_ptr->value.len;
        rec.nodes.push_back(move(newNode));
    }
    else {
        if (dim < 0 || dim >= A.shapeLen) {
            ostringstream errmsg;
            errmsg << "shape error in node[" << recLen << "] (mean), dim has to be a valid shape index, dim=" << dim << ", shapeLen=" << A.shapeLen << endl;
            throw invalid_argument(errmsg.str());
        }

        size_t newLen = A.shapeLen - 1;
        int* newShape = new int[newLen];

        copy(A.shape, A.shape + dim, newShape);
        copy(A.shape + dim + 1, A.shape + A.shapeLen, newShape + dim);

        auto newNode = make_unique<Node_mean_dim<T>>(Operations<T>::mean_dim, Gradients<T>::mean_dim_grad, A_ptr, newShape, newLen);
        Strides<T>::mean_along_dim(A, *newNode.get(), dim);
        rec.nodes.push_back(move(newNode));
    }

    return rec.nodes.back().get();
}
