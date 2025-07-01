#include "gradients.h"
#include "compGraph.h"
#include "strides.h"
#include "kernels.h"

template <typename T, class Kernel>
struct UnaryKernels {
    using Op = class Kernel::Op;
    using Grad = class Kernel::Grad;
    unaryOp<T,Op> op = Operations<T,Op>::unary_pointwise;
    unaryGrad<T,Grad> grad = Gradients<T,Grad>::unary_pointwise;
};

template <typename T, class Kernel>
INode<T>* unOperator(CompGraph<T>& rec, INode<T>* A_ptr , const UnaryKernels<T,Kernel> kernels) {
    Tensor<T>& A = A_ptr->value;

    int* newShape = new int[A.nDims];
    copy(A.shape, A.shape + A.nDims, newShape);

    auto newNode = make_unique<Node_unary<T,Kernel>>(kernels.op, kernels.grad, A_ptr, newShape, A.nDims);
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
INode<T>* transpose(CompGraph<T>& rec, INode<T>* A_ptr, initializer_list<int> perm={}) {
    int recLen = rec.nodes.size();
    Tensor<T>& A = A_ptr->value;

    if (A.nDims < 2) {
        ostringstream errmsg;
        errmsg << "shape error in node[" << recLen << "] (transpose), A nDims hast to be > 1, shape1 ";
        print_arr(A.shape, A.shape + A.nDims, errmsg);
        throw invalid_argument(errmsg.str());
    }

    int *shape_T, *stride_T;
    if (perm.size() == 0) {
        shape_T = new int[A.nDims];
        stride_T = new int[A.nDims];
        copy(A.shape, A.shape + A.nDims, shape_T);
        copy(A.stride, A.stride + A.nDims, stride_T);
        
        transp(A.shape, A.stride, A.nDims, shape_T, stride_T);
    }
    else {
        if (perm.size() != A.nDims) {
            ostringstream errmsg;
            errmsg << "shape error in node[" << recLen << "] (transpose), length of perm has to be same as nDimsgth of A, perm ";
            print_arr(perm.begin(), perm.end(), errmsg);
            errmsg << ", shape1";
            print_arr(A.shape, A.shape + A.nDims, errmsg);
            throw invalid_argument(errmsg.str());
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
                ostringstream errmsg;
                errmsg << "argument error in node[" << recLen << "] (transpose), invalid permutation, perm has to contain index of every dimension exactly once, perm ";
                print_arr(perm.begin(), perm.end(), errmsg);
                throw invalid_argument(errmsg.str());
            }
        }
    }

    using Kernel = class Kernels<T>::Transp;
    using Op = class Kernel::Op;
    using Grad = class Kernel::Grad;
    unaryOp<T,Op> op = Operations<T,Op>::transpose;
    unaryGrad<T,Grad> grad = Gradients<T,Grad>::unary_pointwise;

    auto newNode = make_unique<Node_unary<T,Kernel>>(op, grad, A_ptr, shape_T, stride_T, A.nDims);
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

        using Op = typename NullOp::Op;
        using Grad = typename NullOp::Grad;
        unaryOp<T,Op> op = Operations<T,Op>::sum;
        unaryGrad<T,Grad> grad = Gradients<T,Grad>::sum_grad;
        auto newNode = make_unique<Node_unary<T,NullOp>>(op, grad, A_ptr, newShape, 1);
        newNode->len = A_ptr->value.len;
        rec.nodes.push_back(move(newNode));
    }
    else {
        if (dim < 0 || dim >= A.nDims) {
            ostringstream errmsg;
            errmsg << "shape error in node[" << recLen << "] (sum), dim has to be a valid shape index, dim=" << dim << ", nDims=" << A.nDims << endl;
            throw invalid_argument(errmsg.str());
        }

        size_t newLen = A.nDims - 1;
        int* newShape = new int[newLen];

        copy(A.shape, A.shape + dim, newShape);
        copy(A.shape + dim + 1, A.shape + A.nDims, newShape + dim);

        using Kernel = class Kernels<T>::Sum;
        using Op = class Kernel::Op;
        using Grad = class Kernel::Grad;
        flexUnaryOp<T,Op> op = Operations<T,Op>::unary_flexible;
        flexUnaryGrad<T,Grad> grad = Gradients<T,Grad>::unary_flexible;

        auto newNode = make_unique<Node_unary_flex<T,Kernel>>(op, grad, A_ptr, newShape, newLen);
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
        unaryOp<T,Op> op = Operations<T,Op>::mean;
        unaryGrad<T,Grad> grad = Gradients<T,Grad>::mean_grad;
        auto newNode = make_unique<Node_unary<T,NullOp>>(op, grad, A_ptr, newShape, 1);
        newNode->len = A_ptr->value.len;
        rec.nodes.push_back(move(newNode));
    }
    else {
        if (dim < 0 || dim >= A.nDims) {
            ostringstream errmsg;
            errmsg << "shape error in node[" << recLen << "] (mean), dim has to be a valid shape index, dim=" << dim << ", nDims=" << A.nDims << endl;
            throw invalid_argument(errmsg.str());
        }

        size_t newLen = A.nDims - 1;
        int* newShape = new int[newLen];

        copy(A.shape, A.shape + dim, newShape);
        copy(A.shape + dim + 1, A.shape + A.nDims, newShape + dim);

        using Op = typename NullOp::Op;
        using Grad = typename NullOp::Grad;
        meanDimOp<T> op = Operations<T,Op>::mean_dim;
        meanDimGrad<T> grad = Gradients<T,Grad>::mean_dim_grad;
        auto newNode = make_unique<Node_mean_dim<T>>(op, grad, A_ptr, newShape, newLen);
        Strides<T>::mean_along_dim(A, *newNode.get(), dim);
        rec.nodes.push_back(move(newNode));
    }

    return rec.nodes.back().get();
}