#include "gradients.h"
#include "recorder.h"
#include "strides.h"

template <typename T>
struct UnaryKernels {
    tensorOp<T> Op;
    gradientOp<T> Grad;
};

template <typename T>
int unaryOp(Recorder<T>& rec, int indA, const UnaryKernels<T> kernels) {
    int recLen = rec.nodes.size();
    Tensor<T>& A = rec.nodes[indA].value;

    int* newShape = new int[A.shapeLen];
    copy(A.shape, A.shape + A.shapeLen, newShape);

    rec.nodes.emplace_back(kernels.Op, kernels.Grad, indA, -1, newShape, A.shapeLen);
    Strides<T>::pointwise(rec.nodes[recLen]);

    return recLen;
}

// negate A
// where A is a tensor
template <typename T>
int negative(Recorder<T>& rec, int indA) {
    static const UnaryKernels negK = {
        Operations<T>::negate,
        Gradients<T>::negate_grad
    };

    return unaryOp(rec, indA, negK);
}

// square A
// where A is a tensor
template <typename T>
int square(Recorder<T>& rec, int indA) {
    static const UnaryKernels squareK = {
        Operations<T>::square,
        Gradients<T>::square_grad
    };

    return unaryOp(rec, indA, squareK);
}

// compte squareroot of A
// where A is a tensor
template <typename T>
int sqrt(Recorder<T>& rec, int indA) {
    static const UnaryKernels sqrtK = {
        Operations<T>::sqrt,
        Gradients<T>::sqrt_grad
    };

    return unaryOp(rec, indA, sqrtK);
}

// compute logarithm base e of A
// where A is a tensor
template <typename T>
int log(Recorder<T>& rec, int indA) {
    static const UnaryKernels logK = {
        Operations<T>::log,
        Gradients<T>::log_grad
    };

    return unaryOp(rec, indA, logK);
}

// raise A to the power of e
// where A is a tensor
template <typename T>
int exp(Recorder<T>& rec, int indA) {
    static const UnaryKernels expK = {
        Operations<T>::exp,
        Gradients<T>::exp_grad
    };

    return unaryOp(rec, indA, expK);
}

// compute the absolute value of A
// where A is a tensor
template <typename T>
int abs(Recorder<T>& rec, int indA) {
    static const UnaryKernels absK = {
        Operations<T>::abs,
        Gradients<T>::abs_grad
    };

    return unaryOp(rec, indA, absK);
}

// transpose A
// if given A is transposed according to perm
// where A is a tensor
template <typename T>
int transpose(Recorder<T>& rec, int indA, initializer_list<int> perm={}) {
    int recLen = rec.nodes.size();
    Tensor<T>& A = rec.nodes[indA].value;

    if (A.shapeLen < 2) {
        ostringstream errmsg;
        errmsg << "shape error in node[" << recLen << "] (transpose), In1 shapeLen hast to be > 1, shape1 ";
        print_arr(A.shape, A.shape + A.shapeLen, errmsg);
        throw invalid_argument(errmsg.str());
    }

    if (perm.size() == 0) {
        int* shape_T = new int[A.shapeLen];
        int* stride_T = new int[A.shapeLen];
        copy(A.shape, A.shape + A.shapeLen, shape_T);
        copy(A.stride, A.stride + A.shapeLen, stride_T);
        
        transp(A.shape, A.stride, A.shapeLen, shape_T, stride_T);
        rec.nodes.emplace_back(Operations<T>::transpose, Gradients<T>::transp_grad, indA, -1, shape_T, stride_T, A.shapeLen);
        Strides<T>::pointwise(rec.nodes[recLen]);
    }
    else {
        if (perm.size() != A.shapeLen) {
            ostringstream errmsg;
            errmsg << "shape error in node[" << recLen << "] (transpose), perm has to be the same shapeLength as In1, perm ";
            print_arr(perm.begin(), perm.end(), errmsg);
            errmsg << ", shape1";
            print_arr(A.shape, A.shape + A.shapeLen, errmsg);
            throw invalid_argument(errmsg.str());
        }

        int* shape_T = new int[A.shapeLen];
        int* stride_T = new int[A.shapeLen];

        int* sh = shape_T;
        int* st = stride_T;
        for (int idx : perm) {
            *(sh++) = A.shape[idx];
            *(st++) = A.stride[idx];
        } 

        rec.nodes.emplace_back(Operations<T>::transpose, Gradients<T>::transp_grad, indA, -1, shape_T, stride_T, A.shapeLen);
        Strides<T>::pointwise(rec.nodes[recLen]);
    }

    return recLen;
}

template <typename T>
int sum(Recorder<T>& rec, int indA) {
    int recLen = rec.nodes.size();
    
    int* newShape = new int[] { 1 };
    rec.nodes.emplace_back(Operations<T>::sum, Gradients<T>::sum_grad, indA, -1, newShape, 1);
    Strides<T>::iterOverInp(rec.nodes[indA].value, rec.nodes[indA].value, rec.nodes[recLen]);

    return recLen;
}

template <typename T>
int sum(Recorder<T>& rec, int indA, int dim) {
    int recLen = rec.nodes.size();
    Tensor<T>& A = rec.nodes[indA].value;

    if (dim < 0 || dim >= A.shapeLen) {
        ostringstream errmsg;
        errmsg << "shape error in node[" << recLen << "] (sum), dim has to be a valid shape index, dim=" << dim << ", shapeLen=" << A.shapeLen << endl;
        throw invalid_argument(errmsg.str());
    }

    size_t newLen = A.shapeLen - 1;
    int* newShape = new int[newLen];

    copy(A.shape, A.shape + dim, newShape);
    copy(A.shape + dim + 1, A.shape + A.shapeLen, newShape + dim);

    rec.nodes.emplace_back(Operations<T>::sum_dim, Gradients<T>::sum_dim_grad, indA, -1, newShape, newLen);
    Strides<T>::along_dim(rec.nodes[indA].value, rec.nodes[recLen], dim);

    return recLen;
}

template <typename T>
int mean(Recorder<T>& rec, int indA) {
    int recLen = rec.nodes.size();
    Tensor<T>& A = rec.nodes[indA].value;
    
    int* newShape = new int[] { 1 };
    rec.nodes.emplace_back(Operations<T>::mean, Gradients<T>::mean_grad, indA, -1, newShape, 1);
    Strides<T>::iterOverInp(rec.nodes[indA].value, rec.nodes[indA].value, rec.nodes[recLen]);

    return recLen;
}

template <typename T>
int mean(Recorder<T>& rec, int indA, int dim) {
    int recLen = rec.nodes.size();
    Tensor<T>& A = rec.nodes[indA].value;

    if (dim < 0 || dim >= A.shapeLen) {
        ostringstream errmsg;
        errmsg << "shape error in node[" << recLen << "] (mean), dim has to be a valid shape index, dim=" << dim << ", shapeLen=" << A.shapeLen << endl;
        throw invalid_argument(errmsg.str());
    }

    size_t newLen = A.shapeLen - 1;
    int* newShape = new int[newLen];

    copy(A.shape, A.shape + dim, newShape);
    copy(A.shape + dim + 1, A.shape + A.shapeLen, newShape + dim);

    rec.nodes.emplace_back(Operations<T>::mean_dim, Gradients<T>::mean_dim_grad, indA, -1, newShape, newLen);
    Strides<T>::mean_along_dim(rec.nodes[indA].value, rec.nodes[recLen], dim);

    return recLen;
}
