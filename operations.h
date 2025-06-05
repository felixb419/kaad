#pragma once

#include "tensor.h"

#include <cmath>
#include <sstream>
#include <stdexcept>
#include <vector>

using namespace std;

// returns a dynamically allocated array that represents the resulting shape of broadcasting two tensors
// d1_n == d2_n || d1_n == 1 || d2_n == 1
void combine_flexible(int* shape1, const size_t shapeLen1, int* shape2, const size_t shapeLen2, int* newShape, size_t newLen) {
    int ind = newLen - 1;
    for (int i = 1; i <= newLen; i++, ind--) {
        int ind1 = shapeLen1 - i;
        int ind2 = shapeLen2 - i;
        if (ind1 >= 0 && ind2 >= 0) {
            if (shape1[ind1] != shape2[ind2] && shape1[ind1] != 1 && shape2[ind2] != 1) {
                ostringstream errmsg;
                errmsg << "can not broadcast Tensors with shapes ";
                print_arr(shape1, shapeLen1, errmsg);
                errmsg << " ";
                print_arr(shape2, shapeLen2, errmsg);
                throw invalid_argument(errmsg.str());
            }
            newShape[ind] = max(shape1[ind1], shape2[ind2]);
        }
        else {
            newShape[ind] = ind1 >= 0 ? shape1[ind1] : shape2[ind2];
        }
    }
}

// returns a dynamically allocated array that represents the resulting shape of broadcasting two tensors by matrix multiplication
// matmul: (n?,k),(k,m?) -> (n?,m?)
void combine_matrix(int* shape1, const size_t shapeLen1, int* shape2, const size_t shapeLen2, int* newShape, size_t newLen) {
    if (shape1[shapeLen1 - 1] != shape2[shapeLen2 - 2]) {
        ostringstream errmsg;
        errmsg << "can not matrix multiply Tensors with shapes ";
        print_arr(shape1, shapeLen1, errmsg);
        errmsg << " ";
        print_arr(shape2, shapeLen2, errmsg);
        throw invalid_argument(errmsg.str());
    }
    fill(newShape, newShape + newLen, 0);

    newShape[newLen - 1] = shape2[shapeLen2 - 1];
    newShape[newLen - 2] = shape1[shapeLen1 - 2];

    int ind = newLen - 3;
    for (int i = 3; i <= newLen; i++, ind--) {
        int ind1 = shapeLen1 - i;
        int ind2 = shapeLen2 - i;
        if (ind1 >= 0 && ind2 >= 0) {
            if (shape1[ind1] != shape2[ind2] && shape1[ind1] != 1 && shape2[ind2] != 1) {
                throw invalid_argument("Tensor shapes do not map");
            }
            newShape[ind] = max(shape1[ind1], shape2[ind2]);
        }
        else {
            newShape[ind] = ind1 >= 0 ? shape1[ind1] : shape2[ind2];
        }
    }
}

void transp(int* shape, int* stride, size_t len) {
    int temp;
    for (int i = 0, j = len - 1; i < len / 2; i++, j--) {
        temp = shape[i];
        shape[i] = shape[j];
        shape[j] = temp;

        temp = stride[i];
        stride[i] = stride[j];
        stride[j] = temp;
    }
}

void transp(int* shape, int* stride,size_t len, int* shape_T, int* stride_T) {
    for (int i = 0, j = len - 1; i < len; i++, j--) {
        shape_T[j] = shape[i];
        stride_T[j] = stride[i];
    }
}

template <typename T>
struct Operations {
    // add so that: out[m,n,...] = tensor[m,n,...] + scalar[0]
    // shapes of out and tensor must be the same, shape of scalar must be (1)
    static void scalarAdd(const T* A, const T* B, T* C, int* strideA, int* strideB, int* strideC, int* reps, int* count, size_t strideLen) {
        for (size_t i = 0; i < strideLen; i++) {
            C[i] = A[i] + B[0];
        }
    }
    // pointwise add so that: C = A + B
    // shape of all operands must be indentical
    static void pointAdd(const T* A, const T* B, T* C, int* strideA, int* strideB, int* strideC, int* reps, int* count, size_t strideLen) {
        for (size_t i = 0; i < strideLen; i++) {
            C[i] = A[i] + B[i];
        }
    }
    // flexible add so that: C = A + B
    // shape of C must be a valid broadcast of A and B
    static void flexAdd(const T* A, const T* B, T* C, int* strideA, int* strideB, int* strideC, int* reps, int* count, size_t strideLen) {
        int indA = 0, indB = 0, indC = 0;
        while (1) {

            C[indC] = A[indA] + B[indB];

            for (int dim = strideLen - 1; dim >= 0; dim--) {
                count[dim]--;
                if (count[dim] >= 0) {
                    indA += strideA[dim];
                    indB += strideB[dim];
                    indC += strideC[dim];
                    break;
                }

                count[dim] = reps[dim];
                if (dim == 0) goto end;
            }
        }
        end:;
    }
    // flexible add so that: A += B
    // shapes of A and B must be broadcastable
    static void flexAdd_inplace(tView<T>* A, const tView<T>* B, bool iterOverInp=true) {
        const tView<T>* big = iterOverInp ? B : A;
        const tView<T>* small = iterOverInp ? A : B;
        
        int offset = big->shapeLen - small->shapeLen;
        int* effstride = new int[big->shapeLen * 2];
    
        for (size_t i = 0; i < big->shapeLen; i++) {
            int dim = i - offset;
            effstride[i] = dim >= 0 && small->shape[dim] != 1 ? small->stride[dim] : 0;
        }
    
        int indA = 0, indB = 0;
        int* cords = effstride + big->shapeLen;
        fill(cords, cords + big->shapeLen, 0);
    
        int* strideA = iterOverInp ? effstride : A->stride;
        int* strideB = iterOverInp ? B->stride : effstride;
    
        for (int i = 0; i < big->len; i++) {
        
            A->val[indA] += B->val[indB];
            
            for (int dim = big->shapeLen - 1; dim >= 0; dim--) {
                cords[dim]++;
                indA += strideA[dim];
                indB += strideB[dim];
            
                if (cords[dim] < big->shape[dim]) {
                    break;
                }
                else {
                    cords[dim] = 0;
                    indA -= strideA[dim] * big->shape[dim];
                    indB -= strideB[dim] * big->shape[dim];
                }
            }
        }
        delete[] effstride;
    }
        
    // subtract so that: out[m,n,...] = tensor[m,n,...] - scalar[0]
    // shapes of out and tensor must be the same, shape of scalar must be (1)
    static void scalarSub(const T* A, const T* B, T* C, int* strideA, int* strideB, int* strideC, int* reps, int* count, size_t strideLen) {
        for (size_t i = 0; i < strideLen; i++) {
            C[i] = A[i] - B[0];
        }
    }
    // subtract so that: out[m,n,...] = scalar[0] - tensor[m,n,...]
    // shapes of out and tensor must be the same, shape of scalar must be (1)
    static void scalarSub_inv(const T* A, const T* B, T* C, int* strideA, int* strideB, int* strideC, int* reps, int* count, size_t strideLen) {
        for (size_t i = 0; i < strideLen; i++) {
            C[i] = A[0] - B[i]; 
        }
    }
    // pointwise subtract so that: C = A - B
    // shape of all operands must be indentical
    static void pointSub(const T* A, const T* B, T* C, int* strideA, int* strideB, int* strideC, int* reps, int* count, size_t strideLen) {
        for (size_t i = 0; i < strideLen; i++) {
            C[i] = A[i] - B[i];
        }
    }
    // flexible subtract so that: C = A - B
    // shape of C must be a valid broadcast of A and B
    static void flexSub(const T* A, const T* B, T* C, int* strideA, int* strideB, int* strideC, int* reps, int* count, size_t strideLen) {
        int indA = 0, indB = 0, indC = 0;
        while (1) {

            C[indC] = A[indA] - B[indB];

            for (int dim = strideLen - 1; dim >= 0; dim--) {
                count[dim]--;
                if (count[dim] >= 0) {
                    indA += strideA[dim];
                    indB += strideB[dim];
                    indC += strideC[dim];
                    break;
                }

                count[dim] = reps[dim];
                if (dim == 0) goto end;
            }
        }
        end:;
    }
    // flexible subtract so that: A = A - B
    // shapes of A and B must be broadcastable
    static void flexSub_inplace(tView<T>* A, const tView<T>* B, bool iterOverInp=true) {
        const tView<T>* big = iterOverInp ? B : A;
        const tView<T>* small = iterOverInp ? A : B;
        
        int offset = big->shapeLen - small->shapeLen;
        int* effstride = new int[B->shapeLen * 2];
    
        for (size_t i = 0; i < big->shapeLen; i++) {
            int dim = i - offset;
            effstride[i] = dim >= 0 && small->shape[dim] != 1 ? small->stride[dim] : 0;
        }
    
        int indA = 0, indB = 0;
        int* cords = effstride + big->shapeLen;
        fill(cords, cords + big->shapeLen, 0);
    
        int* strideA = iterOverInp ? effstride : A->stride;
        int* strideB = iterOverInp ? B->stride : effstride;
    
        for (int i = 0; i < big->len; i++) {
        
            A->val[indA] -= B->val[indB];
            
            for (int dim = big->shapeLen - 1; dim >= 0; dim--) {
                cords[dim]++;
                indA += strideA[dim];
                indB += strideB[dim];
            
                if (cords[dim] < big->shape[dim]) {
                    break;
                }
                else {
                    cords[dim] = 0;
                    indA -= strideA[dim] * big->shape[dim];
                    indB -= strideB[dim] * big->shape[dim];
                }
            }
        }
        delete[] effstride;
    }
        
    // multiply so that: out[m,n,...] = tensor[m,n,...] * scalar[0]
    // shapes of out and tensor must be the same, shape of scalar must be (1)
    static void scalarMul(const T* A, const T* B, T* C, int* strideA, int* strideB, int* strideC, int* reps, int* count, size_t strideLen) {
        for (size_t i = 0; i < strideLen; i++) {
            C[i] = A[i] * B[0];
        }
    }
    // pointwise multiply so that: C = A * B
    // shape of all operands must be indentical
    static void pointMul(const T* A, const T* B, T* C, int* strideA, int* strideB, int* strideC, int* reps, int* count, size_t strideLen) {
        for (size_t i = 0; i < strideLen; i++) {
            C[i] = A[i] * B[i];
        }
    }
    // flexible multiply so that: C = A * B
    // shape of C must be a valid broadcast of A and B
    static void flexMul(const T* A, const T* B, T* C, int* strideA, int* strideB, int* strideC, int* reps, int* count, size_t strideLen) {
        int indA = 0, indB = 0, indC = 0;
        while (1) {

            C[indC] = A[indA] * B[indB];

            for (int dim = strideLen - 1; dim >= 0; dim--) {
                count[dim]--;
                if (count[dim] >= 0) {
                    indA += strideA[dim];
                    indB += strideB[dim];
                    indC += strideC[dim];
                    break;
                }

                count[dim] = reps[dim];
                if (dim == 0) goto end;
            }
        }
        end:;
    }
    // flexible multiply so that: A = A * B
    // shapes of A and B must be broadcastable
    static void flexMul_inplace(tView<T>* A, const tView<T>* B, bool iterOverInp=true) {
        const tView<T>* big = iterOverInp ? B : A;
        const tView<T>* small = iterOverInp ? A : B;
        
        int offset = big->shapeLen - small->shapeLen;
        int* effstride = new int[big->shapeLen * 2];
    
        for (size_t i = 0; i < big->shapeLen; i++) {
            int dim = i - offset;
            effstride[i] = dim >= 0 && small->shape[dim] != 1 ? small->stride[dim] : 0;
        }
    
        int indA = 0, indB = 0;
        int* cords = effstride + big->shapeLen;
        fill(cords, cords + big->shapeLen, 0);
    
        int* strideA = iterOverInp ? effstride : A->stride;
        int* strideB = iterOverInp ? B->stride : effstride;
    
        for (int i = 0; i < big->len; i++) {
        
            A->val[indA] *= B->val[indB];
            
            for (int dim = big->shapeLen - 1; dim >= 0; dim--) {
                cords[dim]++;
                indA += strideA[dim];
                indB += strideB[dim];
            
                if (cords[dim] < big->shape[dim]) {
                    break;
                }
                else {
                    cords[dim] = 0;
                    indA -= strideA[dim] * big->shape[dim];
                    indB -= strideB[dim] * big->shape[dim];
                }
            }
        }
        delete[] effstride;
    }
        
    // divide so that: out[m,n,...] = tensor[m,n,...] - scalar[0]
    // shapes of out and tensor must be the same, shape of scalar must be (1)
    static void scalarDiv(const T* A, const T* B, T* C, int* strideA, int* strideB, int* strideC, int* reps, int* count, size_t strideLen) {
        for (size_t i = 0; i < strideLen; i++) {
            C[i] = A[i] / B[0];
        }
    }
    // divide so that: out[m,n,...] = scalar[0] - tensor[m,n,...]
    // shapes of out and tensor must be the same, shape of scalar must be (1)
    static void scalarDiv_inv(const T* A, const T* B, T* C, int* strideA, int* strideB, int* strideC, int* reps, int* count, size_t strideLen) {
        for (size_t i = 0; i < strideLen; i++) {
            C[i] = A[0] / B[i]; 
        }
    }
    // pointwise divide so that: C = A - B
    // shape of all operands must be indentical
    static void pointDiv(const T* A, const T* B, T* C, int* strideA, int* strideB, int* strideC, int* reps, int* count, size_t strideLen) {
        for (size_t i = 0; i < strideLen; i++) {
            C[i] = A[i] / B[i];
        }
    }
    // flexible divide so that: C = A - B
    // shape of C must be a valid broadcast of A and B
    static void flexDiv(const T* A, const T* B, T* C, int* strideA, int* strideB, int* strideC, int* reps, int* count, size_t strideLen) {
        int indA = 0, indB = 0, indC = 0;
        while (1) {

            C[indC] = A[indA] / B[indB];

            for (int dim = strideLen - 1; dim >= 0; dim--) {
                count[dim]--;
                if (count[dim] >= 0) {
                    indA += strideA[dim];
                    indB += strideB[dim];
                    indC += strideC[dim];
                    break;
                }

                count[dim] = reps[dim];
                if (dim == 0) goto end;
            }
        }
        end:;
    }
    // flexible divide so that: A = A - B
    // shapes of A and B must be broadcastable
    static void flexDiv_inplace(tView<T>* A, const tView<T>* B, bool iterOverInp=true) {
        const tView<T>* big = iterOverInp ? B : A;
        const tView<T>* small = iterOverInp ? A : B;
        
        int offset = big->shapeLen - small->shapeLen;
        int* effstride = new int[B->shapeLen * 2];
    
        for (size_t i = 0; i < big->shapeLen; i++) {
            int dim = i - offset;
            effstride[i] = dim >= 0 && small->shape[dim] != 1 ? small->stride[dim] : 0;
        }
    
        int indA = 0, indB = 0;
        int* cords = effstride + big->shapeLen;
        fill(cords, cords + big->shapeLen, 0);
    
        int* strideA = iterOverInp ? effstride : A->stride;
        int* strideB = iterOverInp ? B->stride : effstride;
    
        for (int i = 0; i < big->len; i++) {
        
            A->val[indA] /= B->val[indB];
            
            for (int dim = big->shapeLen - 1; dim >= 0; dim--) {
                cords[dim]++;
                indA += strideA[dim];
                indB += strideB[dim];
            
                if (cords[dim] < big->shape[dim]) {
                    break;
                }
                else {
                    cords[dim] = 0;
                    indA -= strideA[dim] * big->shape[dim];
                    indB -= strideB[dim] * big->shape[dim];
                }
            }
        }
        delete[] effstride;
    }
        
    // raise to power so that: out[m,n,...] = tensor[m,n,...] - scalar[0]
    // shapes of out and tensor must be the same, shape of scalar must be (1)
    static void scalarPow(const T* A, const T* B, T* C, int* strideA, int* strideB, int* strideC, int* reps, int* count, size_t strideLen) {
        for (size_t i = 0; i < C->len; i++) {
            C->val[i] = pow(A->val[i], B->val[0]);
        }
    }
    // raise to power so that: out[m,n,...] = scalar[0] - tensor[m,n,...]
    // shapes of out and tensor must be the same, shape of scalar must be (1)
    static void scalarPow_inv(const T* A, const T* B, T* C, int* strideA, int* strideB, int* strideC, int* reps, int* count, size_t strideLen) {
        for (size_t i = 0; i < C->len; i++) {
            C->val[i] = pow(A->val[0], B->val[i]);
        }
    }
    // pointwise raise to power so that: C = A - B
    // shape of all operands must be indentical
    static void pointPow(const T* A, const T* B, T* C, int* strideA, int* strideB, int* strideC, int* reps, int* count, size_t strideLen) {
        for (size_t i = 0; i < C->len; i++) {
            C->val[i] = pow(A->val[i], B->val[i]);
        }
    }
    // flexible raise to power so that: C = A - B
    // shape of C must be a valid broadcast of A and B
    static void flexPow(const T* A, const T* B, T* C, int* strideA, int* strideB, int* strideC, int* reps, int* count, size_t strideLen) {
        int offsetA = C->shapeLen - A->shapeLen;
        int offsetB = C->shapeLen - B->shapeLen;
    
        int* effstrideA = new int[C->shapeLen * 3];
        int* effstrideB = effstrideA + C->shapeLen;
    
        for (size_t i = 0; i < C->shapeLen; i++) {
            int aDim = i - offsetA;
            effstrideA[i] = aDim >= 0 && A->shape[aDim] != 1 ? A->stride[aDim] : 0;
            int bDim = i - offsetB;
            effstrideB[i] = bDim >= 0 && B->shape[bDim] != 1 ? B->stride[bDim] : 0;
        }
    
        int indA = 0, indB = 0, indC = 0;
        int* cords = effstrideB + C->shapeLen;
        fill(cords, cords + C->shapeLen, 0);
        for (int i = 0; i < C->len; i++) {
        
            C->val[indC] = pow(A->val[indA], B->val[indB]);
        
            for (int dim = C->shapeLen - 1; dim >= 0; dim--) {
                cords[dim]++;
                indA += effstrideA[dim];
                indB += effstrideB[dim];
                indC += C->stride[dim];
            
                if (cords[dim] < C->shape[dim]) {
                    break;
                }
                else {
                    cords[dim] = 0;
                    indA -= effstrideA[dim] * C->shape[dim];
                    indB -= effstrideB[dim] * C->shape[dim];
                    indC -= C->stride[dim] * C->shape[dim];
                }
            }
        
        }    
        delete[] effstrideA;
    }
    // flexible raise to power so that: A = A - B
    // shapes of A and B must be broadcastable
    static void flexPow_inplace(tView<T>* A, const tView<T>* B, bool iterOverInp=true) {
        const tView<T>* big = iterOverInp ? B : A;
        const tView<T>* small = iterOverInp ? A : B;
        
        int offset = big->shapeLen - small->shapeLen;
        int* effstride = new int[B->shapeLen * 2];
    
        for (size_t i = 0; i < big->shapeLen; i++) {
            int dim = i - offset;
            effstride[i] = dim >= 0 && small->shape[dim] != 1 ? small->stride[dim] : 0;
        }
    
        int indA = 0, indB = 0;
        int* cords = effstride + big->shapeLen;
        fill(cords, cords + big->shapeLen, 0);
    
        int* strideA = iterOverInp ? effstride : A->stride;
        int* strideB = iterOverInp ? B->stride : effstride;
    
        for (int i = 0; i < big->len; i++) {
        
            A->val[indA] = pow(A->val[indA], B->val[indB]);
            
            for (int dim = big->shapeLen - 1; dim >= 0; dim--) {
                cords[dim]++;
                indA += strideA[dim];
                indB += strideB[dim];
            
                if (cords[dim] < big->shape[dim]) {
                    break;
                }
                else {
                    cords[dim] = 0;
                    indA -= strideA[dim] * big->shape[dim];
                    indB -= strideB[dim] * big->shape[dim];
                }
            }
        }
        delete[] effstride;
    }
        
    // negate A so that out[i] = -A[i]
    // A and out must have same shape
    static void negate(const tView<T>* A, const tView<T>* _, tView<T>* out, int* strideA=nullptr, int* strideB=nullptr, int* strideC=nullptr, int* reps=nullptr, int* count=nullptr, size_t strideLen=0, void* ctx=nullptr) {
        for (size_t i = 0; i < A->len; i++) {
            out->val[i] = -A->val[i];
        }
    }
        
    // square A so that out[i] = A[i]*A[i]
    // A and out must have same shape
    static void square(const tView<T>* A, const tView<T>* _, tView<T>* out, int* strideA=nullptr, int* strideB=nullptr, int* strideC=nullptr, int* reps=nullptr, int* count=nullptr, size_t strideLen=0, void* ctx=nullptr) {
        for (size_t i = 0; i < A->len; i++) {
            out->val[i] = A->val[i] * A->val[i];
        }
    }
        
    // root of A so that out[i] = sqrt(A[i])
    // A and out must have same shape
    static void sqrt(const tView<T>* A, const tView<T>* _, tView<T>* out, int* strideA=nullptr, int* strideB=nullptr, int* strideC=nullptr, int* reps=nullptr, int* count=nullptr, size_t strideLen=0, void* ctx=nullptr) {
        for (size_t i = 0; i < A->len; i++) {
            out->val[i] = sqrt(A->val[i]);
        }
    }
        
    // log of A so that out[i] = ln(A[i])
    // A and out must have same shape
    static void log(const tView<T>* A, const tView<T>* _, tView<T>* out, int* strideA=nullptr, int* strideB=nullptr, int* strideC=nullptr, int* reps=nullptr, int* count=nullptr, size_t strideLen=0, void* ctx=nullptr) {
        for (size_t i = 0; i < A->len; i++) {
            out->val[i] = log(A->val[i]);
        }
    }
        
    // exponent of A so that out[i] = e^A[i]
    // A and out must have same shape
    static void exp(const tView<T>* A, const tView<T>* _, tView<T>* out, int* strideA=nullptr, int* strideB=nullptr, int* strideC=nullptr, int* reps=nullptr, int* count=nullptr, size_t strideLen=0, void* ctx=nullptr) {
        for (size_t i = 0; i < A->len; i++) {
            out->val[i] = exp(A->val[i]);
        }
    }
        
    // absolute value of A so that out[i] = abs(A[i])
    // A and out must have same shape
    static void abs(const tView<T>* A, const tView<T>* _, tView<T>* out, int* strideA=nullptr, int* strideB=nullptr, int* strideC=nullptr, int* reps=nullptr, int* count=nullptr, size_t strideLen=0, void* ctx=nullptr) {
        for (size_t i = 0; i < A->len; i++) {
            out->val[i] = abs(A->val[i]);
        }
    }

    // compute do product of A and B into C
    // A and B must be 1d vectors of same length, C must be scalar
    static void dot(const T* A, const T* B, T* C, int* strideA, int* strideB, int* strideC, int* reps, int* count, size_t strideLen) {
        C->val[0] = 0;
        for (size_t i = 0; i < A->len; i++) {
            C->val[0] += A->val[i] * B->val[i]; 
        }
    }

    // compute do product of A and B into C
    // A must be 1d vector, B and C must be scalar
    static void scalarDot(const T* A, const T* B, T* C, int* strideA, int* strideB, int* strideC, int* reps, int* count, size_t strideLen) {
        C->val[0] = 0;
        for (size_t i = 0; i < A->len; i++) {
            C->val[0] += A->val[i] * B->val[0]; 
        }
    }

    static void outer(const T* A, const T* B, T* C, int* strideA, int* strideB, int* strideC, int* reps, int* count, size_t strideLen) {
        int offsetA = C->shapeLen - A->shapeLen;
        int offsetB = C->shapeLen - B->shapeLen;
    
        int* effstrideA = new int[C->shapeLen * 3];
        int* effstrideB = effstrideA + C->shapeLen;
    
        fill(effstrideA, effstrideA + C->shapeLen, 0);
        copy(A->stride, A->stride + A->shapeLen, effstrideA);
        fill(effstrideB, effstrideB + C->shapeLen, 0);
        copy(B->stride, B->stride + B->shapeLen, effstrideB + C->shapeLen - B->shapeLen);

        int indA = 0, indB = 0, indC = 0;
        int* cords = effstrideB + C->shapeLen;
        fill(cords, cords + C->shapeLen, 0);
        for (int i = 0; i < C->len; i++) {
        
            C->val[indC] = A->val[indA] * B->val[indB];
        
            for (int dim = C->shapeLen - 1; dim >= 0; dim--) {
                cords[dim]++;
                indA += effstrideA[dim];
                indB += effstrideB[dim];
                indC += C->stride[dim];
            
                if (cords[dim] < C->shape[dim]) {
                    break;
                }
                else {
                    cords[dim] = 0;
                    indA -= effstrideA[dim] * C->shape[dim];
                    indB -= effstrideB[dim] * C->shape[dim];
                    indC -= C->stride[dim] * C->shape[dim];
                }
            }
        
        }    

        delete[] effstrideA;
    }

    // matrix multiply A and B so that C = AB
    // A and B must be 2d and width of A is equalt to height of B
    static void matmul(const T* A, const T* B, T* C, int* strideA, int* strideB, int* strideC, int* reps, int* count, size_t strideLen) {
        int k = reps[2];
        int indA = 0, indB = 0, prev_B, indC = 0;
        for (int row = 0; row < reps[0]; row++) {
            prev_B = indB;
            for (int col = 0; col < reps[1]; col++) {
                for (int i = 0; i < k; i++) {
                    C[indC] += A[indA + i*strideA[1]] * B[indB + i*strideB[0]];
                }
                indC += strideC[1];
                indB += strideB[1];
            }
            indC += strideC[0];
            indA += strideA[0];
            indB = prev_B;
        }
    }
        
    // matrix multiply A and B so that C = AB
    // A and B must be 2d and width of A is equalt to height of B
    // all dimensions higher than 2 are regarded as batch dimensions
    static void batch_matmul(const T* A, const T* B, T* C, int* strideA, int* strideB, int* strideC, int* reps, int* count, size_t strideLen) {
        int indA = 0, indB = 0, indC = 0;
        while (1) {

            C[indC] = A[indA] + B[indB];

            for (int dim = strideLen - 1; dim >= 0; dim--) {
                count[dim]--;
                if (count[dim] >= 0) {
                    indA += strideA[dim];
                    indB += strideB[dim];
                    indC += strideC[dim];
                    break;
                }

                count[dim] = reps[dim];
                if (dim == 0) goto end;
            }
        }
        end:;
    }


    static void scalarMin(const T* A, const T* B, T* C, int* strideA, int* strideB, int* strideC, int* reps, int* count, size_t strideLen) {
        T B_val = B->val[0];
        for (size_t i = 0; i < A->shapeLen; i++) {
            T A_val = A->val[i];
            C->val[i] = A_val < B_val ? A_val : B_val;
        }
    }

    static void pointMin(const T* A, const T* B, T* C, int* strideA, int* strideB, int* strideC, int* reps, int* count, size_t strideLen) {
        for (size_t i = 0; i < A->shapeLen; i++) {
            T A_val = A->val[i];
            T B_val = B->val[i];
            C->val[i] = A_val < B_val ? A_val : B_val;
        }
    }

    static void flexMin(const T* A, const T* B, T* C, int* strideA, int* strideB, int* strideC, int* reps, int* count, size_t strideLen) {
        int offsetA = C->shapeLen - A->shapeLen;
        int offsetB = C->shapeLen - B->shapeLen;
    
        int* effstrideA = new int[C->shapeLen * 3];
        int* effstrideB = effstrideA + C->shapeLen;
    
        for (size_t i = 0; i < C->shapeLen; i++) {
            int aDim = i - offsetA;
            effstrideA[i] = aDim >= 0 && A->shape[aDim] != 1 ? A->stride[aDim] : 0;
            int bDim = i - offsetB;
            effstrideB[i] = bDim >= 0 && B->shape[bDim] != 1 ? B->stride[bDim] : 0;
        }
    
        int indA = 0, indB = 0, indC = 0;
        int* cords = effstrideB + C->shapeLen;
        fill(cords, cords + C->shapeLen, 0);
        for (int i = 0; i < C->len; i++) {
        
            T A_val = A->val[indA];
            T B_val = B->val[indB];
            C->val[indC] = A_val < B_val ? A_val : B_val;
        
            for (int dim = C->shapeLen - 1; dim >= 0; dim--) {
                cords[dim]++;
                indA += effstrideA[dim];
                indB += effstrideB[dim];
                indC += C->stride[dim];
            
                if (cords[dim] < C->shape[dim]) {
                    break;
                }
                else {
                    cords[dim] = 0;
                    indA -= effstrideA[dim] * C->shape[dim];
                    indB -= effstrideB[dim] * C->shape[dim];
                    indC -= C->stride[dim] * C->shape[dim];
                }
            }
        
        }    
        delete[] effstrideA;
    }
    
    static void scalarMax(const T* A, const T* B, T* C, int* strideA, int* strideB, int* strideC, int* reps, int* count, size_t strideLen) {
        T B_val = B->val[0];
        for (size_t i = 0; i < A->shapeLen; i++) {
            T A_val = A->val[i];
            C->val[i] = A_val > B_val ? A_val : B_val;
        }
    }

    static void pointMax(const T* A, const T* B, T* C, int* strideA, int* strideB, int* strideC, int* reps, int* count, size_t strideLen) {
        for (size_t i = 0; i < A->shapeLen; i++) {
            T A_val = A->val[i];
            T B_val = B->val[i];
            C->val[i] = A_val > B_val ? A_val : B_val;
        }
    }

    static void flexMax(const T* A, const T* B, T* C, int* strideA, int* strideB, int* strideC, int* reps, int* count, size_t strideLen) {
        int offsetA = C->shapeLen - A->shapeLen;
        int offsetB = C->shapeLen - B->shapeLen;
    
        int* effstrideA = new int[C->shapeLen * 3];
        int* effstrideB = effstrideA + C->shapeLen;
    
        for (size_t i = 0; i < C->shapeLen; i++) {
            int aDim = i - offsetA;
            effstrideA[i] = aDim >= 0 && A->shape[aDim] != 1 ? A->stride[aDim] : 0;
            int bDim = i - offsetB;
            effstrideB[i] = bDim >= 0 && B->shape[bDim] != 1 ? B->stride[bDim] : 0;
        }
    
        int indA = 0, indB = 0, indC = 0;
        int* cords = effstrideB + C->shapeLen;
        fill(cords, cords + C->shapeLen, 0);
        for (int i = 0; i < C->len; i++) {
        
            T A_val = A->val[indA];
            T B_val = B->val[indB];
            C->val[indC] = A_val > B_val ? A_val : B_val;
        
            for (int dim = C->shapeLen - 1; dim >= 0; dim--) {
                cords[dim]++;
                indA += effstrideA[dim];
                indB += effstrideB[dim];
                indC += C->stride[dim];
            
                if (cords[dim] < C->shape[dim]) {
                    break;
                }
                else {
                    cords[dim] = 0;
                    indA -= effstrideA[dim] * C->shape[dim];
                    indB -= effstrideB[dim] * C->shape[dim];
                    indC -= C->stride[dim] * C->shape[dim];
                }
            }
        
        }    
        delete[] effstrideA;
    }

    // adds every element of A to out
    // B has to be a scalar
    static void sum(const tView<T>* A, const tView<T>* _, tView<T>* out, int* strideA=nullptr, int* strideB=nullptr, int* strideC=nullptr, int* reps=nullptr, int* count=nullptr, size_t strideLen=0, void* ctx=nullptr) {
        for (size_t i = 0; i < A->len; i++) {
            out->val[0] += A->val[i];
        }
    }
    
    // sums tensor along dimension
    // out must be same shape as A with one dimension missing
    // dimensions index over which is summed is saved in B.shape
    static void sum_dim(const tView<T>* A, const tView<T>* _, tView<T>* C, int* strideA=nullptr, int* strideB=nullptr, int* strideC=nullptr, int* reps=nullptr, int* count=nullptr, size_t strideLen=0, void* ctx=nullptr) {
        
        fill(C->val, C->val + C->len, 0);

        int k = *(static_cast<int*>(ctx));
        int* effstride = new int[A->shapeLen * 2];
        copy(A->stride, A->stride + A->shapeLen, effstride);
        effstride[k] = 0;
        for (int i = 0; i < k; i++) {
            effstride[i] /= A->shape[k];
        }

        int indA = 0, indC = 0; 
        int* cords = effstride + A->shapeLen;
        fill(cords, cords + A->shapeLen, 0);

        for (int i = 0; i < A->len; i++) {
            
            C->val[indC] += A->val[indA];

            for (int dim = A->shapeLen - 1; dim >= 0; dim--) {
                cords[dim]++;
                indA += A->stride[dim];
                indC += effstride[dim];

                if (cords[dim] < A->shape[dim]) {
                    break;
                }
                else {
                    cords[dim] = 0;
                    indA -= A->stride[dim] * A->shape[dim];
                    indC -= effstride[dim] * A->shape[dim];
                }
            }
        }

        delete[] effstride;
    }

    // saves mean of A into out
    // B has to be a scalar
    static void mean(const tView<T>* A, const tView<T>* _, tView<T>* out, int* strideA=nullptr, int* strideB=nullptr, int* strideC=nullptr, int* reps=nullptr, int* count=nullptr, size_t strideLen=0, void* ctx=nullptr) {
        out->val[0] = 0;
        for (size_t i = 0; i < A->len; i++) {
            out->val[0] += A->val[i];
        }
        out->val[0] /= A->len;
    }

    // computes mean of tensor along dimension
    // out must be same shape as A with one dimension missing
    // dimensions index over which is summed is saved in B.shape
    static void mean_dim(const tView<T>* A, const tView<T>* _, tView<T>* C, int* strideA=nullptr, int* strideB=nullptr, int* strideC=nullptr, int* reps=nullptr, int* count=nullptr, size_t strideLen=0, void* ctx=nullptr) {
        
        fill(C->val, C->val + C->len, 0);

        int k = *(static_cast<int*>(ctx));
        int* effstride = new int[A->shapeLen * 2];
        copy(A->stride, A->stride + A->shapeLen, effstride);
        effstride[k] = 0;
        for (int i = 0; i < k; i++) {
            effstride[i] /= A->shape[k];
        }

        int indA = 0, indC = 0; 
        int* cords = effstride + A->shapeLen;
        fill(cords, cords + A->shapeLen, 0);

        T scale = ((T)1) / A->shape[k];
        for (int i = 0; i < A->len; i++) {
            
            C->val[indC] += A->val[indA] * scale;

            for (int dim = A->shapeLen - 1; dim >= 0; dim--) {
                cords[dim]++;
                indA += A->stride[dim];
                indC += effstride[dim];

                if (cords[dim] < A->shape[dim]) {
                    break;
                }
                else {
                    cords[dim] = 0;
                    indA -= A->stride[dim] * A->shape[dim];
                    indC -= effstride[dim] * A->shape[dim];
                }
            }
        }

        delete[] effstride;
    }

    static void transpose(const T* A, const T* B, T* C, int* strideA, int* strideB, int* strideC, int* reps, int* count, size_t strideLen) {
        copy(A->val, A->val + A->len, C->val);
    }

    static void tile(const tView<T>* A, const tView<T>* _, tView<T>* C, int* strideA=nullptr, int* strideB=nullptr, int* strideC=nullptr, int* reps=nullptr, int* count=nullptr, size_t strideLen=0, void* ctx=nullptr) {
        int* m = static_cast<int*>(ctx);
        
        int offset = C->shapeLen - A->shapeLen;
        int* effstride = new int[C->shapeLen * 3];
        fill(effstride, effstride + offset, 0);
        copy(A->stride, A->stride + A->shapeLen, effstride + offset);

        int* superstride = effstride + C->shapeLen;
        copy(A->shape, A->shape + A->shapeLen, superstride);
        for (int i = A->shapeLen - 2; i >= 0; i--) {
            superstride[i] *= superstride[i + 1];
        }

        int indA = 0, indC = 0;
        int* cords = effstride + C->shapeLen * 2;
        fill(cords, cords + C->shapeLen, 0);

        for (int i = 0; i < C->len; i++) {

            C->val[indC] = A->val[indA];

            for (int dim = C->shapeLen - 1; dim >= 0; dim--) {
                cords[dim]++;
                indA += effstride[dim];
                indA -= cords[dim] == A->shape[dim] && m[dim] != 1 ? superstride[dim] : 0;
                indC += C->stride[dim];

                if (cords[dim] < C->shape[dim]) {
                    break;
                }
                else {
                    cords[dim] = 0;
                    indA -= effstride[dim] * A->shape[dim];
                    indC -= C->stride[dim] * C->shape[dim];
                }
            }
        }

        delete[] effstride;
    }

    static void slice(const tView<T>* A, const tView<T>* _, tView<T>* C, int* strideA=nullptr, int* strideB=nullptr, int* strideC=nullptr, int* reps=nullptr, int* count=nullptr, size_t strideLen=0, void* ctx=nullptr) {
        int offset = A->shapeLen - C->shapeLen;
        int* effstride = new int[C->shapeLen * 2];
        copy(A->stride + offset, A->stride + A->shapeLen - offset + 1, effstride);

        int* multiples = static_cast<int*>(ctx);
        int indA = 0, indC = 0;
        for (int i = 0; i < C->shapeLen; i++) {
            indA += multiples[i] * A->stride[i];
        }

        int* cords = effstride + C->shapeLen;
        fill(cords, cords + C->shapeLen, 0);
        for (int i = 0; i < C->len; i++) {

            C->val[indC] = A->val[indA];

            for (int dim = C->shapeLen - 1; dim >= 0; dim--) {
                cords[dim]++;
                indA += effstride[dim];
                indC += C->stride[dim];

                if (cords[dim] < C->shape[dim]) {
                    break;
                }
                else {
                    cords[dim] = 0;
                    indA -= effstride[dim] * C->shape[dim];
                    indC -= C->stride[dim] * C->shape[dim];
                }
            }
        }

        delete[] effstride;
    }
};
