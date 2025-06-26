#pragma once

#include "tensor.h"

#include <cmath>
#include <sstream>
#include <stdexcept>
#include <vector>

using namespace std;


template <typename T>
using unaryOp = void(*)(const T* A, T* C, size_t len);
template <typename T>
using binaryOp = void(*)(const T* A, const T* B, T* C, size_t len);
template <typename T>
using flexUnaryOp = void(*)(const T* A, T* C, int* strideA, int* strideC, int* reps, int* count, size_t strideLen);
template <typename T>
using flexBinaryOp = void(*)(const T* A, const T* B, T* C, int* strideA, int* strideB, int* strideC, int* reps, int* count, size_t strideLen);
template <typename T>
using matmulOp = void(*)(const T* A, const T* B, T* C, int a_dim, int b_dim, int k, int* strideA, int* strideB, int* strideC);
template <typename T>
using batchmatmulOp = void(*)(const T* A, const T* B, T* C, int a_off, int b_off, int k, int* strideA, int* strideB, int* strideC, int* reps, int* count, size_t strideLen);

template <typename T>
struct Operations {
    /*
    BINARY OPS
    */

    // add so that: C[m,n,...] = A[m,n,...] + B[0]
    // shapes of C and A must be the same, shape of B must be (1)
    static void scalarAddRt(const T* A, const T* B, T* C, size_t len) {
        for (size_t i = 0; i < len; i++) {
            C[i] = A[i] + B[0];
        }
    }
    // add so that: C[m,n,...] = A[0] + B[m,n,...]}
    // shapes of out and tensor must be the same, shape of scalar must be (1)
    static void scalarAddLt(const T* A, const T* B, T* C, size_t len) {
        for (size_t i = 0; i < len; i++) {
            C[i] = A[0] + B[i];
        }
    }
    // pointwise add so that: C = A + B
    // shape of all operands must be indentical
    static void pointAdd(const T* A, const T* B, T* C, size_t len) {
        for (size_t i = 0; i < len; i++) {
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
        
    // subtract so that: C[m,n,...] = A[m,n,...] - B[0]
    // shapes of C and A must be the same, shape of B must be (1)
    static void scalarSubRt(const T* A, const T* B, T* C, size_t len) {
        for (size_t i = 0; i < len; i++) {
            C[i] = A[i] - B[0];
        }
    }
    // subtract so that: C[m,n,...] = A[0] - B[m,n,...]}
    // shapes of out and tensor must be the same, shape of scalar must be (1)
    static void scalarSubLt(const T* A, const T* B, T* C, size_t len) {
        for (size_t i = 0; i < len; i++) {
            C[i] = A[0] - B[i]; 
        }
    }
    // pointwise subtract so that: C = A - B
    // shape of all operands must be indentical
    static void pointSub(const T* A, const T* B, T* C, size_t len) {
        for (size_t i = 0; i < len; i++) {
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
        
    // multiply so that: C[m,n,...] = A[m,n,...] * B[0]
    // shapes of C and A must be the same, shape of B must be (1)
    static void scalarMulRt(const T* A, const T* B, T* C, size_t len) {
        for (size_t i = 0; i < len; i++) {
            C[i] = A[i] * B[0];
        }
    }
    // multiply so that: C[m,n,...] = A[0] * B[m,n,...]}
    // shapes of out and tensor must be the same, shape of scalar must be (1)
    static void scalarMulLt(const T* A, const T* B, T* C, size_t len) {
        for (size_t i = 0; i < len; i++) {
            C[i] = A[0] * B[i]; 
        }
    }
    // pointwise multiply so that: C = A * B
    // shape of all operands must be indentical
    static void pointMul(const T* A, const T* B, T* C, size_t len) {
        for (size_t i = 0; i < len; i++) {
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

    // divide so that: C[m,n,...] = A[m,n,...] / B[0]
    // shapes of C and A must be the same, shape of B must be (1)
    static void scalarDivRt(const T* A, const T* B, T* C, size_t len) {
        for (size_t i = 0; i < len; i++) {
            C[i] = A[i] / B[0];
        }
    }
    // divide so that: C[m,n,...] = A[0] / B[m,n,...]
    // shapes of C and B must be the same, shape of A must be (1)
    static void scalarDivLt(const T* A, const T* B, T* C, size_t len) {
        for (size_t i = 0; i < len; i++) {
            C[i] = A[0] / B[i]; 
        }
    }
    // pointwise divide so that: C = A / B
    // shape of all operands must be indentical
    static void pointDiv(const T* A, const T* B, T* C, size_t len) {
        for (size_t i = 0; i < len; i++) {
            C[i] = A[i] / B[i];
        }
    }
    // flexible divide so that: C = A / B
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

    // raise to power so that: C[m,n,...] = A[m,n,...] ^ B[0]
    // shapes of C and A must be the same, shape of B must be (1)
    static void scalarPowRt(const T* A, const T* B, T* C, size_t len) {
        for (size_t i = 0; i < len; i++) {
            C[i] = pow(A[i], B[0]);
        }
    }
    // raise to power so that: C[m,n,...] = A[0] ^ B[m,n,...]
    // shapes of C and B must be the same, shape of A must be (1)
    static void scalarPowLt(const T* A, const T* B, T* C, size_t len) {
        for (size_t i = 0; i < len; i++) {
            C[i] = pow(A[0], B[i]);
        }
    }
    // pointwise raise to power so that: C = A ^ B
    // shape of all operands must be indentical
    static void pointPow(const T* A, const T* B, T* C, size_t len) {
        for (size_t i = 0; i < len; i++) {
            C[i] = pow(A[i], B[i]);
        }
    }
    // flexible raise to power so that: C = A ^ B
    // shape of C must be a valid broadcast of A and B
    static void flexPow(const T* A, const T* B, T* C, int* strideA, int* strideB, int* strideC, int* reps, int* count, size_t strideLen) {
        int indA = 0, indB = 0, indC = 0;
        while (1) {

            C[indC] = pow(A[indA], B[indB]);

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

    // compute do product of A and B into C
    // A and B must be 1d vectors of same length, C must be scalar
    static void dot(const T* A, const T* B, T* C, size_t len) {
        for (size_t i = 0; i < len; i++) {
            C[0] += A[i] * B[i]; 
        }
    }
    // compute do product of A and B into C
    // A must be 1d vector, B and C must be scalar
    static void scalarDot(const T* A, const T* B, T* C, size_t len) {
        for (size_t i = 0; i < len; i++) {
            C[0] += A[i] * B[0]; 
        }
    }

    // matrix multiply A and B so that C = AB
    // A and B must be 2d and width of A is equalt to height of B
    static void matmul(const T* A, const T* B, T* C, int a_dim, int b_dim, int k, int* strideA, int* strideB, int* strideC) {
        const T* pa = A;
        const T* pb = B;
        T* pc = C;
        const T* _pa;
        const T* _pb;
        const T* __pb;
        for (int a_idx = 0; a_idx < a_dim; a_idx++) {
            _pb = pb;
            for (int b_idx = 0; b_idx < b_dim; b_idx++) {
                _pa = pa;
                __pb = _pb;
                for (int i = 0; i < k; i++) {
                    *pc += (*_pa) * (*__pb);

                   _pa += strideA[1];
                   __pb += strideB[0];
                }
                _pb += strideB[1];
                pc += strideC[1];
            }
            pa += strideA[0];
            pc += strideC[0];
        }
    }
        
    // matrix multiply A and B so that C = AB
    // last two dimensions of A and B must me matrix multipliable
    // all dimensions higher than 2 are regarded as batch dimensions
    static void batch_matmul(const T* A, const T* B, T* C, int a_off, int b_off, int k, int* strideA, int* strideB, int* strideC, int* reps, int* count, size_t strideLen) {
        int indA = 0, indB = 0, indC = 0;
        while (1) {

            for (int i = 0; i < k; i++) {
                C[indC] += A[indA + i*a_off] * B[indB + i*b_off];
            }

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

    // take minimum of A and B so that C[i] = min(A[i], B[0])
    // B must be scalar
    static void scalarMinRt(const T* A, const T* B, T* C, size_t len) {
        T B_val = B[0];
        for (size_t i = 0; i < len; i++) {
            T A_val = A[i];
            C[i] = A_val < B_val ? A_val : B_val;
        }
    }
    // take minimum of A and B so that C[i] = min(A[0], B[i])
    // A must be scalar
    static void scalarMinLt(const T* A, const T* B, T* C, size_t len) {
        T A_val = A[0];
        for (size_t i = 0; i < len; i++) {
            T B_val = B[i];
            C[i] = A_val < B_val ? A_val : B_val;
        }
    }
    // take pointwise minimum of A and B so that C[i] = min(A[i], B[i])
    // shape of all operands must be indentical
    static void pointMin(const T* A, const T* B, T* C, size_t len) {
        for (size_t i = 0; i < len; i++) {
            T A_val = A[i];
            T B_val = B[i];
            C[i] = A_val < B_val ? A_val : B_val;
        }
    }
    // take flexible minimum of A and B so that C[i] = min(A[i], B[i])
    // shape of C must be a valid broadcast of A and B
    static void flexMin(const T* A, const T* B, T* C, int* strideA, int* strideB, int* strideC, int* reps, int* count, size_t strideLen) {
        int indA = 0, indB = 0, indC = 0;
        while (1) {

            T A_val = A[indA];
            T B_val = B[indB];
            C[indC] = A_val < B_val ? A_val : B_val;

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

    // take maximum of A and B so that C[i] = max(A[i], B[0])
    // B must be scalar
    static void scalarMaxRt(const T* A, const T* B, T* C, size_t len) {
        T B_val = B[0];
        for (size_t i = 0; i < len; i++) {
            T A_val = A[i];
            C[i] = A_val > B_val ? A_val : B_val;
        }
    }
    // take maximum of A and B so that C[i] = max(A[0], B[i])
    // A must be scalar
    static void scalarMaxLt(const T* A, const T* B, T* C, size_t len) {
        T A_val = A[0];
        for (size_t i = 0; i < len; i++) {
            T B_val = B[i];
            C[i] = A_val > B_val ? A_val : B_val;
        }
    }
    // take pointwise maximum of A and B so that C[i] = max(A[i], B[i])
    // shape of all operands must be indentical
    static void pointMax(const T* A, const T* B, T* C, size_t len) {
        for (size_t i = 0; i < len; i++) {
            T A_val = A[i];
            T B_val = B[i];
            C[i] = A_val > B_val ? A_val : B_val;
        }
    }
    // take flexible minimum of A and B so that C[i] = min(A[i], B[i])
    // shape of C must be a valid broadcast of A and B
    static void flexMax(const T* A, const T* B, T* C, int* strideA, int* strideB, int* strideC, int* reps, int* count, size_t strideLen) {
        int indA = 0, indB = 0, indC = 0;
        while (1) {

            T A_val = A[indA];
            T B_val = B[indB];
            C[indC] = A_val > B_val ? A_val : B_val;

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
    
    /*
    UNARY OPS
    */

    // negate A so that C[i] = -A[i]
    // A and C must have same shape
    static void negate(const T* A, T* C, size_t len) {
        for (size_t i = 0; i < len; i++) {
            C[i] = -A[i];
        }
    }
        
    // square A so that out[i] = A[i]*A[i]
    // A and out must have same shape
    static void square(const T* A, T* C, size_t len) {
        for (size_t i = 0; i < len; i++) {
            C[i] = A[i] * A[i];
        }
    }
        
    // root of A so that C[i] = sqrt(A[i])
    // A and C must have same shape
    static void sqrt(const T* A, T* C, size_t len) {
        for (size_t i = 0; i < len; i++) {
            C[i] = std::sqrt(A[i]);
        }
    }
        
    // log of A so that C[i] = ln(A[i])
    // A and C must have same shape
    static void log(const T* A, const T* _, T* C, size_t len) {
        for (size_t i = 0; i < len; i++) {
            C[i] = std::log(A[i]);
        }
    }
        
    // exponent of A so that C[i] = e^A[i]
    // A and C must have same shape
    static void exp(const T* A, T* C, size_t len) {
        for (size_t i = 0; i < len; i++) {
            C[i] = std::exp(A[i]);
        }
    }
        
    // absolute value of A so that C[i] = abs(A[i])
    // A and C must have same shape
    static void abs(const T* A, T* C, size_t len) {
        for (size_t i = 0; i < len; i++) {
            C[i] = std::abs(A[i]);
        }
    }

    // transposing doesnt change the value array so A gets copied to C
    static void transpose(const T* A, T* C, size_t len) {
        copy(A, A + len, C);
    }

    // adds every element of A to out
    // B has to be a scalar
    static void sum(const T* A, T* C, size_t len) {
        for (size_t i = 0; i < len; i++) {
            C[0] += A[i];
        }
    }
    // sums tensor along dimension
    // out must be same shape as A with one dimension missing
    // dimensions index over which is summed is saved in B.shape
    static void sum_dim(const T* A, T* C, int* strideA, int* strideC, int* reps, int* count, size_t strideLen) {
        int indA = 0, indC = 0;
        while (1) {

            C[indC] += A[indA];

            for (int dim = strideLen - 1; dim >= 0; dim--) {
                count[dim]--;
                if (count[dim] >= 0) {
                    indA += strideA[dim];
                    indC += strideC[dim];
                    break;
                }

                count[dim] = reps[dim];
                if (dim == 0) goto end;
            }
        }
        end:;
    }

    // saves mean of A into out
    // B has to be a scalar
    static void mean(const T* A, T* C, size_t len) {
        for (size_t i = 0; i < len; i++) {
            C[0] += A[i];
        }
        C[0] /= len;
    }

    // computes mean of tensor along dimension
    // out must be same shape as A with one dimension missing
    // dimensions index over which is summed is saved in B.shape
    static void mean_dim(const T* A, T* C, int* strideA, int* strideC, int* reps, int* count, size_t strideLen) {
        int indA = 0, indC = 0;
        while (1) {

            C[indC] += A[indA];

            for (int dim = strideLen - 1; dim >= 0; dim--) {
                count[dim]--;
                if (count[dim] >= 0) {
                    indA += strideA[dim];
                    indC += strideC[dim];
                    break;
                }

                count[dim] = reps[dim];
                if (dim == 0) goto end;
            }
        }
        end:;

        T divisor = reps[strideLen];
        for (size_t i = 0; i < reps[strideLen + 1]; i++) {
            C[i] /= divisor;
        }
    }

    /*
    UNCATEGORIZED
    */
    /*
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
    */
};
