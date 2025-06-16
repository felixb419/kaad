#pragma once

#include "tensor.h"

#include <cmath>
#include <sstream>
#include <stdexcept>
#include <vector>

using namespace std;


template <typename T>
using tensorOp = void(*)(const T* A, const T* B, T* C, int* strideA, int* strideB, int* strideC, int* reps, int* count, size_t strideLen);

template <typename T>
struct Operations {
    /*
    BINARY OPS
    */

    // add so that: C[m,n,...] = A[m,n,...] + B[0]
    // shapes of C and A must be the same, shape of B must be (1)
    static void scalarAddRt(const T* A, const T* B, T* C, int* strideA, int* strideB, int* strideC, int* reps, int* count, size_t strideLen) {
        for (size_t i = 0; i < strideLen; i++) {
            C[i] = A[i] + B[0];
        }
    }
    // add so that: C[m,n,...] = A[0] + B[m,n,...]}
    // shapes of out and tensor must be the same, shape of scalar must be (1)
    static void scalarAddLt(const T* A, const T* B, T* C, int* strideA, int* strideB, int* strideC, int* reps, int* count, size_t strideLen) {
        for (size_t i = 0; i < strideLen; i++) {
            C[i] = A[0] + B[i];
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
        
    // subtract so that: C[m,n,...] = A[m,n,...] - B[0]
    // shapes of C and A must be the same, shape of B must be (1)
    static void scalarSubRt(const T* A, const T* B, T* C, int* strideA, int* strideB, int* strideC, int* reps, int* count, size_t strideLen) {
        for (size_t i = 0; i < strideLen; i++) {
            C[i] = A[i] - B[0];
        }
    }
    // subtract so that: C[m,n,...] = A[0] - B[m,n,...]}
    // shapes of out and tensor must be the same, shape of scalar must be (1)
    static void scalarSubLt(const T* A, const T* B, T* C, int* strideA, int* strideB, int* strideC, int* reps, int* count, size_t strideLen) {
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
        
    // multiply so that: C[m,n,...] = A[m,n,...] * B[0]
    // shapes of C and A must be the same, shape of B must be (1)
    static void scalarMulRt(const T* A, const T* B, T* C, int* strideA, int* strideB, int* strideC, int* reps, int* count, size_t strideLen) {
        for (size_t i = 0; i < strideLen; i++) {
            C[i] = A[i] * B[0];
        }
    }
    // multiply so that: C[m,n,...] = A[0] * B[m,n,...]}
    // shapes of out and tensor must be the same, shape of scalar must be (1)
    static void scalarMulLt(const T* A, const T* B, T* C, int* strideA, int* strideB, int* strideC, int* reps, int* count, size_t strideLen) {
        for (size_t i = 0; i < strideLen; i++) {
            C[i] = A[0] * B[i]; 
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

    // divide so that: C[m,n,...] = A[m,n,...] / B[0]
    // shapes of C and A must be the same, shape of B must be (1)
    static void scalarDivRt(const T* A, const T* B, T* C, int* strideA, int* strideB, int* strideC, int* reps, int* count, size_t strideLen) {
        for (size_t i = 0; i < strideLen; i++) {
            C[i] = A[i] / B[0];
        }
    }
    // divide so that: C[m,n,...] = A[0] / B[m,n,...]
    // shapes of C and B must be the same, shape of A must be (1)
    static void scalarDivLt(const T* A, const T* B, T* C, int* strideA, int* strideB, int* strideC, int* reps, int* count, size_t strideLen) {
        for (size_t i = 0; i < strideLen; i++) {
            C[i] = A[0] / B[i]; 
        }
    }
    // pointwise divide so that: C = A / B
    // shape of all operands must be indentical
    static void pointDiv(const T* A, const T* B, T* C, int* strideA, int* strideB, int* strideC, int* reps, int* count, size_t strideLen) {
        for (size_t i = 0; i < strideLen; i++) {
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
    static void scalarPowRt(const T* A, const T* B, T* C, int* strideA, int* strideB, int* strideC, int* reps, int* count, size_t strideLen) {
        for (size_t i = 0; i < strideLen; i++) {
            C[i] = pow(A[i], B[0]);
        }
    }
    // raise to power so that: C[m,n,...] = A[0] ^ B[m,n,...]
    // shapes of C and B must be the same, shape of A must be (1)
    static void scalarPowLt(const T* A, const T* B, T* C, int* strideA, int* strideB, int* strideC, int* reps, int* count, size_t strideLen) {
        for (size_t i = 0; i < strideLen; i++) {
            C[i] = pow(A[0], B[i]);
        }
    }
    // pointwise raise to power so that: C = A ^ B
    // shape of all operands must be indentical
    static void pointPow(const T* A, const T* B, T* C, int* strideA, int* strideB, int* strideC, int* reps, int* count, size_t strideLen) {
        for (size_t i = 0; i < strideLen; i++) {
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
    static void dot(const T* A, const T* B, T* C, int* strideA, int* strideB, int* strideC, int* reps, int* count, size_t strideLen) {
        for (size_t i = 0; i < strideLen; i++) {
            C[0] += A[i] * B[i]; 
        }
    }
    // compute do product of A and B into C
    // A must be 1d vector, B and C must be scalar
    static void scalarDot(const T* A, const T* B, T* C, int* strideA, int* strideB, int* strideC, int* reps, int* count, size_t strideLen) {
        for (size_t i = 0; i < strideLen; i++) {
            C[0] += A[i] * B[0]; 
        }
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
        int k = reps[strideLen], a_offset = strideA[strideLen], b_offset = strideB[strideLen];
        int indA = 0, indB = 0, indC = 0;
        while (1) {

            for (int i = 0; i < k; i++) {
                C[indC] += A[indA + i*a_offset] * B[indB + i*b_offset];
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

    /*
    UNARY OPS
    */

    // negate A so that C[i] = -A[i]
    // A and C must have same shape
    static void negate(const T* A, const T* _, T* C, int* strideA, int* strideB, int* strideC, int* reps, int* count, size_t strideLen) {
        for (size_t i = 0; i < strideLen; i++) {
            C[i] = -A[i];
        }
    }
        
    // square A so that out[i] = A[i]*A[i]
    // A and out must have same shape
    static void square(const T* A, const T* _, T* C, int* strideA, int* strideB, int* strideC, int* reps, int* count, size_t strideLen) {
        for (size_t i = 0; i < strideLen; i++) {
            C[i] = A[i] * A[i];
        }
    }
        
    // root of A so that C[i] = sqrt(A[i])
    // A and C must have same shape
    static void sqrt(const T* A, const T* _, T* C, int* strideA, int* strideB, int* strideC, int* reps, int* count, size_t strideLen) {
        for (size_t i = 0; i < strideLen; i++) {
            C[i] = std::sqrt(A[i]);
        }
    }
        
    // log of A so that C[i] = ln(A[i])
    // A and C must have same shape
    static void log(const T* A, const T* _, T* C, int* strideA, int* strideB, int* strideC, int* reps, int* count, size_t strideLen) {
        for (size_t i = 0; i < strideLen; i++) {
            C[i] = std::log(A[i]);
        }
    }
        
    // exponent of A so that C[i] = e^A[i]
    // A and C must have same shape
    static void exp(const T* A, const T* _, T* C, int* strideA, int* strideB, int* strideC, int* reps, int* count, size_t strideLen) {
        for (size_t i = 0; i < strideLen; i++) {
            C[i] = std::exp(A[i]);
        }
    }
        
    // absolute value of A so that C[i] = abs(A[i])
    // A and C must have same shape
    static void abs(const T* A, const T* _, T* C, int* strideA, int* strideB, int* strideC, int* reps, int* count, size_t strideLen) {
        for (size_t i = 0; i < strideLen; i++) {
            C[i] = std::abs(A[i]);
        }
    }

    // transposing doesnt change the value array so A gets copied to C
    static void transpose(const T* A, const T* B, T* C, int* strideA, int* strideB, int* strideC, int* reps, int* count, size_t strideLen) {
        copy(A, A + strideLen, C);
    }

    // adds every element of A to out
    // B has to be a scalar
    static void sum(const T* A, const T* B, T* C, int* strideA, int* strideB, int* strideC, int* reps, int* count, size_t strideLen) {
        for (size_t i = 0; i < strideLen; i++) {
            C[0] += A[i];
        }
    }
    // sums tensor along dimension
    // out must be same shape as A with one dimension missing
    // dimensions index over which is summed is saved in B.shape
    static void sum_dim(const T* A, const T* B, T* C, int* strideA, int* strideB, int* strideC, int* reps, int* count, size_t strideLen) {
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
    static void mean(const T* A, const T* B, T* C, int* strideA, int* strideB, int* strideC, int* reps, int* count, size_t strideLen) {
        for (size_t i = 0; i < strideLen; i++) {
            C[0] += A[i];
        }
        C[0] /= strideLen;
    }

    // computes mean of tensor along dimension
    // out must be same shape as A with one dimension missing
    // dimensions index over which is summed is saved in B.shape
    static void mean_dim(const T* A, const T* B, T* C, int* strideA, int* strideB, int* strideC, int* reps, int* count, size_t strideLen) {
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
