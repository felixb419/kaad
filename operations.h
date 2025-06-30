#pragma once

#include "tensor.h"

#include <cmath>
#include <sstream>
#include <stdexcept>
#include <vector>

using namespace std;


template <typename T>
using unaryOp = void(*)(const T* A, T* C, size_t len);
template <typename T, class Op>
using binaryOp = void(*)(const T* A, const T* B, T* C, size_t len, Op op);
template <typename T>
using flexUnaryOp = void(*)(const T* A, T* C, int* strideA, int* strideC, int* reps, int* count, size_t strideLen);
template <typename T, class Op>
using flexBinaryOp = void(*)(const T* A, const T* B, T* C, int* strideA, int* strideB, int* strideC, int* reps, int* count, size_t strideLen, Op op);
template <typename T>
using matmulOp = void(*)(const T* A, const T* B, T* C, int a_dim, int b_dim, int k, int* strideA, int* strideB, int* strideC);
template <typename T>
using batchmatmulOp = void(*)(const T* A, const T* B, T* C, int a_off, int b_off, int k, int* strideA, int* strideB, int* strideC, int* reps, int* count, size_t strideLen);
template <typename T>
using meanDimOp = void(*)(const T* A, T* C, T divisor, size_t c_len, int* strideA, int* strideC, int* reps, int* count, size_t strideLen);

template <typename T, class Op>
struct Operations {
    /*
    BINARY OPS
    */

    // perform op so that: C[m,n,...] = op( A[m,n,...], B[0] )
    // shapes of C and A must be the same, shape of B must be (1)
    static void scalarRhs(const T* A, const T* B, T* C, size_t len, Op op) {
        for (size_t i = 0; i < len; i++) {
            op(A[i], B[0], C[i]);
        }
    }
    // perform op so that: C[m,n,...] = op( A[0], B[m,n,...])
    // shapes of out and tensor must be the same, shape of scalar must be (1)
    static void scalarLhs(const T* A, const T* B, T* C, size_t len, Op op) {
        for (size_t i = 0; i < len; i++) {
            op(A[0], B[i], C[i]);
        }
    }
    // perform op so that so that: C[m,n,...] = op( A[m,n,...], B[m,n...] )
    // shape of all operands must be indentical
    static void pointwise(const T* A, const T* B, T* C, size_t len, Op op) {
        for (size_t i = 0; i < len; i++) {
            op(A[i], B[i], C[i]);
        }
    }
    // perform op flexible so that: C = op( A, B )
    // shape of C must be a valid broadcast of A and B
    static void flexible(const T* A, const T* B, T* C, int* strideA, int* strideB, int* strideC, int* reps, int* count, size_t strideLen, Op op) {
        int indA = 0, indB = 0, indC = 0;
        while (1) {

            op(A[indA], B[indB], C[indC]);

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
    // A must be 1d vector, B and C must be scalar
    static void scalarDot(const T* A, const T* B, T* C, size_t len, Op _) {
        for (size_t i = 0; i < len; i++) {
            C[0] += A[i] * B[0]; 
        }
    }
    // compute do product of A and B into C
    // A and B must be 1d vectors of same length, C must be scalar
    static void dot(const T* A, const T* B, T* C, size_t len, Op _) {
        for (size_t i = 0; i < len; i++) {
            C[0] += A[i] * B[i]; 
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
    static void log(const T* A, T* C, size_t len) {
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
    static void mean_dim(const T* A, T* C, T divisor, size_t c_len, int* strideA, int* strideC, int* reps, int* count, size_t strideLen) {
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

        for (T* p = C; p != C + c_len; p++) {
            *p /= divisor;
        }
    }
};
