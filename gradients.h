#pragma once

#include "tensor.h"
#include "operations.h"

#include <unordered_map>
#include <vector>

#define safer_powergradient

template <typename T, class Grad>
using unaryGrad = void(*)(const T* A, T* dA, const T* C, const T* dC, size_t len, Grad grad);
template <typename T, class Grad>
using binaryGrad = void(*)(const T* A, T* dA, const T* B, T* dB, const T* C, const T* dC, size_t len, Grad grad);
template <typename T, class Grad>
using flexUnaryGrad = void(*)(const T* A, T* dA, const T* C, const T* dC, int* strideA, int* strideC, int* reps, int* count, size_t strideLen, Grad grad);
template <typename T, class Grad>
using flexBinaryGrad = void(*)(const T* A, T* dA, const T* B, T* dB, const T* C, const T* dC, int* strideA, int* strideB, int* strideC, int* reps, int* count, size_t strideLen, Grad grad);
template <typename T>
using matmulGrad = void(*)(const T* A, T* dA, const T* B, T* dB, const T* C, const T* dC, int* a_dim, int* b_dim, int* k, int* strideA, int* strideB, int* strideC);
template <typename T>
using batchmatmulGrad = void(*)(const T* A, T* dA, const T* B, T* dB, const T* C, const T* dC, int* a_off, int* b_off, int* k, int** strideA, int** strideB, int** strideC, int** reps, int** count, size_t* strideLen);
template <typename T>
using meanDimGrad = void(*)(const T* A, T* dA, const T* C, const T* dC, T divisor, size_t c_len, int* strideA, int* strideC, int* reps, int* count, size_t strideLen);

template <typename T, class Grad>
struct Gradients {

    //d/dx[ f(g(x,...)) ] = f'(g(x,...)) * g'(x,...)

    /*
    BINARY GRADS
    */
    static void scalarRhs(const T* A, T* dA, const T* B, T* dB, const T* C, const T* dC, size_t len, Grad grad) {
        // B has shape = (1,)
        for (size_t i = 0; i < len; i++) {
            grad(A[i], dA[i], B[0], dB[0], C[i], dC[i]);
        }
    }
    static void scalarLhs(const T* A, T* dA, const T* B, T* dB, const T* C, const T* dC, size_t len, Grad grad) {
        // A has shape = (1,)
        for (size_t i = 0; i < len; i++) {
            grad(A[0], dA[0], B[i], dB[i], C[i], dC[i]);
        }
    }
    static void pointwise(const T* A, T* dA, const T* B, T* dB, const T* C, const T* dC, size_t len, Grad grad) {
        for (size_t i = 0; i < len; i++) {
            grad(A[i], dA[i], B[i], dB[i], C[i], dC[i]);
        }
    }
    static void flexible(const T* A, T* dA, const T* B, T* dB, const T* C, const T* dC, int* strideA, int* strideB, int* strideC, int* reps, int* count, size_t strideLen, Grad grad) {
        int indA = 0, indB = 0, indC = 0;
        while (1) {

            grad(A[indA], dA[indA], B[indB], dB[indB], C[indC], dC[indC]);

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

    // f(A,B) = A dot B
    // df/dA = B
    // df/dB = A
    static void dot_grad(const T* A, T* dA, const T* B, T* dB, const T* C, const T* dC, size_t len, Grad _) {
        for (size_t i = 0; i < len; i++) {
            dA[i] += dC[0] * B[i];
            dB[i] += dC[0] * A[i];
        }
    }
    static void scalarDot_grad(const T* A, T* dA, const T* B, T* dB, const T* C, const T* dC, size_t len, Grad _) {
        for (size_t i = 0; i < len; i++) {
            dA[i] += dC[0] * B[0];
            dB[0] += dC[0] * A[i];
        }
    }

    // f(A,B) = AB
    // dC/dA = B^T
    // dC/dB = A^T
    static void matmul_grad(const T* A, T* dA, const T* B, T* dB, const T* C, const T* dC, int* a_dim, int* b_dim, int* k, int* strideA, int* strideB, int* strideC) {
        // dA = dC * B^T
        Operations<T,nullptr_t>::matmul(dC, B, dA, a_dim[0], b_dim[0], k[0], strideC, strideB, strideA);
        // dB = A^T * dC
        Operations<T,nullptr_t>::matmul(A, dC, dB, a_dim[1], b_dim[1], k[1], strideA+2, strideC+2, strideB+2);
    }
    static void batch_matmul_grad(const T* A, T* dA, const T* B, T* dB, const T* C, const T* dC, int* a_off, int* b_off, int* k, int** strideA, int** strideB, int** strideC, int** reps, int** count, size_t* strideLen) {
        // dA = dC * B^T
        Operations<T,nullptr_t>::batch_matmul(dC, B, dA, a_off[0], b_off[0], k[0], strideC[0], strideB[0], strideA[0], reps[0], count[0], strideLen[0]);
        // dB = A^T * dC
        Operations<T,nullptr_t>::batch_matmul(A, dC, dB, a_off[1], b_off[1], k[1], strideA[1], strideC[1], strideB[1], reps[1], count[1], strideLen[1]);
    }

    /*
    UNARY GRADS
    */

    static void unary_pointwise(const T* A, T* dA, const T* C, const T* dC, size_t len, Grad grad) {
        for (size_t i = 0; i < len; i++) {
            grad(A[i], dA[i], C[i], dC[i]);
        }
    }

    static void unary_flexible(const T* A, T* dA, const T* C, const T* dC, int* strideA, int* strideC, int* reps, int* count, size_t strideLen, Grad grad) {
        int indA = 0, indC = 0;
        while (1) {

            grad(A[indA], dA[indA], C[indC], dC[indC]);

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

    // f(A) = A^T
    // df/dA = 1
    static void transp_grad(const T* A, T* dA, const T* C, const T* dC, size_t len) {
        for (size_t i = 0; i < len; i++) {
            dA[i] += dC[i];
        }
    }

    // f(A) = sum(A)
    // df_dA = tensor with shape of A filled with 1
    static void sum_grad(const T* A, T* dA, const T* C, const T* dC, size_t len, Grad _) {
        for (size_t i = 0; i < len; i++) {
            dA[i] += dC[0];
        }
    }

    // f(A) = mean(A)
    // df_dA = tensor with shape of A filled with 1 / (len of A)
    static void mean_grad(const T* A, T* dA, const T* C, const T* dC, size_t len, Grad _) {
        T mean = dC[0] / len;
        for (size_t i = 0; i < len; i++) {
            dA[i] += mean;
        }
    }

    static void mean_dim_grad(const T* A, T* dA, const T* C, const T* dC, T divisor, size_t c_len, int* strideA, int* strideC, int* reps, int* count, size_t strideLen) {
        Operations<T,nullptr_t>::mean_dim(dC, dA, divisor, c_len, strideC, strideA, reps, count, strideLen);
    }
};
