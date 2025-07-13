#pragma once

#include <cstddef>       // for size_t, std::nullptr_t
#include "operations.h"  // for Operations

namespace kaad {
    template <typename T, class Grad>
    using unaryGrad = void(*)(const T* A, T* dA, const T* C, const T* dC, size_t len, Grad grad);
    template <typename T, class Grad>
    using binaryGrad = void(*)(const T* A, T* dA, const T* B, T* dB, const T* C, const T* dC, size_t len, Grad grad);
    template <typename T, class Grad>
    using flexUnaryGrad = void(*)(const T* A, T* dA, const T* C, const T* dC, int* strideA, int* strideC, int* reps, int* count, size_t D, Grad grad);
    template <typename T, class Grad>
    using flexBinaryGrad = void(*)(const T* A, T* dA, const T* B, T* dB, const T* C, const T* dC, int* strideA, int* strideB, int* strideC, int* len, int N, Grad grad);
    template <typename T>
    using matmulGrad = void(*)(const T* A, T* dA, const T* B, T* dB, const T* C, const T* dC, int* a_dim, int* b_dim, int* k, int* strideA, int* strideB, int* strideC);
    template <typename T>
    using batchmatmulGrad = void(*)(const T* A, T* dA, const T* B, T* dB, const T* C, const T* dC, int* a_off, int* b_off, int* k, int** strideA, int** strideB, int** strideC, int** reps, int** count, size_t* D);
    template <typename T>
    using meanDimGrad = void(*)(const T* A, T* dA, const T* C, const T* dC, T divisor, size_t c_len, int* strideA, int* strideC, int* reps, int* count, size_t D);

    namespace Gradients {

        // d/dx[ f(g(x,...)) ] = f'(g(x,...)) * g'(x,...)

        /*
        BINARY GRADS
        */
        template <typename T, class Grad>
        static void scalarRhs(const T* A, T* dA, const T* B, T* dB, const T* C, const T* dC, size_t len, Grad grad) {
            const T* end = C + len;
            for (; C != end; A++, dA++, C++, dC++) {
                grad(*A, *dA, *B, *dB, *C, *dC);
            }
        }
        template <typename T, class Grad>
        static void scalarLhs(const T* A, T* dA, const T* B, T* dB, const T* C, const T* dC, size_t len, Grad grad) {
            const T* end = C + len;
            for (; C != end; B++, dB++, C++, dC++) {
                grad(*A, *dA, *B, *dB, *C, *dC);
            }
        }
        template <typename T, class Grad>
        static void pointwise(const T* A, T* dA, const T* B, T* dB, const T* C, const T* dC, size_t len, Grad grad) {
            const T* end = C + len;
            for (; C != end; A++, dA++, B++, dB++, C++, dC++) {
                grad(*A, *dA, *B, *dB, *C, *dC);
            }
        }
        template <typename T, class Grad>
        static void flexible(const T* A, T* dA, const T* B, T* dB, const T* C, const T* dC, int* strideA, int* strideB, int* strideC, int* len, int N, Grad grad) {

            const T* end = C + (*len) * (*strideC);
            if (N == 0) {
                for (; C != end;
                    A += *strideA, B += *strideB, C += *strideC,
                    dA += *strideA, dB += *strideB, dC += *strideC) {
                    grad(*A, *dA, *B, *dB, *C, *dC);
                }
            }
            else {
                for (; C < end;
                    A += *strideA, B += *strideB, C += *strideC,
                    dA += *strideA, dB += *strideB, dC += *strideC) {
                    flexible(A, dA, B, dB, C, dC, strideA+1, strideB+1, strideC+1, len+1, N-1, grad);
                }
            }
        }

        // f(A,B) = A dot B
        // df/dA = B
        // df/dB = A
        template <typename T>
        static void dot_grad(const T* A, T* dA, const T* B, T* dB, const T* C, const T* dC, size_t len) {
            const T* end = A + len;
            for (; A != end; A++, dA++, B++, dB++) {
                *dA += *dC * (*B);
                *dB += *dC * (*A);
            }
        }
        template <typename T, class Grad>
        static void scalarDot_grad(const T* A, T* dA, const T* B, T* dB, const T* C, const T* dC, size_t len, Grad _) {
            const T* end = A + len;
            for (; A != end; A++, dA++) {
                *dA += *dC * (*B);
                *dB += *dC * (*A);
            }
        }

        // f(A,B) = AB
        // dC/dA = B^T
        // dC/dB = A^T
        template <typename T>
        static void matmul_grad(const T* A, T* dA, const T* B, T* dB, const T* C, const T* dC, int* a_dim, int* b_dim, int* k, int* strideA, int* strideB, int* strideC) {
            // dA = dC * B^T
            Operations::matmul(dC, B, dA, a_dim[0], b_dim[0], k[0], strideC, strideB, strideA);
            // dB = A^T * dC
            Operations::matmul(A, dC, dB, a_dim[1], b_dim[1], k[1], strideA+2, strideC+2, strideB+2);
        }
        template <typename T>
        static void batch_matmul_grad(const T* A, T* dA, const T* B, T* dB, const T* C, const T* dC, int* a_off, int* b_off, int* k, int** strideA, int** strideB, int** strideC, int** reps, int** count, size_t* D) {
            // dA = dC * B^T
            Operations::batch_matmul<T>(dC, B, dA, a_off[0], b_off[0], k[0], strideC[0], strideB[0], strideA[0], reps[0], count[0], D[0]);
            // dB = A^T * dC
            Operations::batch_matmul<T>(A, dC, dB, a_off[1], b_off[1], k[1], strideA[1], strideC[1], strideB[1], reps[1], count[1], D[1]);
        }

        /*
        UNARY GRADS
        */

        template <typename T, class Grad>
        static void unary_pointwise(const T* A, T* dA, const T* C, const T* dC, size_t len, Grad grad) {
            const T* end = C + len;
            for (; C != end; A++, dA++, C++, dC++) {
                grad(*A, *dA, *C, *dC);
            }
        }

        template <typename T, class Grad>
        static void unary_scalarRhs(const T* A, T* dA, const T* C, const T* dC, size_t len, Grad grad) {
            const T* end = A + len;
            for (; A != end; A++, dA++) {
                grad(*A, *dA, *C, *dC);
            }
        }

        template <typename T, class Grad>
        static void unary_flexible(const T* A, T* dA, const T* C, const T* dC, int* strideA, int* strideC, int* reps, int* count, size_t D, Grad grad) {
            int indA = 0, indC = 0;
            while (1) {

                grad(A[indA], dA[indA], C[indC], dC[indC]);

                for (int dim = D - 1; dim >= 0; dim--) {
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
        template <typename T, class Grad>
        static void transp_grad(const T* A, T* dA, const T* C, const T* dC, size_t len) {
            const T* end = C + len;
            for (; C != end; dA++, dC++) {
                *dA += *dC;
            }
        }

        // f(A) = sum(A)
        // df_dA = tensor with shape of A filled with 1
        template <typename T, class Grad>
        static void sum_grad(const T* A, T* dA, const T* C, const T* dC, size_t len, Grad _) {
            const T* end = C + len;
            for (; C != end; dA++) {
                *dA += *dC;
            }
        }

        // f(A) = mean(A)
        // df_dA = tensor with shape of A filled with 1 / (len of A)
        template <typename T, class Grad>
        static void mean_grad(const T* A, T* dA, const T* C, const T* dC, size_t len, Grad _) {
            T mean = dC[0] / len;
            const T* end = C + len;
            for (; C != end; dA++) {
                *dA += mean;
            }
        }

        template <typename T, class Grad>
        static void mean_dim_grad(const T* A, T* dA, const T* C, const T* dC, T divisor, size_t c_len, int* strideA, int* strideC, int* reps, int* count, size_t D) {
            Operations::mean_dim<T>(dC, dA, divisor, c_len, strideC, strideA, reps, count, D);
        }
    };
}    
