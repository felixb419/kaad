#pragma once

#include "tensor.h"
#include "operations.h"

#include <unordered_map>
#include <vector>

#define safer_powergradient

template <typename T>
using gradientOp = void(*)(const T* A, const T* B, const T* C, T* dA, T* dB, const T* dC, int** strideA, int** strideB, int** strideC, int** reps, int** count, size_t* stridelen);

template <typename T>
struct Gradients {

    //d/dx[ f(g(x,...)) ] = f'(g(x,...)) * g'(x,...)

    /*
    BINARY OPS
    */

    // f(A,B) = A + B
    // df/dA = 1
    // df/dB = 1
    static void scalarAddRt_grad(const T* A, const T* B, const T* C, T* dA, T* dB, const T* dC, int** strideA, int** strideB, int** strideC, int** reps, int** count, size_t* strideLen) {
        // B has shape = (1,)
        for (size_t i = 0; i < strideLen[0]; i++) {
            dA[i] += dC[i];
            dB[0] += dC[i];
        }
    }
    static void scalarAddLt_grad(const T* A, const T* B, const T* C, T* dA, T* dB, const T* dC, int** strideA, int** strideB, int** strideC, int** reps, int** count, size_t* strideLen) {
        // A has shape = (1,)
        for (size_t i = 0; i < strideLen[0]; i++) {
            dA[0] += dC[i];
            dB[i] += dC[i];
        }
    }
    static void pointAdd_grad(const T* A, const T* B, const T* C, T* dA, T* dB, const T* dC, int** strideA, int** strideB, int** strideC, int** reps, int** count, size_t* strideLen) {
        for (size_t i = 0; i < strideLen[0]; i++) {
            dA[i] += dC[i];
            dB[i] += dC[i];
        }
    }
    static void flexAdd_grad(const T* A, const T* B, const T* C, T* dA, T* dB, const T* dC, int** strideA, int** strideB, int** strideC, int** reps, int** count, size_t* strideLen) {
        int indA = 0, indB = 0, indC = 0;
        while (1) {

            T dc_i = dC[indC];
            dA[indA] += dc_i;
            dB[indB] += dc_i;

            for (int dim = *strideLen - 1; dim >= 0; dim--) {
                count[0][dim]--;
                if (count[0][dim] >= 0) {
                    indA += strideA[0][dim];
                    indB += strideB[0][dim];
                    indC += strideC[0][dim];
                    break;
                }

                count[0][dim] = reps[0][dim];
                if (dim == 0) goto end;
            }
        }
        end:;
    }
    
    // f(A,B) = A - B
    // df/dA = 1
    // df/dB = -1
    static void scalarSubRt_grad(const T* A, const T* B, const T* C, T* dA, T* dB, const T* dC, int** strideA, int** strideB, int** strideC, int** reps, int** count, size_t* strideLen) {
        // B has shape = (1,)
        for (size_t i = 0; i < strideLen[0]; i++) {
            dA[i] += dC[i];
            dB[0] -= dC[i];
        }
    }
    static void scalarSubLt_grad(const T* A, const T* B, const T* C, T* dA, T* dB, const T* dC, int** strideA, int** strideB, int** strideC, int** reps, int** count, size_t* strideLen) {
        // A has shape = (1,)
        for (size_t i = 0; i < strideLen[0]; i++) {
            dA[0] += dC[i];
            dB[i] -= dC[i];
        }
    }
    static void pointSub_grad(const T* A, const T* B, const T* C, T* dA, T* dB, const T* dC, int** strideA, int** strideB, int** strideC, int** reps, int** count, size_t* strideLen) {
        for (size_t i = 0; i < strideLen[0]; i++) {
            dA[i] += dC[i];
            dB[i] -= dC[i];
        }
    }
    static void flexSub_grad(const T* A, const T* B, const T* C, T* dA, T* dB, const T* dC, int** strideA, int** strideB, int** strideC, int** reps, int** count, size_t* strideLen) {
        int indA = 0, indB = 0, indC = 0;
        while (1) {

            T dc_i = dC[indC];
            dA[indA] += dc_i;
            dB[indB] -= dc_i;

            for (int dim = *strideLen - 1; dim >= 0; dim--) {
                count[0][dim]--;
                if (count[0][dim] >= 0) {
                    indA += strideA[0][dim];
                    indB += strideB[0][dim];
                    indC += strideC[0][dim];
                    break;
                }

                count[0][dim] = reps[0][dim];
                if (dim == 0) goto end;
            }
        }
        end:;
    }
    
    // f(A,B) = A * B
    // df/dA = B
    // df/dB = A
    static void scalarMulRt_grad(const T* A, const T* B, const T* C, T* dA, T* dB, const T* dC, int** strideA, int** strideB, int** strideC, int** reps, int** count, size_t* strideLen) {
        // B has shape = (1,)
        for (size_t i = 0; i < strideLen[0]; i++) {
            dA[i] += dC[i] * B[0];
            dB[0] += dC[i] * A[i];
        }
    }
    static void scalarMulLt_grad(const T* A, const T* B, const T* C, T* dA, T* dB, const T* dC, int** strideA, int** strideB, int** strideC, int** reps, int** count, size_t* strideLen) {
        // A has shape = (1,)
        for (size_t i = 0; i < strideLen[0]; i++) {
            dA[0] += dC[i] * B[i];
            dB[i] += dC[i] * A[0];
        }
    }
    static void pointMul_grad(const T* A, const T* B, const T* C, T* dA, T* dB, const T* dC, int** strideA, int** strideB, int** strideC, int** reps, int** count, size_t* strideLen) {
        for (size_t i = 0; i < strideLen[0]; i++) {
            dA[i] += dC[i] * B[i];
            dB[i] += dC[i] * A[i];
        }
    }
    static void flexMul_grad(const T* A, const T* B, const T* C, T* dA, T* dB, const T* dC, int** strideA, int** strideB, int** strideC, int** reps, int** count, size_t* strideLen) {
        int indA = 0, indB = 0, indC = 0;
        while (1) {

            T dc_i = dC[indC];
            dA[indA] += dc_i * B[indB];
            dB[indB] += dc_i * A[indA];

            for (int dim = *strideLen - 1; dim >= 0; dim--) {
                count[0][dim]--;
                if (count[0][dim] >= 0) {
                    indA += strideA[0][dim];
                    indB += strideB[0][dim];
                    indC += strideC[0][dim];
                    break;
                }

                count[0][dim] = reps[0][dim];
                if (dim == 0) goto end;
            }
        }
        end:;
    }
    
    // f(A,B) = A / B
    // df/dA = 1 / B
    // df/dB = -A / B^2
    static void scalarDivRt_grad(const T* A, const T* B, const T* C, T* dA, T* dB, const T* dC, int** strideA, int** strideB, int** strideC, int** reps, int** count, size_t* strideLen) {
        // B has shape = (1,)
        T B_inv = 1 / B[0];
        T B_sqr = B[0] * B[0];
        for (size_t i = 0; i < strideLen[0]; i++) {
            dA[i] += dC[i] * B_inv;
            dB[0] -= dC[i] * (A[i] / B_sqr);
        }
    }
    static void scalarDivLt_grad(const T* A, const T* B, const T* C, T* dA, T* dB, const T* dC, int** strideA, int** strideB, int** strideC, int** reps, int** count, size_t* strideLen) {
        // A has shape = (1,)
        T B_inv = 1 / B[0];
        for (size_t i = 0; i < strideLen[0]; i++) {
            dA[0] += dC[i] * (1 / B[i]);
            dB[i] -= dC[i] * (A[0] / (B[i] * B[i]));
        }
    }
    static void pointDiv_grad(const T* A, const T* B, const T* C, T* dA, T* dB, const T* dC, int** strideA, int** strideB, int** strideC, int** reps, int** count, size_t* strideLen) {
        for (size_t i = 0; i < strideLen[0]; i++) {
            dA[i] += dC[i] * (1 / B[i]);
            dB[i] -= dC[i] * (A[i] / (B[i] * B[i]));
        }
    }
    static void flexDiv_grad(const T* A, const T* B, const T* C, T* dA, T* dB, const T* dC, int** strideA, int** strideB, int** strideC, int** reps, int** count, size_t* strideLen) {
        int indA = 0, indB = 0, indC = 0;
        while (1) {

            T b_i = B[indB];
            T dc_i = dC[indC];
            dA[indA] += dc_i * (1 / b_i);
            dB[indB] -= dc_i * (A[indA] / (b_i * b_i));

            for (int dim = *strideLen - 1; dim >= 0; dim--) {
                count[0][dim]--;
                if (count[0][dim] >= 0) {
                    indA += strideA[0][dim];
                    indB += strideB[0][dim];
                    indC += strideC[0][dim];
                    break;
                }

                count[0][dim] = reps[0][dim];
                if (dim == 0) goto end;
            }
        }
        end:;
    }
    
    // f(A,B) = A ^ B
    // df/dA = B * A ^ (B - 1)
    // df/dB = A ^ B * log(|A|)     df/dB is 0 if A is negative for stability
    static void scalarPowRt_grad(const T* A, const T* B, const T* C, T* dA, T* dB, const T* dC, int** strideA, int** strideB, int** strideC, int** reps, int** count, size_t* strideLen) {
        // B has shape = (1,)
        for (size_t i = 0; i < strideLen[0]; i++) {
            dA[i] += dC[i] * (B[0] * pow(A[i], B[0] - 1));

            #ifdef safer_powergradient
            // add 0 to dB[i] instead if NaN if A[i] < 0
            dB[0] += dC[i] * (A[i] < 0 ? 0 : C[i] * log(A[i]));
            #else
            dB[0] += dC[i] * (C[i] * log(A[i]));
            #endif
        }
    }
    static void scalarPowLt_grad(const T* A, const T* B, const T* C, T* dA, T* dB, const T* dC, int** strideA, int** strideB, int** strideC, int** reps, int** count, size_t* strideLen) {
        // A has shape = (1,)
        T A_log = A[0] < 0 ? 0 : log(A[0]);
        for (size_t i = 0; i < strideLen[0]; i++) {
            dA[0] += dC[i] * (B[i] * pow(A[0], B[i] - 1));

            #ifdef safer_powergradient
            // add 0 to dB[i] instead if NaN if A[i] < 0
            dB[i] += dC[i] * (C[i] * A_log);
            #else
            dB[i] += dC[i] * (C[i] * A_log);
            #endif
        }
    }
    static void pointPow_grad(const T* A, const T* B, const T* C, T* dA, T* dB, const T* dC, int** strideA, int** strideB, int** strideC, int** reps, int** count, size_t* strideLen) {
        for (size_t i = 0; i < strideLen[0]; i++) {
            dA[i] += dC[i] * (B[i] * pow(A[i], B[i] - 1));

            #ifdef safer_powergradient
            // add 0 to dB[i] instead if NaN if A[i] < 0
            dB[i] += dC[i] * (C[i] * (A[i] < 0 ? 0 : log(A[i])));
            #else
            dB[i] += dC[i] * (C[i] * log(A[i]));
            #endif
        }
    }
    static void flexPow_grad(const T* A, const T* B, const T* C, T* dA, T* dB, const T* dC, int** strideA, int** strideB, int** strideC, int** reps, int** count, size_t* strideLen) {
        int indA = 0, indB = 0, indC = 0;
        while (1) {

            dA[indA] += dC[indC] * (B[indB] * pow(A[indA], B[indB] - 1));
            #ifdef safer_powergradient
            // add 0 to dB[i] instead if NaN if A[i] < 0
            dB[indB] += dC[indC] * (C[indC] * (A[indA] < 0 ? 0 : log(A[indA])));
            #else
            dB[indB] += dC[indC] * (C[indC] * log(A[indA]));
            #endif

            for (int dim = *strideLen - 1; dim >= 0; dim--) {
                count[0][dim]--;
                if (count[0][dim] >= 0) {
                    indA += strideA[0][dim];
                    indB += strideB[0][dim];
                    indC += strideC[0][dim];
                    break;
                }

                count[0][dim] = reps[0][dim];
                if (dim == 0) goto end;
            }
        }
        end:;
    }

    // f(A,B) = A dot B
    // df/dA = B
    // df/dB = A
    static void dot_grad(const T* A, const T* B, const T* C, T* dA, T* dB, const T* dC, int** strideA, int** strideB, int** strideC, int** reps, int** count, size_t* strideLen) {
        for (size_t i = 0; i < strideLen[0]; i++) {
            dA[i] += dC[0] * B[i];
            dB[i] += dC[0] * A[i];
        }
    }
    static void scalarDot_grad(const T* A, const T* B, const T* C, T* dA, T* dB, const T* dC, int** strideA, int** strideB, int** strideC, int** reps, int** count, size_t* strideLen) {
        for (size_t i = 0; i < strideLen[0]; i++) {
            dA[i] += dC[0] * B[0];
            dB[0] += dC[0] * A[i];
        }
    }

    // f(A,B) = AB
    // dC/dA = B^T
    // dC/dB = A^T
    static void matmul_grad(const T* A, const T* B, const T* C, T* dA, T* dB, const T* dC, int** strideA, int** strideB, int** strideC, int** reps, int** count, size_t* strideLen) {
        // dA = dC * B^T
        Operations<T>::matmul(dC, B, dA, strideC[0], strideB[0], strideA[0], reps[0], count[0], strideLen[0]);
        // dB = A^T * dC
        Operations<T>::matmul(A, dC, dB, strideA[1], strideC[1], strideB[1], reps[1], count[1], strideLen[1]);
    }
    static void batch_matmul_grad(const T* A, const T* B, const T* C, T* dA, T* dB, const T* dC, int** strideA, int** strideB, int** strideC, int** reps, int** count, size_t* strideLen) {
        // dA = dC * B^T
        Operations<T>::batch_matmul(dC, B, dA, strideC[0], strideB[0], strideA[0], reps[0], count[0], strideLen[0]);
        // dB = A^T * dC
        Operations<T>::batch_matmul(A, dC, dB, strideA[1], strideC[1], strideB[1], reps[1], count[1], strideLen[1]);
    }

    // f(A) = min(A,B)
    // df/dA[i] = A < B ? 1 : 0
    // df/dB[i] = A < B ? 0 : 1
    static void scalarMinRt_grad(const T* A, const T* B, const T* C, T* dA, T* dB, const T* dC, int** strideA, int** strideB, int** strideC, int** reps, int** count, size_t* strideLen) {
        for (size_t i = 0; i < strideLen[0]; i++) {
            int smaller = A[i] <= B[0];
            T C_val = dC[i];
            dA[i] += smaller ? C_val : 0;
            dB[0] += smaller ? 0 : C_val;
        }
    }
    static void scalarMinLt_grad(const T* A, const T* B, const T* C, T* dA, T* dB, const T* dC, int** strideA, int** strideB, int** strideC, int** reps, int** count, size_t* strideLen) {
        for (size_t i = 0; i < strideLen[0]; i++) {
            int smaller = A[0] <= B[i];
            T C_val = dC[i];
            dA[0] += smaller ? C_val : 0;
            dB[i] += smaller ? 0 : C_val;
        }
    }
    static void pointMin_grad(const T* A, const T* B, const T* C, T* dA, T* dB, const T* dC, int** strideA, int** strideB, int** strideC, int** reps, int** count, size_t* strideLen) {
        for (size_t i = 0; i < strideLen[0]; i++) {
            int smaller = A[i] <= B[i];
            T C_val = dC[i];
            dA[i] += smaller ? C_val : 0;
            dB[i] += smaller ? 0 : C_val;
        }
    }
    static void flexMin_grad(const T* A, const T* B, const T* C, T* dA, T* dB, const T* dC, int** strideA, int** strideB, int** strideC, int** reps, int** count, size_t* strideLen) {
        int indA = 0, indB = 0, indC = 0;
        while (1) {

            int smaller = A[indA] <= B[indB];
            T C_val = dC[indC];
            dA[indA] += smaller ? C_val : 0;
            dB[indB] += smaller ? 0 : C_val;

            for (int dim = *strideLen - 1; dim >= 0; dim--) {
                count[0][dim]--;
                if (count[0][dim] >= 0) {
                    indA += strideA[0][dim];
                    indB += strideB[0][dim];
                    indC += strideC[0][dim];
                    break;
                }

                count[0][dim] = reps[0][dim];
                if (dim == 0) goto end;
            }
        }
        end:;
    }

    // f(A) = max(A,B)
    // df/dA[i] = A > B ? 1 : 0
    // df/dB[i] = A > B ? 0 : 1
    static void scalarMaxRt_grad(const T* A, const T* B, const T* C, T* dA, T* dB, const T* dC, int** strideA, int** strideB, int** strideC, int** reps, int** count, size_t* strideLen) {
        for (size_t i = 0; i < strideLen[0]; i++) {
            int bigger = A[i] >= B[0];
            T C_val = dC[i];
            dA[i] += bigger ? C_val : 0;
            dB[0] += bigger ? 0 : C_val;
        }
    }
    static void scalarMaxLt_grad(const T* A, const T* B, const T* C, T* dA, T* dB, const T* dC, int** strideA, int** strideB, int** strideC, int** reps, int** count, size_t* strideLen) {
        for (size_t i = 0; i < strideLen[0]; i++) {
            int bigger = A[0] >= B[i];
            T C_val = dC[i];
            dA[0] += bigger ? C_val : 0;
            dB[i] += bigger ? 0 : C_val;
        }
    }
    static void pointMax_grad(const T* A, const T* B, const T* C, T* dA, T* dB, const T* dC, int** strideA, int** strideB, int** strideC, int** reps, int** count, size_t* strideLen) {
        for (size_t i = 0; i < strideLen[0]; i++) {
            int bigger = A[i] >= B[i];
            T C_val = dC[i];
            dA[i] += bigger ? C_val : 0;
            dB[i] += bigger ? 0 : C_val;
        }
    }
    static void flexMax_grad(const T* A, const T* B, const T* C, T* dA, T* dB, const T* dC, int** strideA, int** strideB, int** strideC, int** reps, int** count, size_t* strideLen) {
        int indA = 0, indB = 0, indC = 0;
        while (1) {

            int bigger = A[indA] >= B[indB];
            T C_val = dC[indC];
            dA[indA] += bigger ? C_val : 0;
            dB[indB] += bigger ? 0 : C_val;

            for (int dim = *strideLen - 1; dim >= 0; dim--) {
                count[0][dim]--;
                if (count[0][dim] >= 0) {
                    indA += strideA[0][dim];
                    indB += strideB[0][dim];
                    indC += strideC[0][dim];
                    break;
                }

                count[0][dim] = reps[0][dim];
                if (dim == 0) goto end;
            }
        }
        end:;
    }

    /*
    UNARY OPS
    */

    // f(A) = -A
    // df/dA = -1
    static void negate_grad(const T* A, const T* B, const T* C, T* dA, T* dB, const T* dC, int** strideA, int** strideB, int** strideC, int** reps, int** count, size_t* strideLen) {
        // dA -= dC
        for (size_t i = 0; i < strideLen[0]; i++) {
            dA[i] -= dC[i];
        }
    }
    
    // f(A) = A^2
    // df/dA = 2A
    static void square_grad(const T* A, const T* B, const T* C, T* dA, T* dB, const T* dC, int** strideA, int** strideB, int** strideC, int** reps, int** count, size_t* strideLen) {
        // dA += dC * (A * 2)
        for (size_t i = 0; i < strideLen[0]; i++) {
            dA[i] += dC[i] * (A[i] * 2);
        }
    }
    
    // f(A) = sqrt(A)
    // df/dA = 1 / (2 * sqrt(A))
    static void sqrt_grad(const T* A, const T* B, const T* C, T* dA, T* dB, const T* dC, int** strideA, int** strideB, int** strideC, int** reps, int** count, size_t* strideLen) {
        // dA += dC / (2 * C)
        for (size_t i = 0; i < strideLen[0]; i++) {
            dA[i] += dC[i] / (2 * C[i]);
        }
    }
    
    // f(A) = log(A)
    // df/dA = 1 / x
    static void log_grad(const T* A, const T* B, const T* C, T* dA, T* dB, const T* dC, int** strideA, int** strideB, int** strideC, int** reps, int** count, size_t* strideLen) {
        // dA += dC / x
        for (size_t i = 0; i < strideLen[0]; i++) {
            dA[i] += dC[i] / A[i];
        }
    }
    
    // f(A) = e^A
    // df/dA = e^A
    static void exp_grad(const T* A, const T* B, const T* C, T* dA, T* dB, const T* dC, int** strideA, int** strideB, int** strideC, int** reps, int** count, size_t* strideLen) {
        // dA += dC * C
        for (size_t i = 0; i < strideLen[0]; i++) {
            dA[i] += dC[i] * C[i];
        }
    }
    
    // f(A) = |A|
    // df/dA = |A| / A       (if (A[i] < 0) -1 else 1)
    static void abs_grad(const T* A, const T* B, const T* C, T* dA, T* dB, const T* dC, int** strideA, int** strideB, int** strideC, int** reps, int** count, size_t* strideLen) {
        // dA += dC * if(A < 0) -1 else 1
        for (size_t i = 0; i < strideLen[0]; i++) {
            dA[i] += dC[i] * (A[i] < 0 ? -1 : 1);
        }
    }

    // f(A) = A^T
    // df/dA = 1
    static void transp_grad(const T* A, const T* B, const T* C, T* dA, T* dB, const T* dC, int** strideA, int** strideB, int** strideC, int** reps, int** count, size_t* strideLen) {
        for (size_t i = 0; i < strideLen[0]; i++) {
            dA[i] += dC[i];
        }
    }

    // f(A) = sum(A)
    // df_dA = tensor with shape of A filled with 1
    static void sum_grad(const T* A, const T* B, const T* C, T* dA, T* dB, const T* dC, int** strideA, int** strideB, int** strideC, int** reps, int** count, size_t* strideLen) {
        for (size_t i = 0; i < strideLen[0]; i++) {
            dA[i] += dC[0];
        }
    }
    static void sum_dim_grad(const T* A, const T* B, const T* C, T* dA, T* dB, const T* dC, int** strideA, int** strideB, int** strideC, int** reps, int** count, size_t* strideLen) {
        Operations<T>::sum_dim(dC, nullptr, dA, strideC[0], nullptr, strideA[0], reps[0], count[0], strideLen[0]);
    }

    // f(A) = mean(A)
    // df_dA = tensor with shape of A filled with 1 / (len of A)
    static void mean_grad(const T* A, const T* B, const T* C, T* dA, T* dB, const T* dC, int** strideA, int** strideB, int** strideC, int** reps, int** count, size_t* strideLen) {
        T mean = dC[0] / strideLen[0];
        for (size_t i = 0; i < strideLen[0]; i++) {
            dA[i] += mean;
        }
    }
    static void mean_dim_grad(const T* A, const T* B, const T* C, T* dA, T* dB, const T* dC, int** strideA, int** strideB, int** strideC, int** reps, int** count, size_t* strideLen) {
        Operations<T>::mean_dim(dC, nullptr, dA, strideC[0], nullptr, strideA[0], reps[0], count[0], strideLen[0]);
    }

    /*
    UNCATEGORIZED
    */
    /*

    static void tile_grad(const T* A, const T* B, const T* C, T* dA, T* dB, const T* dC, int** strideA, int** strideB, int** strideC, int** reps, int** count, size_t* strideLen) {
        int* m = B->shape;
        
        int offset = dC->shapeLen - A->shapeLen;
        int* effstride = new int[dC->shapeLen * 3];
        fill(effstride, effstride + offset, 0);
        copy(A->stride, A->stride + A->shapeLen, effstride + offset);

        int* superstride = effstride + dC->shapeLen;
        copy(A->shape, A->shape + A->shapeLen, superstride);
        for (int i = A->shapeLen - 2; i >= 0; i--) {
            superstride[i] *= superstride[i + 1];
        }

        int indA = 0, indC = 0;
        int* cords = effstride + dC->shapeLen * 2;
        fill(cords, cords + dC->shapeLen, 0);

        for (int i = 0; i < dC->len; i++) {

            dA->val[indA] += dC->val[indC];

            for (int dim = dC->shapeLen - 1; dim >= 0; dim--) {
                cords[dim]++;
                indA += effstride[dim];
                indA -= cords[dim] == A->shape[dim] && m[dim] != 1 ? superstride[dim] : 0;
                indC += dC->stride[dim];

                if (cords[dim] < dC->shape[dim]) {
                    break;
                }
                else {
                    cords[dim] = 0;
                    indA -= effstride[dim] * A->shape[dim];
                    indC -= dC->stride[dim] * dC->shape[dim];
                }
            }
        }

        delete[] effstride;
    }

    static void slice_grad(const T* A, const T* B, const T* C, T* dA, T* dB, const T* dC, int** strideA, int** strideB, int** strideC, int** reps, int** count, size_t* strideLen) {
        int offset = A->shapeLen - dC->shapeLen;
        int* effstride = new int[dC->shapeLen * 2];
        copy(A->stride + offset, A->stride + A->shapeLen - offset + 1, effstride);

        int* multiples = 0; // fix later
        int indA = 0, indC = 0;
        for (int i = 0; i < dC->shapeLen; i++) {
            indA += multiples[i] * A->stride[i];
        }

        int* cords = effstride + dC->shapeLen;
        fill(cords, cords + dC->shapeLen, 0);
        for (int i = 0; i < dC->len; i++) {

            dA->val[indA] += dC->val[indC];

            for (int dim = dC->shapeLen - 1; dim >= 0; dim--) {
                cords[dim]++;
                indA += effstride[dim];
                indC += dC->stride[dim];

                if (cords[dim] < dC->shape[dim]) {
                    break;
                }
                else {
                    cords[dim] = 0;
                    indA -= effstride[dim] * dC->shape[dim];
                    indC -= dC->stride[dim] * dC->shape[dim];
                }
            }
        }

        delete[] effstride;
    }
    */
};
