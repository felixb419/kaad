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

    //d/dx[ f(g(x)) ] = f'(g(x)) * g'(x)

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

    /*
    UNCATEGORIZED
    */
    /*
    // f(A,B) = A outer B
    // df/dA = B
    // df/dB = A
    static void outer_grad(const T* A, const T* B, const T* C, T* dA, T* dB, const T* dC, int** strideA, int** strideB, int** strideC, int** reps, int** count, size_t* strideLen) {
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
        
            dA->val[indA] += dC->val[indC] * B->val[indB];
            dB->val[indB] += dC->val[indC] * A->val[indA];
        
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

    // f(A,B) = AB
    // dC/dA = B^T
    // dC/dB = A^T
    static void matmul_grad(const T* A, const T* B, const T* C, T* dA, T* dB, const T* dC, int** strideA, int** strideB, int** strideC, int** reps, int** count, size_t* strideLen) {
        int k = reps[0][2];
        int indA = 0, indB = 0, indC = 0, prev;
        for (int row = 0; row < reps[0][0]; row++) {
            prev = indB;
            for (int col = 0; col < reps[0][1]; col++) {
                for (int i = 0; i < k; i++) {
                    dA[indA] += dC[indC + i*strideC[0][1]] * B[indB + i*strideB[0][0]]; 
                }
                indA += strideA[0][1];
                indB += strideB[0][1];
            }
            indA += strideA[0][0];
            indC += strideC[0][0];
            indB = prev;
        }

        k = reps[1][2];
        indA = 0, indB = 0, indC = 0;
        for (int row = 0; row < reps[1][0]; row++) {
            prev = indC;
            for (int col = 0; col < reps[1][1]; col++) {
                for (int i = 0; i < k; i++) {
                    dB[indB] += A[indA + i*strideA[1][1]] * dC[indC + i*strideC[1][0]]; 
                }
                indB += strideB[1][1];
                indC += strideC[1][1];
            }
            indB += strideB[1][0];
            indA += strideA[1][0];
            indC = prev;
        }
    }
    
    static void batch_matmul_grad(const T* A, const T* B, const T* C, T* dA, T* dB, const T* dC, int** strideA, int** strideB, int** strideC, int** reps, int** count, size_t* strideLen) {
        // dA = dC * B^T
        batch_matmul(dC, B, dA, strideC[0], strideB[0], strideA[0], reps[0], count[0], strideLen[0]);
        // dB = A^T * dC
        batch_matmul(A, dC, dB, strideA[1], strideC[1], strideB[1], reps[1], count[1], strideLen[1]);
    }

    // f(A) = min(A,B)
    // df/dA [i] = A < B ? 1 : 0
    // df/dB [i] = B < A ? 1 : 0
    static void scalarMin_grad(const T* A, const T* B, const T* C, T* dA, T* dB, const T* dC, int** strideA, int** strideB, int** strideC, int** reps, int** count, size_t* strideLen) {
        for (size_t i = 0; i < dC->len; i++) {
            int smaller = A->val[i] <= B->val[0];
            T C_val = dC->val[i];
            dA->val[i] += smaller ? C_val : 0;
            dB->val[0] += smaller ? 0 : C_val;
        }
    }

    static void pointMin_grad(const T* A, const T* B, const T* C, T* dA, T* dB, const T* dC, int** strideA, int** strideB, int** strideC, int** reps, int** count, size_t* strideLen) {
        for (size_t i = 0; i < dC->len; i++) {
            int smaller = A->val[i] <= B->val[i];
            T C_val = dC->val[i];
            dA->val[i] += smaller ? C_val : 0;
            dB->val[i] += smaller ? 0 : C_val;
        }
    }

    static void flexMin_grad(const T* A, const T* B, const T* C, T* dA, T* dB, const T* dC, int** strideA, int** strideB, int** strideC, int** reps, int** count, size_t* strideLen) {
        int offsetA = dC->shapeLen - A->shapeLen;
        int offsetB = dC->shapeLen - B->shapeLen;
    
        int* effstrideA = new int[dC->shapeLen * 3];
        int* effstrideB = effstrideA + dC->shapeLen;
    
        for (size_t i = 0; i < dC->shapeLen; i++) {
            int aDim = i - offsetA;
            effstrideA[i] = aDim >= 0 && A->shape[aDim] != 1 ? A->stride[aDim] : 0;
            int bDim = i - offsetB;
            effstrideB[i] = bDim >= 0 && B->shape[bDim] != 1 ? B->stride[bDim] : 0;
        }
    
        int indA = 0, indB = 0, indC = 0;
        int* cords = effstrideB + dC->shapeLen;
        fill(cords, cords + dC->shapeLen, 0);
        for (int i = 0; i < dC->len; i++) {
        
            T A_val = A->val[indA];
            T B_val = B->val[indB];
            T C_val = dC->val[indC];
            int smaller = A_val <= B_val;

            dA->val[indA] += smaller ? C_val : 0;
            dB->val[indB] += smaller ? 0 : C_val;
        
            for (int dim = dC->shapeLen - 1; dim >= 0; dim--) {
                cords[dim]++;
                indA += effstrideA[dim];
                indB += effstrideB[dim];
                indC += dC->stride[dim];
            
                if (cords[dim] < dC->shape[dim]) {
                    break;
                }
                else {
                    cords[dim] = 0;
                    indA -= effstrideA[dim] * dC->shape[dim];
                    indB -= effstrideB[dim] * dC->shape[dim];
                    indC -= dC->stride[dim] * dC->shape[dim];
                }
            }
        
        }    
        delete[] effstrideA;
    }

    // f(A) = max(A,B)
    // df/dA [i] = A > B ? 1 : 0
    // df/dB [i] = B > A ? 1 : 0
    static void scalarMax_grad(const T* A, const T* B, const T* C, T* dA, T* dB, const T* dC, int** strideA, int** strideB, int** strideC, int** reps, int** count, size_t* strideLen) {
        for (size_t i = 0; i < dC->len; i++) {
            int bigger = A->val[i] >= B->val[0];
            T C_val = dC->val[i];
            dA->val[i] += bigger ? C_val : 0;
            dB->val[0] += bigger ? 0 : C_val;
        }
    }

    static void pointMax_grad(const T* A, const T* B, const T* C, T* dA, T* dB, const T* dC, int** strideA, int** strideB, int** strideC, int** reps, int** count, size_t* strideLen) {
        for (size_t i = 0; i < dC->len; i++) {
            int bigger = A->val[i] >= B->val[i];
            T C_val = dC->val[i];
            dA->val[i] += bigger ? C_val : 0;
            dB->val[i] += bigger ? 0 : C_val;
        }
    }

    static void flexMax_grad(const T* A, const T* B, const T* C, T* dA, T* dB, const T* dC, int** strideA, int** strideB, int** strideC, int** reps, int** count, size_t* strideLen) {
        int offsetA = dC->shapeLen - A->shapeLen;
        int offsetB = dC->shapeLen - B->shapeLen;
    
        int* effstrideA = new int[dC->shapeLen * 3];
        int* effstrideB = effstrideA + dC->shapeLen;
    
        for (size_t i = 0; i < dC->shapeLen; i++) {
            int aDim = i - offsetA;
            effstrideA[i] = aDim >= 0 && A->shape[aDim] != 1 ? A->stride[aDim] : 0;
            int bDim = i - offsetB;
            effstrideB[i] = bDim >= 0 && B->shape[bDim] != 1 ? B->stride[bDim] : 0;
        }
    
        int indA = 0, indB = 0, indC = 0;
        int* cords = effstrideB + dC->shapeLen;
        fill(cords, cords + dC->shapeLen, 0);
        for (int i = 0; i < dC->len; i++) {
        
            T A_val = A->val[indA];
            T B_val = B->val[indB];
            T C_val = dC->val[indC];
            int bigger = A_val >= B_val;

            dA->val[indA] += bigger ? C_val : 0;
            dB->val[indB] += bigger ? 0 : C_val;
        
            for (int dim = dC->shapeLen - 1; dim >= 0; dim--) {
                cords[dim]++;
                indA += effstrideA[dim];
                indB += effstrideB[dim];
                indC += dC->stride[dim];
            
                if (cords[dim] < dC->shape[dim]) {
                    break;
                }
                else {
                    cords[dim] = 0;
                    indA -= effstrideA[dim] * dC->shape[dim];
                    indB -= effstrideB[dim] * dC->shape[dim];
                    indC -= dC->stride[dim] * dC->shape[dim];
                }
            }
        
        }    
        delete[] effstrideA;
    }
    
    // f(A) = sum(A)
    // df_dA = tensor with shape of A filled with 1
    static void sum_grad(const T* A, const T* B, const T* C, T* dA, T* dB, const T* dC, int** strideA, int** strideB, int** strideC, int** reps, int** count, size_t* strideLen) {
       // dA += dC[0]
       for (size_t i = 0; i < dA->len; i++) {
           dA->val[i] += dC->val[0];
       }
    }

    static void sum_dim_grad(const T* A, const T* B, const T* C, T* dA, T* dB, const T* dC, int** strideA, int** strideB, int** strideC, int** reps, int** count, size_t* strideLen) {
        fill(dA->val, dA->val + dA->len, 0);

        int k = 0; // fix later
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
            
            dA->val[indA] += dC->val[indC];

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
    
    static void batch_matmul_grad(const T* A, const T* B, const T* C, T* dA, T* dB, const T* dC, int** strideA, int** strideB, int** strideC, int** reps, int** count, size_t* strideLen) {
    }

    // f(A) = min(A,B)
    // df/dA [i] = A < B ? 1 : 0
    // df/dB [i] = B < A ? 1 : 0
    static void scalarMin_grad(const T* A, const T* B, const T* C, T* dA, T* dB, const T* dC, int** strideA, int** strideB, int** strideC, int** reps, int** count, size_t* strideLen) {
        for (size_t i = 0; i < dC->len; i++) {
            int smaller = A->val[i] <= B->val[0];
            T C_val = dC->val[i];
            dA->val[i] += smaller ? C_val : 0;
            dB->val[0] += smaller ? 0 : C_val;
        }
    }

    static void pointMin_grad(const T* A, const T* B, const T* C, T* dA, T* dB, const T* dC, int** strideA, int** strideB, int** strideC, int** reps, int** count, size_t* strideLen) {
        for (size_t i = 0; i < dC->len; i++) {
            int smaller = A->val[i] <= B->val[i];
            T C_val = dC->val[i];
            dA->val[i] += smaller ? C_val : 0;
            dB->val[i] += smaller ? 0 : C_val;
        }
    }

    static void flexMin_grad(const T* A, const T* B, const T* C, T* dA, T* dB, const T* dC, int** strideA, int** strideB, int** strideC, int** reps, int** count, size_t* strideLen) {
        int offsetA = dC->shapeLen - A->shapeLen;
        int offsetB = dC->shapeLen - B->shapeLen;
    
        int* effstrideA = new int[dC->shapeLen * 3];
        int* effstrideB = effstrideA + dC->shapeLen;
    
        for (size_t i = 0; i < dC->shapeLen; i++) {
            int aDim = i - offsetA;
            effstrideA[i] = aDim >= 0 && A->shape[aDim] != 1 ? A->stride[aDim] : 0;
            int bDim = i - offsetB;
            effstrideB[i] = bDim >= 0 && B->shape[bDim] != 1 ? B->stride[bDim] : 0;
        }
    
        int indA = 0, indB = 0, indC = 0;
        int* cords = effstrideB + dC->shapeLen;
        fill(cords, cords + dC->shapeLen, 0);
        for (int i = 0; i < dC->len; i++) {
        
            T A_val = A->val[indA];
            T B_val = B->val[indB];
            T C_val = dC->val[indC];
            int smaller = A_val <= B_val;

            dA->val[indA] += smaller ? C_val : 0;
            dB->val[indB] += smaller ? 0 : C_val;
        
            for (int dim = dC->shapeLen - 1; dim >= 0; dim--) {
                cords[dim]++;
                indA += effstrideA[dim];
                indB += effstrideB[dim];
                indC += dC->stride[dim];
            
                if (cords[dim] < dC->shape[dim]) {
                    break;
                }
                else {
                    cords[dim] = 0;
                    indA -= effstrideA[dim] * dC->shape[dim];
                    indB -= effstrideB[dim] * dC->shape[dim];
                    indC -= dC->stride[dim] * dC->shape[dim];
                }
            }
        
        }    
        delete[] effstrideA;
    }

    // f(A) = max(A,B)
    // df/dA [i] = A > B ? 1 : 0
    // df/dB [i] = B > A ? 1 : 0
    static void scalarMax_grad(const T* A, const T* B, const T* C, T* dA, T* dB, const T* dC, int** strideA, int** strideB, int** strideC, int** reps, int** count, size_t* strideLen) {
        for (size_t i = 0; i < dC->len; i++) {
            int bigger = A->val[i] >= B->val[0];
            T C_val = dC->val[i];
            dA->val[i] += bigger ? C_val : 0;
            dB->val[0] += bigger ? 0 : C_val;
        }
    }

    static void pointMax_grad(const T* A, const T* B, const T* C, T* dA, T* dB, const T* dC, int** strideA, int** strideB, int** strideC, int** reps, int** count, size_t* strideLen) {
        for (size_t i = 0; i < dC->len; i++) {
            int bigger = A->val[i] >= B->val[i];
            T C_val = dC->val[i];
            dA->val[i] += bigger ? C_val : 0;
            dB->val[i] += bigger ? 0 : C_val;
        }
    }

    static void flexMax_grad(const T* A, const T* B, const T* C, T* dA, T* dB, const T* dC, int** strideA, int** strideB, int** strideC, int** reps, int** count, size_t* strideLen) {
        int offsetA = dC->shapeLen - A->shapeLen;
        int offsetB = dC->shapeLen - B->shapeLen;
    
        int* effstrideA = new int[dC->shapeLen * 3];
        int* effstrideB = effstrideA + dC->shapeLen;
    
        for (size_t i = 0; i < dC->shapeLen; i++) {
            int aDim = i - offsetA;
            effstrideA[i] = aDim >= 0 && A->shape[aDim] != 1 ? A->stride[aDim] : 0;
            int bDim = i - offsetB;
            effstrideB[i] = bDim >= 0 && B->shape[bDim] != 1 ? B->stride[bDim] : 0;
        }
    
        int indA = 0, indB = 0, indC = 0;
        int* cords = effstrideB + dC->shapeLen;
        fill(cords, cords + dC->shapeLen, 0);
        for (int i = 0; i < dC->len; i++) {
        
            T A_val = A->val[indA];
            T B_val = B->val[indB];
            T C_val = dC->val[indC];
            int bigger = A_val >= B_val;

            dA->val[indA] += bigger ? C_val : 0;
            dB->val[indB] += bigger ? 0 : C_val;
        
            for (int dim = dC->shapeLen - 1; dim >= 0; dim--) {
                cords[dim]++;
                indA += effstrideA[dim];
                indB += effstrideB[dim];
                indC += dC->stride[dim];
            
                if (cords[dim] < dC->shape[dim]) {
                    break;
                }
                else {
                    cords[dim] = 0;
                    indA -= effstrideA[dim] * dC->shape[dim];
                    indB -= effstrideB[dim] * dC->shape[dim];
                    indC -= dC->stride[dim] * dC->shape[dim];
                }
            }
        
        }    
        delete[] effstrideA;
    }
    
    // f(A) = sum(A)
    // df_dA = tensor with shape of A filled with 1
    static void sum_grad(const T* A, const T* B, const T* C, T* dA, T* dB, const T* dC, int** strideA, int** strideB, int** strideC, int** reps, int** count, size_t* strideLen) {
       // dA += dC[0]
       for (size_t i = 0; i < dA->len; i++) {
           dA->val[i] += dC->val[0];
       }
    }

    static void sum_dim_grad(const T* A, const T* B, const T* C, T* dA, T* dB, const T* dC, int** strideA, int** strideB, int** strideC, int** reps, int** count, size_t* strideLen) {
        fill(dA->val, dA->val + dA->len, 0);

        int k = 0; // fix later
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
            
            dA->val[indA] += dC->val[indC];

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

    // f(A) = sum(A)
    // df_dA = tensor with shape of A filled with 1 / (len of A)
    static void mean_grad(const T* A, const T* B, const T* C, T* dA, T* dB, const T* dC, int** strideA, int** strideB, int** strideC, int** reps, int** count, size_t* strideLen) {
       // dA += dC[0]
       T inv = dC->val[0] / dC->len;
       for (size_t i = 0; i < dA->len; i++) {
           dA->val[i] += inv;
       }
    }
    
    static void mean_dim_grad(const T* A, const T* B, const T* C, T* dA, T* dB, const T* dC, int** strideA, int** strideB, int** strideC, int** reps, int** count, size_t* strideLen) {
        fill(dA->val, dA->val + dA->len, 0);

        int k = 0; // fix later
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
            
            dA->val[indA] += dC->val[indC] / A->shape[k];

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