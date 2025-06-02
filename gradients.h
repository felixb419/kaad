#pragma once

#include "tensor.h"
#include "operations.h"

#include <unordered_map>
#include <vector>

#define safe_powergradient

template <typename T>
struct Gradients : Operations<T> {
    using Operations<T>::flexAdd_inplace;
    using Operations<T>::flexSub_inplace;
    using Operations<T>::flexMul;
    using Operations<T>::flexMul_inplace;
    using Operations<T>::flexDiv;
    using Operations<T>::flexPow;
    using Operations<T>::matmul;
    using Operations<T>::batch_matmul;

    // f(A,B) = A + B
    // df/dA = 1
    // df/dB = 1
    static void pointAdd_grad(const T* A, const T* B, const T* C, T* dA, T* dB, const T* dC, int** strideA, int** strideB, int** strideC, int** reps, int** count, size_t* strideLen) {
        // dA += dC
        for (size_t i = 0; i < strideLen[0]; i++) {
            dA[i] += dC[i];
        }
        
        // dB += dC
        for (size_t i = 0; i < strideLen[0]; i++) {
            dB[i] += dC[i];
        }
    }
    static void scalarAdd_grad(const T* A, const T* B, const T* C, T* dA, T* dB, const T* dC, int** strideA, int** strideB, int** strideC, int** reps, int** count, size_t* strideLen) {
        // B has shape = (1,)
        // dA += dC
        for (size_t i = 0; i < strideLen[0]; i++) {
            dA[i] += dC[i];
        }
        
        // dB += dC
        for (size_t i = 0; i < strideLen[0]; i++) {
            dB[0] += dC[i];
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
    static void pointSub_grad(const T* A, const T* B, const T* C, T* dA, T* dB, const T* dC, int** strideA, int** strideB, int** strideC, int** reps, int** count, size_t* strideLen) {
        // dA += dC
        for (size_t i = 0; i < strideLen[0]; i++) {
            dA[i] += dC[i];
        }
    
        // dB -= dC
        for (size_t i = 0; i < strideLen[0]; i++) {
            dB[i] -= dC[i];
        }
    }
    static void scalarSub_grad(const T* A, const T* B, const T* C, T* dA, T* dB, const T* dC, int** strideA, int** strideB, int** strideC, int** reps, int** count, size_t* strideLen) {
        // B has shape = (1,)
        // dA += dC
        for (size_t i = 0; i < strideLen[0]; i++) {
            dA[i] += dC[i];
        }
    
        // dB -= dC
        for (size_t i = 0; i < strideLen[0]; i++) {
            dB[0] -= dC[i];
        }
    }
    static void invScalarSub_grad(const T* A, const T* B, const T* C, T* dA, T* dB, const T* dC, int** strideA, int** strideB, int** strideC, int** reps, int** count, size_t* strideLen) {
        // A has shape = (1,)
        // dA += dC
        for (size_t i = 0; i < strideLen[0]; i++) {
            dA[0] += dC[i];
        }
    
        // dB -= dC
        for (size_t i = 0; i < strideLen[0]; i++) {
            dB[i] -= dC[i];
        }
    }
    static void flexSub_grad(const T* A, const T* B, const T* C, T* dA, T* dB, const T* dC, int** strideA, int** strideB, int** strideC, int** reps, int** count, size_t* strideLen) {
        int indA = 0, indB = 0, indC = 0;
        while (1) {

            T dc_i = dC[indC];
            dA[indA] += dc_i;
            dB[indB] -= dc_i;

            for (int dim = strideLen - 1; dim >= 0; dim--) {
                count[dim]--;
                if (count[0][dim] >= 0) {
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
    
    // f(A,B) = A * B
    // df/dA = B
    // df/dB = A
    static void pointMul_grad(const T* A, const T* B, const T* C, T* dA, T* dB, const T* dC, int** strideA, int** strideB, int** strideC, int** reps, int** count, size_t* strideLen) {
        // dA += dC * B
        for (size_t i = 0; i < strideLen[0]; i++) {
            dA[i] += dC[i] * B[i];
        }
    
        // dB += dC * A
        for (size_t i = 0; i < strideLen[0]; i++) {
            dB[i] += dC[i] * A[i];
        }
    }
    static void scalarMul_grad(const T* A, const T* B, const T* C, T* dA, T* dB, const T* dC, int** strideA, int** strideB, int** strideC, int** reps, int** count, size_t* strideLen) {
        // B has shape = (1,)
        // dA += dC * B
        for (size_t i = 0; i < strideLen[0]; i++) {
            dA[i] += dC[i] * B[0];
        }
    
        // dB += dC * A
        for (size_t i = 0; i < strideLen[0]; i++) {
            dB[0] += dC[i] * A[i];
        }
    }
    static void flexMul_grad(const T* A, const T* B, const T* C, T* dA, T* dB, const T* dC, int** strideA, int** strideB, int** strideC, int** reps, int** count, size_t* strideLen) {
        int indA = 0, indB = 0, indC = 0;
        while (1) {

            const T dc_i = dC[indC];
            dA[indA] += dc_i * B[indB];
            dB[indB] += dc_i * A[indA];

            for (int dim = strideLen - 1; dim >= 0; dim--) {
                count[dim]--;
                if (count[0][dim] >= 0) {
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
    
    // f(A,B) = A / B
    // df/dA = 1 / B
    // df/dB = -A / B^2
    static void pointDiv_grad(const T* A, const T* B, const T* C, T* dA, T* dB, const T* dC, int** strideA, int** strideB, int** strideC, int** reps, int** count, size_t* strideLen) {
        // dA += dC * (1 / B)
        for (size_t i = 0; i < strideLen[0]; i++) {
            dA[i] += dC[i] * (1 / B[i]);
        }
    
        // dB -= dC * (A / B^2)
        for (size_t i = 0; i < strideLen[0]; i++) {
            dB[i] -= dC[i] * (A[i] / (B[i] * B[i]));
        }
    }
    static void scalarDiv_grad(const T* A, const T* B, const T* C, T* dA, T* dB, const T* dC, int** strideA, int** strideB, int** strideC, int** reps, int** count, size_t* strideLen) {
        // B has shape = (1,)
        // dA += dC * (1 / B)
        T B_inv = 1 / B[0];
        for (size_t i = 0; i < strideLen[0]; i++) {
            dA[i] += dC[i] * B_inv;
        }
    
        // dB -= dC * (A / B^2)
        T B_sqr = B[0] * B[0];
        for (size_t i = 0; i < strideLen[0]; i++) {
            dB[0] -= dC[i] * (A[i] / B_sqr);
        }
    }
    static void invScalarDiv_grad(const T* A, const T* B, const T* C, T* dA, T* dB, const T* dC, int** strideA, int** strideB, int** strideC, int** reps, int** count, size_t* strideLen) {
        // A has shape = (1,)
        // dA += dC * (1 / B)
        T B_inv = 1 / B[0];
        for (size_t i = 0; i < strideLen[0]; i++) {
            dA[0] += dC[i] * (1 / B[i]);
        }
    
        // dB -= dC * (A / B^2)
        for (size_t i = 0; i < strideLen[0]; i++) {
            dB[i] -= dC[i] * (A[0] / (B[i] * B[i]));
        }
    }
    static void flexDiv_grad(const T* A, const T* B, const T* C, T* dA, T* dB, const T* dC, int** strideA, int** strideB, int** strideC, int** reps, int** count, size_t* strideLen) {
        int indA = 0, indB = 0, indC = 0;
        while (1) {

            T b_i = B[indB];
            T dc_i = dC[indC];
            dA[indA] += dc_i * (1 / b_i);
            dB[indB] -= dc_i * (A[indA] / (b_i * b_i));

            for (int dim = strideLen - 1; dim >= 0; dim--) {
                count[dim]--;
                if (count[0][dim] >= 0) {
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
    
    // f(A,B) = A ^ B
    // df/dA = B * A ^ (B - 1)
    // df/dB = A ^ B * log(|A|)     df/dB is 0 if A is negative for stability
    static void pointPow_grad(const T* A, const T* B, const T* C, T* dA, T* dB, const T* dC, int** strideA, int** strideB, int** strideC, int** reps, int** count, size_t* strideLen) {
        // dA += dC * (B * (A^(B - 1)))
        for (size_t i = 0; i < dC->len; i++) {
            dA->val[i] += dC->val[i] * (B->val[i] * pow(A->val[i], B->val[i] - 1));
        }
    
        // dB += dC * (C * log(A))
        for (size_t i = 0; i < dC->len; i++) {
            #ifdef safe_powergradient
            dB->val[i] += dC->val[i] * (C->val[i] * (A->val[i] < 0 ? 0 : log(A->val[i])));
            #else
            dB->val[i] += dC->val[i] * (C->val[i] * log(A->val[i]));
            #endif
        }
    }
    static void scalarPow_grad(const T* A, const T* B, const T* C, T* dA, T* dB, const T* dC, int** strideA, int** strideB, int** strideC, int** reps, int** count, size_t* strideLen) {
        // B has shape = (1,)
        // dA += dC * (B * (A^(B - 1)))
        for (size_t i = 0; i < dC->len; i++) {
            dA->val[i] += dC->val[i] * (B->val[0] * pow(A->val[i], B->val[0] - 1));
        }
    
        // dB += dC * (C * log(A))
        for (size_t i = 0; i < dC->len; i++) {
            #ifdef safe_powergradient
            dB->val[0] += dC->val[i] * (A->val[i] < 0 ? 0 : C->val[i] * log(A->val[i]));
            #else
            dB->val[0] += dC->val[i] * (C->val[i] * log(A->val[i]));
            #endif
        }
    }
    static void invScalarPow_grad(const T* A, const T* B, const T* C, T* dA, T* dB, const T* dC, int** strideA, int** strideB, int** strideC, int** reps, int** count, size_t* strideLen) {
        // A has shape = (1,)
        // dA += dC * (B * (A^(B - 1)))
        for (size_t i = 0; i < dC->len; i++) {
            dA->val[0] += dC->val[i] * (B->val[i] * pow(A->val[0], B->val[i] - 1));
        }
    
        // dB += dC * (C * log(A))
        T A_log = A->val[0] < 0 ? 0 : log(A->val[0]);
        for (size_t i = 0; i < dC->len; i++) {
            #ifdef safe_powergradient
            dB->val[i] += dC->val[i] * (C->val[i] * A_log);
            #else
            dB->val[i] += dC->val[i] * (C->val[i] * A_log);
            #endif
        }
    }
    static void flexPow_grad(const T* A, const T* B, const T* C, T* dA, T* dB, const T* dC, int** strideA, int** strideB, int** strideC, int** reps, int** count, size_t* strideLen) {
        T* cache = new T[dC->len + dC->len];
        // alt cache to astatic void cache conflict in flexible operation
        T* cache2 = cache + dC->len;
        tView<T> C_cache(dC);
        C_cache.val = cache;
    
        tView<T> A_cache (*A);
        A_cache.val = cache2;
        tView<T> B_cache (*B);
        B_cache.val = cache2;
    
        // dA += dC * (B * (A^(B - 1)))
        for (size_t i = 0; i < B->len; i++) {
            cache2[i] = B->val[i] - 1;
        }
        flexPow(A, &B_cache, &C_cache);
        flexMul_inplace(&C_cache, B, false);
        for (size_t i = 0; i < dC->len; i++) {
            cache[i] *= dC->val[i];
        }
        flexAdd_inplace(dA, &C_cache);
    
        // dB += dC * (C * log(A))
        for (size_t i = 0; i < A->len; i++) {
            #ifdef safe_powergradient
            cache2[i] = A->val[i] < 0 ? 0 : log(A->val[i]);
            #else
            cache2[i] = log(A->val[i]);
            #endif
        }
        flexMul(C, &A_cache, &C_cache);
        for (size_t i = 0; i < A->len; i++) {
            C_cache.val[i] *= dC->val[i];
        }
        flexAdd_inplace(dB, &C_cache);
    
        delete[] cache;
    }
    
    // f(A) = -A
    // df/dA = -1
    static void negate_grad(const T* A, const T* B, const T* C, T* dA, T* dB, const T* dC, int** strideA, int** strideB, int** strideC, int** reps, int** count, size_t* strideLen) {
        // dA -= dC
        flexSub_inplace(dA, dC);
    }
    
    // f(A) = A^2
    // df/dA = 2A
    static void square_grad(const T* A, const T* B, const T* C, T* dA, T* dB, const T* dC, int** strideA, int** strideB, int** strideC, int** reps, int** count, size_t* strideLen) {
        // dA += dC * (A * 2)
        for (size_t i = 0; i < dC->len; i++) {
            dA->val[i] += dC->val[i] * (A->val[i] * 2);
        }
    }
    
    // f(A) = sqrt(A)
    // df/dA = 1 / (2 * sqrt(A))
    static void sqrt_grad(const T* A, const T* B, const T* C, T* dA, T* dB, const T* dC, int** strideA, int** strideB, int** strideC, int** reps, int** count, size_t* strideLen) {
        // dA += dC / (2 * C)
        for (size_t i = 0; i < dC->len; i++) {
            dA->val[i] += dC->val[i] / (2 * C->val[i]);
        }
    }
    
    // f(A) = log(A)
    // df/dA = 1 / x
    static void log_grad(const T* A, const T* B, const T* C, T* dA, T* dB, const T* dC, int** strideA, int** strideB, int** strideC, int** reps, int** count, size_t* strideLen) {
        // dA += dC / x
        for (size_t i = 0; i < dC->len; i++) {
            dA->val[i] += dC->val[i] / A->val[i];
        }
    }
    
    // f(A) = e^A
    // df/dA = e^A
    static void exp_grad(const T* A, const T* B, const T* C, T* dA, T* dB, const T* dC, int** strideA, int** strideB, int** strideC, int** reps, int** count, size_t* strideLen) {
        // dA += dC * C
        for (size_t i = 0; i < dC->len; i++) {
            dA->val[i] += dC->val[i] * C->val[i];
        }
    }
    
    // f(A) = |A|
    // df/dA = |A| / A       (if (A[i] < 0) -1 else 1)
    static void abs_grad(const T* A, const T* B, const T* C, T* dA, T* dB, const T* dC, int** strideA, int** strideB, int** strideC, int** reps, int** count, size_t* strideLen) {
        // dA += dC * if(A < 0) -1 else 1
        for (size_t i = 0; i < dC->len; i++) {
            dA->val[i] += dC->val[i] * (A->val[i] < 0 ? -1 : 1);
        }
    }

    // f(A,B) = A dot B
    // df/dA = B
    // df/dB = A
    static void dot_grad(const T* A, const T* B, const T* C, T* dA, T* dB, const T* dC, int** strideA, int** strideB, int** strideC, int** reps, int** count, size_t* strideLen) {
        // dA += dC * dB
        for (size_t i = 0; i < dA->len; i++) {
            dA->val[i] += dC->val[0] * B->val[i];
        }

        // dB += dC * dA
        for (size_t i = 0; i < dB->len; i++) {
            dB->val[i] += dC->val[0] * A->val[i];
        }
    }

    // f(A,B) = A dot B
    // df/dA = B
    // df/dB = sum A
    static void scalarDot_grad(const T* A, const T* B, const T* C, T* dA, T* dB, const T* dC, int** strideA, int** strideB, int** strideC, int** reps, int** count, size_t* strideLen) {
        // dA += dC * dB
        for (size_t i = 0; i < dA->len; i++) {
            dA->val[i] += dC->val[0] * B->val[0];
        }

        // dB += dC * dA
        for (size_t i = 0; i < dA->len; i++) {
            dB->val[0] += dC->val[0] * A->val[i];
        }
    }
    
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
        // Allocate separate caches for each intermediate result
        size_t cacheLen = max(max(A->len, B->len), dC->len);
        T* cache = new T[cacheLen];
    
        tView<T> A_cache(*A);
        A_cache.val = cache;
    
        tView<T> B_cache(*B);
        B_cache.val = cache;
    
        int* shapeBlock = new int[A->shapeLen * 2 + B->shapeLen * 2];
        // Transposed view of A
        tView<T> A_T(*A);
        A_T.shape = shapeBlock;
        A_T.stride = shapeBlock + A->shapeLen;
        std::copy(A->shape, A->shape + A->shapeLen, A_T.shape);
        std::copy(A->stride, A->stride + A->shapeLen, A_T.stride);
        transp(A_T.shape, A_T.stride, A_T.shapeLen);
    
        // Transposed view of B
        tView<T> B_T(*B);
        B_T.shape = A_T.stride + A->shapeLen;
        B_T.stride = B_T.shape + A->shapeLen;
        std::copy(B->shape, B->shape + B->shapeLen, B_T.shape);
        std::copy(B->stride, B->stride + B->shapeLen, B_T.stride);
        transp(B_T.shape, B_T.stride, B_T.shapeLen);
    
        // dA += dC * B_T
        matmul(dC, &B_T, &A_cache);
        for (size_t i = 0; i < A->len; i++) {
            dA->val[i] += A_cache.val[i];
        }
    
        // dB += A_T * dC
        matmul(&A_T, dC, &B_cache);
        for (size_t i = 0; i < B->len; i++) {
            dB->val[i] += B_cache.val[i];
        }
    
        // Cleanup
        delete[] cache;
        delete[] shapeBlock;
    }
    
    static void batch_matmul_grad(const T* A, const T* B, const T* C, T* dA, T* dB, const T* dC, int** strideA, int** strideB, int** strideC, int** reps, int** count, size_t* strideLen) {
        // allocate memory
        size_t big_len1 = max(dC->shapeLen, B->shapeLen);
        size_t big_len2 = max(A->shapeLen, dC->shapeLen);
    
        int* shapeBlock = new int[A->shapeLen*2 + B->shapeLen*2 + big_len1*2 + big_len2*2];
        int* shapeA = shapeBlock;
        int* effstrideA = shapeA + A->shapeLen;
        int* shapeB = effstrideA + A->shapeLen;
        int* effstrideB = shapeB + B->shapeLen;
    
        int* big_shapeA = effstrideB + B->shapeLen;
        int* big_effstrideA = big_shapeA + big_len1;
    
        int* big_shapeB= big_effstrideA + big_len1;
        int* big_effstrideB = big_shapeB + big_len2;
    
        // transpose A
        copy(A->shape, A->shape + A->shapeLen, shapeA);
        copy(A->stride, A->stride + A->shapeLen, effstrideA);
        
        int temp = shapeA[A->shapeLen - 1];
        shapeA[A->shapeLen - 1] = shapeA[A->shapeLen - 2];
        shapeA[A->shapeLen - 2] = temp;
        temp = effstrideA[A->shapeLen - 1];
        effstrideA[A->shapeLen - 1] = effstrideA[A->shapeLen - 2];
        effstrideA[A->shapeLen - 2] = temp;
    
        tView<T> A_T(shapeA, effstrideA, A->shapeLen, A->val, A->len);
    
    
        // transpose B
        copy(B->shape, B->shape + B->shapeLen, shapeB);
        copy(B->stride, B->stride + B->shapeLen, effstrideB);
        
        temp = shapeB[B->shapeLen - 1];
        shapeB[B->shapeLen - 1] = shapeB[B->shapeLen - 2];
        shapeB[B->shapeLen - 2] = temp;
        temp = effstrideB[B->shapeLen - 1];
        effstrideB[B->shapeLen - 1] = effstrideB[B->shapeLen - 2];
        effstrideB[B->shapeLen - 2] = temp;
    
        tView<T> B_T(shapeB, effstrideB, B->shapeLen, B->val, B->len);
    
    
        // make A_big
        combine_matrix(dC->shape, dC->shapeLen, B_T.shape, B_T.shapeLen, big_shapeA, big_len1);
        size_t lenA = 1;
    
        lenA *= big_shapeA[big_len1 - 1];
        big_effstrideA[big_len1 - 1] = 1;
        for (int i = big_len1 - 2; i >= 0; i--) {
            lenA *= big_shapeA[i];
            big_effstrideA[i] = big_effstrideA[i + 1] * big_shapeA[i + 1];
        }
    
        tView<T> A_big(big_shapeA, big_effstrideA, big_len1, nullptr, lenA);
    
    
        // make B_big
        combine_matrix(A_T.shape, A_T.shapeLen, dC->shape, dC->shapeLen, big_shapeB, big_len2);
        size_t lenB = 1;
    
        lenB *= big_shapeB[big_len2 - 1];
        big_effstrideB[big_len2 - 1] = 1;
        for (int i = big_len2 - 2; i >= 0; i--) {
            lenB *= big_shapeB[i];
            big_effstrideB[i] = big_effstrideB[i + 1] * big_shapeB[i + 1];
        }
    
        tView<T> B_big(big_shapeB, big_effstrideB, big_len2, nullptr, lenB);

        T* cache = new T[max(lenA, lenB)];
        A_big.val = cache;
        B_big.val = cache;
    
    
    
        // dA += dC * B_T
        batch_matmul(dC, &B_T, &A_big);
        flexAdd_inplace(dA, &A_big);
    
        // dB += A_T * dC
        batch_matmul(&A_T, dC, &B_big);
        flexAdd_inplace(dB, &B_big);
    
        delete[] cache;
        delete[] shapeBlock;
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

    // f(A) = A^T
    // df/dA = 1
    static void transp_grad(const T* A, const T* B, const T* C, T* dA, T* dB, const T* dC, int** strideA, int** strideB, int** strideC, int** reps, int** count, size_t* strideLen) {
        for (size_t i = 0; i < dC->len; i++) {
            dA->val[i] += dC->val[i];
        }
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
};
