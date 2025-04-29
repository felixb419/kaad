#pragma once

#include "tensor.h"
#include "operations.h"

#include <unordered_map>
#include <vector>

#define safe_powergradient

template <typename T>
struct Gradients : Operations<T> {
    using Operations<T>::flexAdd_inplace;
    using Operations<T>::matmul;
    using Operations<T>::batch_matmul;

    // f(A,B) = A + B
    // df/dA = 1
    // df/dB = 1
    static void pointAdd_grad(const tView<T>* seed, tView<T>* in1, tView<T>* d_in1, tView<T>* in2, tView<T>* d_in2, tView<T>* res) {
        // d_in1 += seed
        for (size_t i = 0; i < seed->len; i++) {
            d_in1->val[i] += seed->val[i];
        }
        
        // d_in2 += seed
        for (size_t i = 0; i < seed->len; i++) {
            d_in2->val[i] += seed->val[i];
        }
    }
    static void scalarAdd_grad(const tView<T>* seed, tView<T>* in1, tView<T>* d_in1, tView<T>* in2, tView<T>* d_in2, tView<T>* res) {
        // in2 has shape = (1,)
        // d_in1 += seed
        for (size_t i = 0; i < seed->len; i++) {
            d_in1->val[i] += seed->val[i];
        }
        
        // d_in2 += seed
        for (size_t i = 0; i < seed->len; i++) {
            d_in2->val[0] += seed->val[i];
        }
    }
    static void flexAdd_grad(const tView<T>* seed, tView<T>* in1, tView<T>* d_in1, tView<T>* in2, tView<T>* d_in2, tView<T>* res) {
        // d_in1 += seed
        flexAdd_inplace(d_in1, seed);
    
        // d_in2 += seed
        flexAdd_inplace(d_in2, seed);
    }
    
    // f(A,B) = A - B
    // df/dA = 1
    // df/dB = -1
    static void pointSub_grad(const tView<T>* seed, tView<T>* in1, tView<T>* d_in1, tView<T>* in2, tView<T>* d_in2, tView<T>* res) {
        // d_in1 += seed
        for (size_t i = 0; i < seed->len; i++) {
            d_in1->val[i] += seed->val[i];
        }
    
        // d_in2 -= seed
        for (size_t i = 0; i < seed->len; i++) {
            d_in2->val[i] -= seed->val[i];
        }
    }
    static void scalarSub_grad(const tView<T>* seed, tView<T>* in1, tView<T>* d_in1, tView<T>* in2, tView<T>* d_in2, tView<T>* res) {
        // in2 has shape = (1,)
        // d_in1 += seed
        for (size_t i = 0; i < seed->len; i++) {
            d_in1->val[i] += seed->val[i];
        }
    
        // d_in2 -= seed
        for (size_t i = 0; i < seed->len; i++) {
            d_in2->val[0] -= seed->val[i];
        }
    }
    static void invScalarSub_grad(const tView<T>* seed, tView<T>* in1, tView<T>* d_in1, tView<T>* in2, tView<T>* d_in2, tView<T>* res) {
        // in1 has shape = (1,)
        // d_in1 += seed
        for (size_t i = 0; i < seed->len; i++) {
            d_in1->val[0] += seed->val[i];
        }
    
        // d_in2 -= seed
        for (size_t i = 0; i < seed->len; i++) {
            d_in2->val[i] -= seed->val[i];
        }
    }
    static void flexSub_grad(const tView<T>* seed, tView<T>* in1, tView<T>* d_in1, tView<T>* in2, tView<T>* d_in2, tView<T>* res) {
        // d_in1 += seed
        flexAdd_inplace(d_in1, seed);
    
        // d_in2 -= seed
        flexSub_inplace(d_in2, seed);
    }
    
    // f(A,B) = A * B
    // df/dA = B
    // df/dB = A
    static void pointMul_grad(const tView<T>* seed, tView<T>* in1, tView<T>* d_in1, tView<T>* in2, tView<T>* d_in2, tView<T>* res) {
        // d_in1 += seed * in2
        for (size_t i = 0; i < seed->len; i++) {
            d_in1->val[i] += seed->val[i] * in2->val[i];
        }
    
        // d_in2 += seed * in1
        for (size_t i = 0; i < seed->len; i++) {
            d_in2->val[i] += seed->val[i] * in1->val[i];
        }
    }
    static void scalarMul_grad(const tView<T>* seed, tView<T>* in1, tView<T>* d_in1, tView<T>* in2, tView<T>* d_in2, tView<T>* res) {
        // in2 has shape = (1,)
        // d_in1 += seed * in2
        for (size_t i = 0; i < seed->len; i++) {
            d_in1->val[i] += seed->val[i] * in2->val[0];
        }
    
        // d_in2 += seed * in1
        for (size_t i = 0; i < seed->len; i++) {
            d_in2->val[0] += seed->val[i] * in1->val[i];
        }
    }
    static void flexMul_grad(const tView<T>* seed, tView<T>* in1, tView<T>* d_in1, tView<T>* in2, tView<T>* d_in2, tView<T>* res) {
        T* cache = new T[seed->len];
        tView<T> seed_cache = *seed;
        seed_cache.val = cache;
    
        // d_in1 += seed * in2
        flexMul(seed, in2, &seed_cache);
        flexAdd_inplace(d_in1, &seed_cache);
    
        // d_in2 += seed * in1
        flexMul(seed, in1, &seed_cache);
        flexAdd_inplace(d_in2, &seed_cache);
        
        delete[] cache;
    }
    
    static void pointDiv_grad(const tView<T>* seed, tView<T>* in1, tView<T>* d_in1, tView<T>* in2, tView<T>* d_in2, tView<T>* res) {
        // d_in1 += seed * (1 / in2)
        for (size_t i = 0; i < seed->len; i++) {
            d_in1->val[i] += seed->val[i] * (1 / in2->val[i]);
        }
    
        // d_in2 -= seed * (in1 / in2^2)
        for (size_t i = 0; i < seed->len; i++) {
            d_in2->val[i] -= seed->val[i] * (in1->val[i] / (in2->val[i] * in2->val[i]));
        }
    }
    static void scalarDiv_grad(const tView<T>* seed, tView<T>* in1, tView<T>* d_in1, tView<T>* in2, tView<T>* d_in2, tView<T>* res) {
        // in2 has shape = (1,)
        // d_in1 += seed * (1 / in2)
        T in2_inv = 1 / in2->val[0];
        for (size_t i = 0; i < seed->len; i++) {
            d_in1->val[i] += seed->val[i] * in2_inv;
        }
    
        // d_in2 -= seed * (in1 / in2^2)
        T in2_sqr = in2->val[0] * in2->val[0];
        for (size_t i = 0; i < seed->len; i++) {
            d_in2->val[0] -= seed->val[i] * (in1->val[i] / in2_sqr);
        }
    }
    static void invScalarDiv_grad(const tView<T>* seed, tView<T>* in1, tView<T>* d_in1, tView<T>* in2, tView<T>* d_in2, tView<T>* res) {
        // in1 has shape = (1,)
        // d_in1 += seed * (1 / in2)
        T in2_inv = 1 / in2->val[0];
        for (size_t i = 0; i < seed->len; i++) {
            d_in1->val[0] += seed->val[i] * (1 / in2->val[i]);
        }
    
        // d_in2 -= seed * (in1 / in2^2)
        for (size_t i = 0; i < seed->len; i++) {
            d_in2->val[i] -= seed->val[i] * (in1->val[0] / (in2->val[i] * in2->val[i]));
        }
    }
    static void flexDiv_grad(const tView<T>* seed, tView<T>* in1, tView<T>* d_in1, tView<T>* in2, tView<T>* d_in2, tView<T>* res) {
        T* cache = new T[seed->len];
        tView<T> seed_cache = *seed;
        seed_cache.val = cache;
        tView<T> in2_cache = *in2;
        in2_cache.val = cache;
    
        // d_in1 += seed * (1 / in2)
        for (size_t i = 0; i < in2->len; i++) {
            cache[i] = 1 / in2->val[i];
        }
        flexMul(seed, &in2_cache, &seed_cache);
        flexAdd_inplace(d_in1, &seed_cache);
    
        // d_in2 -= seed * (in1 / in2^2)
        for (size_t i = 0; i < in2->len; i++) {
            cache[i] = in2->val[i] * in2->val[i];
        }
        flexDiv(in1, &in2_cache, &seed_cache);
        for (size_t i = 0; i < seed->len; i++) {
            cache[i] *= seed->val[i];
        }
        flexSub_inplace(d_in2, &seed_cache);
    
        delete[] cache;
    }
    
    // f(A,B) = A ^ B
    // df/dA = B * A ^ (B - 1)
    // df/dB = A ^ B * log(|A|)     df/dB is 0 if A is negative for stability
    static void pointPow_grad(const tView<T>* seed, tView<T>* in1, tView<T>* d_in1, tView<T>* in2, tView<T>* d_in2, tView<T>* res) {
        // d_in1 += seed * (in2 * (in1^(in2 - 1)))
        for (size_t i = 0; i < seed->len; i++) {
            d_in1->val[i] += seed->val[i] * (in2->val[i] * pow(in1->val[i], in2->val[i] - 1));
        }
    
        // d_in2 += seed * (res * log(in1))
        for (size_t i = 0; i < seed->len; i++) {
            #ifdef safe_powergradient
            d_in2->val[i] += seed->val[i] * (res->val[i] * (in1->val[i] < 0 ? 0 : log(in1->val[i])));
            #else
            d_in2->val[i] += seed->val[i] * (res->val[i] * log(in1->val[i]));
            #endif
        }
    }
    static void scalarPow_grad(const tView<T>* seed, tView<T>* in1, tView<T>* d_in1, tView<T>* in2, tView<T>* d_in2, tView<T>* res) {
        // in2 has shape = (1,)
        // d_in1 += seed * (in2 * (in1^(in2 - 1)))
        for (size_t i = 0; i < seed->len; i++) {
            d_in1->val[i] += seed->val[i] * (in2->val[0] * pow(in1->val[i], in2->val[0] - 1));
        }
    
        // d_in2 += seed * (res * log(in1))
        for (size_t i = 0; i < seed->len; i++) {
            #ifdef safe_powergradient
            d_in2->val[0] += seed->val[i] * (in1->val[i] < 0 ? 0 : res->val[i] * log(in1->val[i]));
            #else
            d_in2->val[0] += seed->val[i] * (res->val[i] * log(in1->val[i]));
            #endif
        }
    }
    static void invScalarPow_grad(const tView<T>* seed, tView<T>* in1, tView<T>* d_in1, tView<T>* in2, tView<T>* d_in2, tView<T>* res) {
        // in1 has shape = (1,)
        // d_in1 += seed * (in2 * (in1^(in2 - 1)))
        for (size_t i = 0; i < seed->len; i++) {
            d_in1->val[0] += seed->val[i] * (in2->val[i] * pow(in1->val[0], in2->val[i] - 1));
        }
    
        // d_in2 += seed * (res * log(in1))
        T in1_log = in1->val[0] < 0 ? 0 : log(in1->val[0]);
        for (size_t i = 0; i < seed->len; i++) {
            #ifdef safe_powergradient
            d_in2->val[i] += seed->val[i] * (res->val[i] * in1_log);
            #else
            d_in2->val[i] += seed->val[i] * (res->val[i] * in1_log);
            #endif
        }
    }
    static void flexPow_grad(const tView<T>* seed, tView<T>* in1, tView<T>* d_in1, tView<T>* in2, tView<T>* d_in2, tView<T>* res) {
        T* cache = new T[seed->len + seed->len];
        // alt cache to astatic void cache conflict in flexible operation
        T* cache2 = cache + seed->len;
        tView<T> seed_cache = *seed;
        seed_cache.val = cache;
    
        tView<T> in1_cache = *in1;
        in1_cache.val = cache2;
        tView<T> in2_cache = *in2;
        in2_cache.val = cache2;
    
        // d_in1 += seed * (in2 * (in1^(in2 - 1)))
        for (size_t i = 0; i < in2->len; i++) {
            cache2[i] = in2->val[i] - 1;
        }
        flexPow(in1, &in2_cache, &seed_cache);
        flexMul_inplace(&seed_cache, in2, false);
        for (size_t i = 0; i < seed->len; i++) {
            cache[i] *= seed->val[i];
        }
        flexAdd_inplace(d_in1, &seed_cache);
    
        // d_in2 += seed * (res * log(in1))
        for (size_t i = 0; i < in1->len; i++) {
            #ifdef safe_powergradient
            cache2[i] = in1->val[i] < 0 ? 0 : log(in1->val[i]);
            #else
            cache2[i] = log(in1->val[i]);
            #endif
        }
        flexMul(res, &in1_cache, &seed_cache);
        for (size_t i = 0; i < in1->len; i++) {
            seed_cache.val[i] *= seed->val[i];
        }
        flexAdd_inplace(d_in2, &seed_cache);
    
        delete[] cache;
    }
    
    // f(A) = -A
    // df/dA = -1
    static void negate_grad(const tView<T>* seed, tView<T>* in1, tView<T>* d_in1, tView<T>* in2, tView<T>* d_in2, tView<T>* res) {
        // d_in1 -= seed
        flexSub_inplace(d_in1, seed);
    }
    
    // f(A) = A^2
    // df/dA = 2A
    static void square_grad(const tView<T>* seed, tView<T>* in1, tView<T>* d_in1, tView<T>* in2, tView<T>* d_in2, tView<T>* res) {
        // d_in1 += seed * (in1 * 2)
        for (size_t i = 0; i < seed->len; i++) {
            d_in1->val[i] += seed->val[i] * (in1->val[i] * 2);
        }
    }
    
    // f(A) = sqrt(A)
    // df/dA = 1 / (2 * sqrt(A))
    static void sqrt_grad(const tView<T>* seed, tView<T>* in1, tView<T>* d_in1, tView<T>* in2, tView<T>* d_in2, tView<T>* res) {
        // d_in1 += seed / (2 * res)
        for (size_t i = 0; i < seed->len; i++) {
            d_in1->val[i] += seed->val[i] / (2 * res->val[i]);
        }
    }
    
    // f(A) = log(A)
    // df/dA = 1 / x
    static void log_grad(const tView<T>* seed, tView<T>* in1, tView<T>* d_in1, tView<T>* in2, tView<T>* d_in2, tView<T>* res) {
        // d_in1 += seed / x
        for (size_t i = 0; i < seed->len; i++) {
            d_in1->val[i] += seed->val[i] / in1->val[i];
        }
    }
    
    // f(A) = e^A
    // df/dA = e^A
    static void exp_grad(const tView<T>* seed, tView<T>* in1, tView<T>* d_in1, tView<T>* in2, tView<T>* d_in2, tView<T>* res) {
        // d_in1 += seed * res
        for (size_t i = 0; i < seed->len; i++) {
            d_in1->val[i] += seed->val[i] * res->val[i];
        }
    }
    
    // f(A) = |A|
    // df/dA = |A| / A       (if (A[i] < 0) -1 else 1)
    static void abs_grad(const tView<T>* seed, tView<T>* in1, tView<T>* d_in1, tView<T>* in2, tView<T>* d_in2, tView<T>* res) {
        // d_in1 += seed * if(A < 0) -1 else 1
        for (size_t i = 0; i < seed->len; i++) {
            d_in1->val[i] += seed->val[i] * (in1->val[i] < 0 ? -1 : 1);
        }
    }
    
    // gradient for f(A,B) = AB
    // dC/dA = B^T
    // dC/dB = A^T
    static void matmul_grad(const tView<T>* seed, tView<T>* in1, tView<T>* d_in1, tView<T>* in2, tView<T>* d_in2, tView<T>* res) {
        T* cache = new T[seed->len];
        tView<T> in1_cache = *in1;
        in1_cache.val = cache;
        tView<T> in2_cache = *in2;
        in2_cache.val = cache;
    
        tView<T> in1_T = *in1;
        transp(in1_T.shape, in1_T.stride, in1_T.shapeLen);
        tView<T> in2_T = *in2;
        transp(in2_T.shape, in2_T.stride, in2_T.shapeLen);
        
        // d_in1 += seed * in2_T
        matmul(seed, &in2_T, &in1_cache);
        for (size_t i = 0; i < in1->len; i++) {
            d_in1->val[i] += in1_cache.val[i];
        }
    
        // d_in2 += in1_T * seed
        matmul(&in1_T, seed, &in2_cache);
        for (size_t i = 0; i < in2->len; i++) {
            d_in2->val[i] += in2_cache.val[i];
        }
    }
    static void batch_matmul_grad(const tView<T>* seed, tView<T>* in1, tView<T>* d_in1, tView<T>* in2, tView<T>* d_in2, tView<T>* res) {
        // allocate memory
        T* cache = new T[seed->len];
    
        size_t big_len1 = max(seed->shapeLen, in2->shapeLen);
        size_t big_len2 = max(in1->shapeLen, seed->shapeLen);
    
        int* shapeBlock = new int[in1->shapeLen*2 + in2->shapeLen*2 + big_len1*2 + big_len2*2];
        int* shape1 = shapeBlock;
        int* stride1 = shape1 + in1->shapeLen;
        int* shape2 = stride1 + in1->shapeLen;
        int* stride2 = shape2 + in2->shapeLen;
    
        int* big_shape1 = stride2 + in2->shapeLen;
        int* big_stride1 = big_shape1 + big_len1;
    
        int* big_shape2= big_stride1 + big_len1;
        int* big_stride2 = big_shape2 + big_len2;
    
        // transpose in1
        copy(in1->shape, in1->shape + in1->shapeLen, shape1);
        copy(in1->stride, in1->stride + in1->shapeLen, stride1);
        
        int temp = shape1[in1->shapeLen - 1];
        shape1[in1->shapeLen - 1] = shape1[in1->shapeLen - 2];
        shape1[in1->shapeLen - 2] = temp;
        temp = stride1[in1->shapeLen - 1];
        stride1[in1->shapeLen - 1] = stride1[in1->shapeLen - 2];
        stride1[in1->shapeLen - 2] = temp;
    
        tView<T> in1_T(shape1, stride1, in1->shapeLen, in1->val, in1->len);
    
    
        // transpose in2
        copy(in2->shape, in2->shape + in2->shapeLen, shape2);
        copy(in2->stride, in2->stride + in2->shapeLen, stride2);
        
        temp = shape2[in2->shapeLen - 1];
        shape2[in2->shapeLen - 1] = shape2[in2->shapeLen - 2];
        shape2[in2->shapeLen - 2] = temp;
        temp = stride2[in2->shapeLen - 1];
        stride2[in2->shapeLen - 1] = stride2[in2->shapeLen - 2];
        stride2[in2->shapeLen - 2] = temp;
    
        tView<T> in2_T(shape2, stride2, in2->shapeLen, in2->val, in2->len);
    
    
        // make in1_big
        combine_matrix(seed->shape, seed->shapeLen, in2_T.shape, in2_T.shapeLen, big_shape1, big_len1);
        size_t valLen = 1;
    
        valLen *= big_shape1[big_len1 - 1];
        big_stride1[big_len1 - 1] = 1;
        for (int i = big_len1 - 2; i >= 0; i--) {
            valLen *= big_shape1[i];
            big_stride1[i] = big_stride1[i + 1] * big_shape1[i + 1];
        }
    
        tView<T> in1_big(big_shape1, big_stride1, big_len1, cache, valLen);
    
    
        // make in2_big
        combine_matrix(in1_T.shape, in1_T.shapeLen, seed->shape, seed->shapeLen, big_shape2, big_len2);
        valLen = 1;
    
        valLen *= big_shape2[big_len2 - 1];
        big_stride2[big_len2 - 1] = 1;
        for (int i = big_len2 - 2; i >= 0; i--) {
            valLen *= big_shape2[i];
            big_stride2[i] = big_stride2[i + 1] * big_shape2[i + 1];
        }
    
        tView<T> in2_big(big_shape2, big_stride2, big_len2, cache, valLen);
    
    
    
    
        // d_in1 += seed * in2_T
        batch_matmul(seed, &in2_T, &in1_big);
        flexAdd_inplace(d_in1, &in1_big);
    
        // d_in2 += in1_T * seed
        batch_matmul(&in1_T, seed, &in2_big);
        flexAdd_inplace(d_in2, &in2_big);
    
        delete[] cache;
        delete[] shapeBlock;
    }
};
