#pragma once

#include "dispatchers.h"
#include "gradients.h"  // for Gradients, binaryGrad, flexBinaryGrad
#include "kernels.h"    // for Kernels, Kernels::NullOp
#include "operations.h" // for Operations, binaryOp, flexBinaryOp
#include "strides.h"    // for Strides
#include "tensor.h"     // for Tensor (ptr only), print_arr, combine_flexible
#include <algorithm>    // for equal, max
#include <cstddef>
#include <memory>    // for make_unique
#include <sstream>   // for operator<<, basic_ostream, char_traits, ostr...
#include <stddef.h>  // for size_t
#include <stdexcept> // for std::invalid_argument

namespace kaad {
template <typename T, class Kernel> struct Node_binary;
template <typename T, class Kernel> struct Node_binary_flex;
template <typename T> struct CompGraph;
template <typename T> struct INode;
template <typename T> struct Node_batch_matmul;
template <typename T> struct Node_matmul;

template <typename T, class Kernel> struct BinaryKernels {
	using Op = class Kernel::Op;
	using Grad = class Kernel::Grad;

	binaryOp<T, Op> scalarOpRhs = Operations::scalarRhs<T, Op>;
	binaryOp<T, Op> scalarOpLhs = Operations::scalarLhs<T, Op>;
	binaryOp<T, Op> pointOp = Operations::pointwise<T, Op>;
	flexBinaryOp<T, Op> flexOp = Operations::flexible<T, Op>;

	binaryGrad<T, Grad> scalarGradRhs = Gradients::scalarRhs<T, Grad>;
	binaryGrad<T, Grad> scalarGradLhs = Gradients::scalarLhs<T, Grad>;
	binaryGrad<T, Grad> pointGrad = Gradients::pointwise<T, Grad>;
	flexBinaryGrad<T, Grad> flexGrad = Gradients::flexible<T, Grad>;
};

template <typename T, class Kernel>
INode<T> *binOperator(CompGraph<T> &rec, INode<T> *A_ptr, INode<T> *B_ptr,
                      const BinaryKernels<T, Kernel> kernels,
                      const char *opName) {
	int recLen = rec.nodes.size();
	Tensor<T> &A = A_ptr->value;
	Tensor<T> &B = B_ptr->value;
	bool A_scalar = A.nDims == 1 && A.shape[0] == 1;
	bool B_scalar = B.nDims == 1 && B.shape[0] == 1;

	size_t newLen = std::max(A.nDims, B.nDims);
	int *newShape = new int[newLen];

	if (B_scalar) {
		std::copy(A.shape, A.shape + A.nDims, newShape);

		auto newNode = std::make_unique<Node_binary<T, Kernel>>(
		    kernels.scalarOpRhs, kernels.scalarGradRhs, A_ptr, B_ptr, newShape,
		    A.nDims);
		newNode->len = newNode->value.len;
		rec.nodes.push_back(move(newNode));
	} else if (A_scalar) {
		std::copy(B.shape, B.shape + B.nDims, newShape);

		auto newNode = std::make_unique<Node_binary<T, Kernel>>(
		    kernels.scalarOpLhs, kernels.scalarGradLhs, A_ptr, B_ptr, newShape,
		    B.nDims);
		newNode->len = newNode->value.len;
		rec.nodes.push_back(move(newNode));
	} else if (A.nDims == B.nDims &&
	           std::equal(A.shape, A.shape + A.nDims, B.shape)) {
		std::copy(A.shape, A.shape + A.nDims, newShape);

		auto newNode = std::make_unique<Node_binary<T, Kernel>>(
		    kernels.pointOp, kernels.pointGrad, A_ptr, B_ptr, newShape,
		    A.nDims);
		newNode->len = newNode->value.len;
		rec.nodes.push_back(move(newNode));
	} else if (combine_flexible(A.shape, A.nDims, B.shape, B.nDims, newShape,
	                            newLen)) {
		using Op = typename Kernel::Op;
		using Grad = typename Kernel::Grad;
		flexBinaryOp<T, Op> operation = kernels.flexOp;
		flexBinaryGrad<T, Grad> gradient = kernels.flexGrad;
		if (newLen <= KAAD_MAX_NDIMS) {
			operation = get_flexOp_dispatcher<T, Op>()[newLen];
			gradient = get_flexGrad_dispatcher<T, Grad>()[newLen];
		}

		auto newNode = std::make_unique<Node_binary_flex<T, Kernel>>(
		    operation, gradient, A_ptr, B_ptr, newShape, newLen);
		Strides::flexible_binary<T>(A, B, *newNode.get());
		rec.nodes.push_back(move(newNode));
	} else {
		std::ostringstream errmsg;
		errmsg << "shape error in node[" << recLen << "] (" << opName
		       << "), tensor shapes are not broadcastable (A.shape=";
		print_arr(A.shape, A.shape + A.nDims, errmsg);
		errmsg << ", B.shape=";
		print_arr(B.shape, B.shape + B.nDims, errmsg);
		errmsg << ")";
		throw std::invalid_argument(errmsg.str());
	}
	return rec.nodes[recLen].get();
}

// add A and B
// where A and B are Tensors with Broadcastable shapes
template <typename T>
INode<T> *add(CompGraph<T> &rec, INode<T> *A_ptr, INode<T> *B_ptr) {
	static const BinaryKernels<T, class Kernels::Add<T>> addK;
	return binOperator(rec, A_ptr, B_ptr, addK, "add");
}

// subtract B from A
// where A and B are Tensors with Broadcastable shapes
template <typename T>
INode<T> *sub(CompGraph<T> &rec, INode<T> *A_ptr, INode<T> *B_ptr) {
	static const BinaryKernels<T, class Kernels::Sub<T>> subK;
	return binOperator(rec, A_ptr, B_ptr, subK, "sub");
}

// multiply A and B
// where A and B are Tensors with Broadcastable shapes
template <typename T>
INode<T> *mul(CompGraph<T> &rec, INode<T> *A_ptr, INode<T> *B_ptr) {
	static const BinaryKernels<T, class Kernels::Mul<T>> mulK;
	return binOperator(rec, A_ptr, B_ptr, mulK, "mul");
}

// divide A by B
// where A and B are Tensors with Broadcastable shapes
template <typename T>
INode<T> *div(CompGraph<T> &rec, INode<T> *A_ptr, INode<T> *B_ptr) {
	static const BinaryKernels<T, class Kernels::Div<T>> divK;
	return binOperator(rec, A_ptr, B_ptr, divK, "div");
}

// raise A to the power of B
// where A and B are Tensors with Broadcastable shapes
template <typename T>
INode<T> *pow(CompGraph<T> &rec, INode<T> *A_ptr, INode<T> *B_ptr) {
	static const BinaryKernels<T, class Kernels::Pow<T>> powK;
	return binOperator(rec, A_ptr, B_ptr, powK, "pow");
}

// compute dot prodcut of A and B
// where A and B are scalars or vectors with the same length
template <typename T>
INode<T> *dot(CompGraph<T> &rec, INode<T> *A_ptr, INode<T> *B_ptr) {
	using Op = class Kernels::NullOp::Op;
	using Grad = class Kernels::NullOp::Grad;
	binaryOp<T, Op> scalar = Operations::scalarDot<T, Op>;
	binaryGrad<T, Grad> scalar_grad = Gradients::scalarDot<T, Grad>;
	binaryOp<T, Op> dot = Operations::dot<T, Op>;
	binaryGrad<T, Grad> dot_grad = Gradients::dot<T, Grad>;

	int recLen = rec.nodes.size();
	Tensor<T> &A = A_ptr->value;
	Tensor<T> &B = B_ptr->value;

	bool A_scalar = A.nDims == 1 && A.shape[0] == 1;
	bool B_scalar = B.nDims == 1 && B.shape[0] == 1;
	if (B_scalar) {
		auto newNode = std::make_unique<Node_binary<T, Kernels::NullOp>>(
		    scalar, scalar_grad, A_ptr, B_ptr, ((T)0));

		newNode->len = A.len;
		rec.nodes.push_back(move(newNode));
	} else if (A_scalar) {
		auto newNode = std::make_unique<Node_binary<T, Kernels::NullOp>>(
		    scalar, scalar_grad, B_ptr, A_ptr, ((T)0));
		newNode->len = B.len;
		rec.nodes.push_back(move(newNode));
	} else if (A.nDims == B.nDims &&
	           std::equal(A.shape, A.shape + A.nDims, B.shape)) {
		auto newNode = std::make_unique<Node_binary<T, Kernels::NullOp>>(
		    dot, dot_grad, A_ptr, B_ptr, ((T)0));
		newNode->len = A.len;
		rec.nodes.push_back(move(newNode));

	} else {
		std::ostringstream errmsg;
		errmsg << "shape error in node[" << recLen
		       << "] (dot), tensor shapes arent valid for dot product (shape1=";
		print_arr(A.shape, A.shape + A.nDims, errmsg);
		errmsg << ", shape2=";
		print_arr(B.shape, B.shape + B.nDims, errmsg);
		errmsg << ")";
		throw std::invalid_argument(errmsg.str());
	}

	return rec.nodes.back().get();
}

// matrix multiply A and B
// where A and B are Tensors with valid dimensions
template <typename T>
INode<T> *matmul(CompGraph<T> &rec, INode<T> *A_ptr, INode<T> *B_ptr) {
	int recLen = rec.nodes.size();
	Tensor<T> &A = A_ptr->value;
	Tensor<T> &B = B_ptr->value;

	size_t newLen = std::max(A.nDims, B.nDims);
	int *newShape = new int[newLen];

	const char *opName = newLen == 2 ? "matmul" : "batch_matmul";
	if (!combine_matrix(A.shape, A.nDims, B.shape, B.nDims, newShape, newLen)) {
		std::ostringstream errmsg;
		errmsg << "shape error in node[" << recLen << "] (" << opName
		       << "), tensor shapes arent valid for " << opName << " (shape1=";
		print_arr(A.shape, A.shape + A.nDims, errmsg);
		errmsg << ", shape2=";
		print_arr(B.shape, B.shape + B.nDims, errmsg);
		errmsg << ")";
		throw std::invalid_argument(errmsg.str());
	}

	if (newLen == 2) {
		auto newNode = std::make_unique<Node_matmul<T>>(
		    Operations::matmul<T>, Gradients::matmul<T>, A_ptr, B_ptr, newShape,
		    newLen);
		Strides::matmul<T>(A, B, *newNode.get());
		rec.nodes.push_back(move(newNode));
	} else {
		batchmatmulOp<T> operation = Operations::batch_matmul<T>;
		batchmatmulGrad<T> gradient = Gradients::batch_matmul<T>;
		if (newLen <= KAAD_MAX_NDIMS) {
			operation = get_batch_matmul_dispatcher<T>()[newLen];
			gradient = get_batch_matmul_grad_dispatcher<T>()[newLen];
		}

		auto newNode = std::make_unique<Node_batch_matmul<T>>(
		    operation, gradient, A_ptr, B_ptr, newShape, newLen);
		Strides::batch_matmul<T>(A, B, *newNode.get());
		rec.nodes.push_back(move(newNode));
	}

	return rec.nodes.back().get();
}

// compute outer product of A and B
// where A and B are Tensors
template <typename T>
INode<T> *outer(CompGraph<T> &rec, INode<T> *A_ptr, INode<T> *B_ptr) {
	int recLen = rec.nodes.size();
	Tensor<T> &A = A_ptr->value;
	Tensor<T> &B = B_ptr->value;

	size_t newLen = A.nDims + B.nDims;
	int *newShape = new int[newLen];
	std::copy(A.shape, A.shape + A.nDims, newShape);
	std::copy(B.shape, B.shape + B.nDims, newShape + A.nDims);

	using Kernel = typename Kernels::Mul<T>;
	using Op = typename Kernel::Op;
	using Grad = typename Kernel::Grad;

	auto newNode = std::make_unique<Node_binary_flex<T, Kernel>>(
	    Operations::flexible<T, Op>, Gradients::flexible<T, Grad>, A_ptr, B_ptr,
	    newShape, newLen);
	auto raw = newNode.get();
	Strides::outer<T>(A, B, *raw);
	rec.nodes.push_back(move(newNode));

	return raw;
}

// compute pointwise minimum of A and B
// where A and B are Tensors with broadcastable shapes
template <typename T>
INode<T> *min(CompGraph<T> &rec, INode<T> *A_ptr, INode<T> *B_ptr) {
	static const BinaryKernels<T, class Kernels::Min<T>> minK;
	return binOperator(rec, A_ptr, B_ptr, minK, "minimum");
}

// compute pointwise maximum of A and B
// where A and B are Tensors with broadcastable shapes
template <typename T>
INode<T> *max(CompGraph<T> &rec, INode<T> *A_ptr, INode<T> *B_ptr) {
	static const BinaryKernels<T, class Kernels::Max<T>> maxK;
	return binOperator(rec, A_ptr, B_ptr, maxK, "minimum");
}
} // namespace kaad
