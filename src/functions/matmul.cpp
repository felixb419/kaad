#include <kaad/functions/matmul.hpp>

#include <kaad/scalar.hpp>              // for Scalar
#include <kaad/tensor/tensor_types.hpp> // for Strides, Shape, ShapeView
#include <kaad/tensor/tensor_view.hpp>  // for TensorViewConst

namespace kaad::functions {

bool Matmul::broadcast(ShapeView lhs, ShapeView rhs, Shape &new_shape) {

    if (lhs.size() != 2 || rhs.size() != 2) {
        return false;
    }

    if (lhs[1] != rhs[0]) {
        return false;
    }

    new_shape = Shape(2);
    new_shape[0] = lhs[0];
    new_shape[1] = rhs[1];

    return true;
}

Matmul::Metadata::Metadata(TensorViewConst lhs, TensorViewConst rhs,
                           TensorViewConst res) {

    this->lhs_rows = lhs.shape[0];
    this->rhs_cols = rhs.shape[1];
    this->shared_dim = lhs.shape[1];

    this->eff_lhs = Strides(lhs.strides);
    this->eff_rhs = Strides(rhs.strides);
    this->eff_res = Strides(res.strides);

    int idx;
    int res_idx;
    int res_offset = 0;
    int res_prev;
    for (int i = 1; i <= 2; i++) {
        idx = 2 - i;

        res_idx = static_cast<int>(res.rank()) - i;
        res_prev = res_offset;
        res_offset +=
            ((res_idx >= 0 ? res.shape[res_idx] : i) - 1) * this->eff_res[idx];
        this->eff_res[idx] -=
            res_prev + (res_idx + 1 < 2 ? this->eff_res[res_idx + 1] : 0);
    }
}

void Matmul::primal(const Scalar *lhs, const Scalar *rhs, Scalar *res,
                    const Metadata &mdata) noexcept {
    const Scalar *row_lhs;
    const Scalar *col_rhs;
    const Scalar *elem_rhs;
    for (int lhs_idx = 0; lhs_idx < mdata.lhs_rows;
         lhs_idx++, lhs += mdata.eff_lhs[0], res += mdata.eff_res[0]) {

        col_rhs = rhs;

        for (int rhs_idx = 0; rhs_idx < mdata.rhs_cols;
             rhs_idx++, col_rhs += mdata.eff_rhs[1], res += mdata.eff_res[1]) {

            row_lhs = lhs;
            elem_rhs = col_rhs;

            for (int i = 0; i < mdata.shared_dim; i++,
                     row_lhs += mdata.eff_lhs[1],
                     elem_rhs += mdata.eff_rhs[0]) {

                *res += (*row_lhs) * (*elem_rhs);
            }
        }
    }
}

void Matmul::adjoint(const Scalar *lhs, Scalar *d_lhs, const Scalar *rhs,
                     Scalar *d_rhs, const Scalar *d_res,
                     const Metadata &wrt_lhs,
                     const Metadata &wrt_rhs) noexcept {
    // d_res * rhs^T = d_lhs
    primal(d_res, rhs, d_lhs, wrt_lhs);

    // lhs^T * d_res = d_rhs
    primal(lhs, d_res, d_rhs, wrt_rhs);
}

} // namespace kaad::functions
