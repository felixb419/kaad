#include "kaad/graph/operators/internal/kernels.hpp"
#include "kaad/graph/operators/internal/safe_kernels.hpp"

#include <kaad/scalar.hpp>

int main() {

    kaad::operations::kernels::Add<kaad::Scalar> add_kernel;
    kaad::operations::kernels::Sub<kaad::Scalar> sub_kernel;
    kaad::operations::kernels::Mul<kaad::Scalar> mul_kernel;
    kaad::operations::kernels::Div<kaad::Scalar> div_kernel;
    kaad::operations::kernels::Pow<kaad::Scalar> pow_kernel;
    kaad::operations::kernels::Dot<kaad::Scalar> dot_kernel;
    kaad::operations::kernels::Min<kaad::Scalar> min_kernel;
    kaad::operations::kernels::Max<kaad::Scalar> max_kernel;
    kaad::operations::kernels::NoOp<kaad::Scalar> noop_kernel;
    kaad::operations::kernels::Neg<kaad::Scalar> neg_kernel;
    kaad::operations::kernels::Square<kaad::Scalar> square_kernel;
    kaad::operations::kernels::Sqrt<kaad::Scalar> sqrt_kernel;
    kaad::operations::kernels::Log<kaad::Scalar> log_kernel;
    kaad::operations::kernels::Exp<kaad::Scalar> exp_kernel;
    kaad::operations::kernels::Abs<kaad::Scalar> abs_kernel;

    kaad::operations::kernels::SafeDiv<kaad::Scalar> safe_div_kernel;
    kaad::operations::kernels::SafePow<kaad::Scalar> safe_pow_kernel;
    kaad::operations::kernels::SafeSqrt<kaad::Scalar> safe_sqrt_kernel;
    kaad::operations::kernels::SafeLog<kaad::Scalar> safe_log_kernel;
    kaad::operations::kernels::SafeExp<kaad::Scalar> safe_exp_kernel;

    kaad::Scalar value;
    value = kaad::operations::kernels::EPSILON<kaad::Scalar>;
    value = kaad::operations::kernels::MAX_FINITE<kaad::Scalar>;
    value = kaad::operations::kernels::MIN_FINITE<kaad::Scalar>;
    value = kaad::operations::kernels::MAX_EXP<kaad::Scalar>;
    value = kaad::operations::kernels::MIN_EXP<kaad::Scalar>;

    return 0;
}
