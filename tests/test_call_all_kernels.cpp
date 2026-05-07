#include "../src/operations/safe_kernels.hpp"
#include <kaad/operations/kernels.hpp> // for Kernls
#include <kaad/scalar.hpp>             // for kaad::Scalar

int main() {

    kaad::Kernels::Add<kaad::Scalar> add_kernel;
    kaad::Kernels::Sub<kaad::Scalar> sub_kernel;
    kaad::Kernels::Mul<kaad::Scalar> mul_kernel;
    kaad::Kernels::Div<kaad::Scalar> div_kernel;
    kaad::Kernels::Pow<kaad::Scalar> pow_kernel;
    kaad::Kernels::Dot<kaad::Scalar> dot_kernel;
    kaad::Kernels::Min<kaad::Scalar> min_kernel;
    kaad::Kernels::Max<kaad::Scalar> max_kernel;
    kaad::Kernels::NoOp<kaad::Scalar> noop_kernel;
    kaad::Kernels::Neg<kaad::Scalar> neg_kernel;
    kaad::Kernels::Square<kaad::Scalar> square_kernel;
    kaad::Kernels::Sqrt<kaad::Scalar> sqrt_kernel;
    kaad::Kernels::Log<kaad::Scalar> log_kernel;
    kaad::Kernels::Exp<kaad::Scalar> exp_kernel;
    kaad::Kernels::Abs<kaad::Scalar> abs_kernel;

    kaad::Kernels::SafeDiv<kaad::Scalar> safe_div_kernel;
    kaad::Kernels::SafePow<kaad::Scalar> safe_pow_kernel;
    kaad::Kernels::SafeSqrt<kaad::Scalar> safe_sqrt_kernel;
    kaad::Kernels::SafeLog<kaad::Scalar> safe_log_kernel;
    kaad::Kernels::SafeExp<kaad::Scalar> safe_exp_kernel;

    kaad::Scalar value;
    value = kaad::Kernels::EPSILON<kaad::Scalar>;
    value = kaad::Kernels::MAX_FINITE<kaad::Scalar>;
    value = kaad::Kernels::MIN_FINITE<kaad::Scalar>;
    value = kaad::Kernels::MAX_EXP<kaad::Scalar>;
    value = kaad::Kernels::MIN_EXP<kaad::Scalar>;

    return 0;
}
