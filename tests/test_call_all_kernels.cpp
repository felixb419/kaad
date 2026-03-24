#include <kaad/functions/kernels.hpp>      // for Kernls
#include <kaad/functions/safe_kernels.hpp> // for Kernls
#include <kaad/scalar.hpp>                 // for kaad::Scalar

int main() {

    kaad::Kernels::Add<kaad::Scalar> Add_kernel;
    kaad::Kernels::Sub<kaad::Scalar> Sub_kernel;
    kaad::Kernels::Mul<kaad::Scalar> Mul_kernel;
    kaad::Kernels::Div<kaad::Scalar> Div_kernel;
    kaad::Kernels::Pow<kaad::Scalar> Pow_kernel;
    kaad::Kernels::Dot<kaad::Scalar> Dot_kernel;
    kaad::Kernels::Min<kaad::Scalar> Min_kernel;
    kaad::Kernels::Max<kaad::Scalar> Max_kernel;
    kaad::Kernels::NoOp<kaad::Scalar> NoOp_kernel;
    kaad::Kernels::Sum<kaad::Scalar> Sum_kernel;
    kaad::Kernels::Neg<kaad::Scalar> Neg_kernel;
    kaad::Kernels::Square<kaad::Scalar> Square_kernel;
    kaad::Kernels::Sqrt<kaad::Scalar> Sqrt_kernel;
    kaad::Kernels::Log<kaad::Scalar> Log_kernel;
    kaad::Kernels::Exp<kaad::Scalar> Exp_kernel;
    kaad::Kernels::Abs<kaad::Scalar> Abs_kernel;

    kaad::Kernels::SafeDiv<kaad::Scalar> safe_Div_kernel;
    kaad::Kernels::SafePow<kaad::Scalar> safe_Pow_kernel;
    kaad::Kernels::SafeSqrt<kaad::Scalar> safe_Sqrt_kernel;
    kaad::Kernels::SafeLog<kaad::Scalar> safe_Log_kernel;
    kaad::Kernels::SafeExp<kaad::Scalar> safe_Exp_kernel;

    kaad::Scalar value;
    value = kaad::Kernels::EPSILON<kaad::Scalar>;
    value = kaad::Kernels::MAX_FINITE<kaad::Scalar>;
    value = kaad::Kernels::MIN_FINITE<kaad::Scalar>;
    value = kaad::Kernels::MAX_EXP<kaad::Scalar>;
    value = kaad::Kernels::MIN_EXP<kaad::Scalar>;

    return 0;
}
