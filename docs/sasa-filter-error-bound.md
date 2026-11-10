# SASA mixed-precision predicate guarantee

## Scope and reference

The guarantee is equality of the exposed-point integer counts to the
CPU-order binary64 predicate. It is not a bound on discretization of the
continuous solvent-accessible surface.

The public GPU dispatcher accepts finite binary64 coordinates with
absolute values <= 10^6, finite atomic radii >= 0.1 whose inflated radii
r = RN64(radius + 1.4) are <= 10, and the generated 960 binary64 sphere
points with coordinate magnitudes <= 1. Unsupported cases use the CPU.
Thus each inflated radius is > 1 and all expressions below are finite.
The same inflated radii and sphere points enter both paths.

The two distance predicates use IEEE binary32/binary64 round-to-nearest-even,
separate multiply and add operations, no reassociation or FMA contraction.
The error-envelope construction may use FMA; fewer rounding steps only
improve the envelope-rounding bound used below.
The kernels use explicit libdevice *_rn operations and fastmath=False.
Normal CUDA compilation does not flush subnormals; even a binary32
flush-to-zero absolute error <= 2^-126 is covered by the slack below.
The comparison target is the existing CPU binary64 operation order,
not an infinitely precise geometry.

## Definitions

Let u = 2^-24 and v = 2^-53 be unit roundoffs.
For a target center x, occluder y, inflated radii r,R, and sphere point p,
define d_k = RN64(y_k - x_k), for k = 0,1,2, and real bounds

S = r + R + sum_k |d_k|,
C = max(10, max_k |x_k|, max_k |y_k|) + 10.

Here 2 < S < 6,000,021 and 20 <= C <= 1,000,010.

CPU displacement per coordinate is
D_k = RN64(RN64(x_k + RN64(p_k*r)) - y_k).

The filter displacement is
h_k = RN32(RN32(RN32(p_k)*RN32(r)) - RN32(d_k)).

The CPU squared-distance comparison is
Q = RN64(RN64(D_0*D_0 + D_1*D_1) + D_2*D_2),
T = RN64(R*R).
Each multiplication is rounded before each addition.

The filter uses the same left-to-right reduction in binary32, yielding q,
t = RN32(T), and m = RN32(q - t).

## Forward-error inequalities

For a normal result, |RN(z)-z| <= unit_roundoff*|z|.
For subnormal results add at most half a minimum subnormal; using the
larger 2^-126 bound for binary32 also covers possible flushing.
All such additive terms are smaller than u*S by over 10^30 and fit the
explicit slack used below.

Applying this elementary bound successively to the three casts, product,
and subtraction in h, and to the CPU product/add/subtract and d, gives

|h_k - D_k| <= 8*u*S + 8*v*C,
|h_k| <= 2*S, |D_k| <= 2*S.

These constants include the subnormal terms. The magnitude inequalities
follow also at the extreme C/S because C <= 1,000,010 and S > 2.
Consequently, the discrepancy between the real sums of squares is at most

sum_k |h_k^2 - D_k^2|
 <= sum_k |h_k-D_k|*(|h_k|+|D_k|)
 <= 96*u*S^2 + 96*v*S*C.

Five rounded operations evaluate each sum of three squares. Since
gamma_5 = 5*u/(1-5*u) < 6*u, their errors are bounded by

72*u*S^2 (binary32), 72*v*S^2 (binary64).

Rounding T to binary32 contributes at most 2*u*S^2.
Rounding q-t contributes at most 15*u*S^2, with one further u*S^2
reserved for subnormal/slack terms. Using
96*v*S*C <= 48*v*S^2 + 48*v*C^2 and v/u = 2^-29 gives the deliberately
looser final envelope

|m - (Q-T)| < 200*u*S^2 + 64*v*C^2.                 (1)

## Envelope implemented in the kernel

The kernel computes S in binary64 and C as above, then stores

B = RN32(RN64(256*u*S_hat^2 + 512*v*C_hat^2)).

The implementation spells this as twice
(64*eps32*S_hat^2 + 128*eps64*C_hat^2).
S_hat includes four positive additions, so its relative error is < 5*v.
C_hat and subsequent binary64 arithmetic have fewer than 16 rounding
operations. Including the binary32 cast, the stored envelope obeys

B > 255*u*S^2 + 510*v*C^2
  > 200*u*S^2 + 64*v*C^2.                           (2)

No overflow or underflow can affect B in the accepted domain.

From (1)-(2), m > B implies Q-T > 0, while m < -B implies Q-T < 0.
All remaining cases, including equality, are recomputed with the exact
CPU-order binary64 predicate. Therefore the filter cannot accept a
wrong sign anywhere in the accepted input domain.

## Boundary fallback and composition

The fallback uses absolute centers and the exact CPU product/add/subtract
order, with explicit binary64 RN intrinsics. Its strict Q < T comparison
preserves CPU treatment of equality.

Every candidate occluder therefore produces the same boolean decision.
Reordering occluders changes early-exit work but not their logical OR.
Each exposed-point flag agrees, so the integer count agrees exactly.
Host surface-area conversion uses the existing division and radius factor.

Neighbor-list correctness is a separate prerequisite. Existing conservative
candidate generation and capacity/coincident-input guards remain in force.
This proof establishes the mixed-precision filter, not the correctness of
every spatial search, trajectory reconstruction, or scientific assumption.

## Execution-model references

NVIDIA documents rounding and the separate-operation RN intrinsics:
https://docs.nvidia.com/cuda/floating-point/index.html
https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__SINGLE.html

The derivation is an analytic error proof under the stated arithmetic model.
It has not been machine-checked in a proof assistant. Regression tests and
PTX inspection validate implementation assumptions but do not replace it.
