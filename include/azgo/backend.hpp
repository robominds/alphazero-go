// The numeric core. Almost all of a ResNet's time goes into one operation:
// general matrix multiply, C = alpha * A * B + beta * C. A convolution
// becomes a gemm once im2col copies every k x k input patch into a column,
// so a layer is "unfold, multiply" forward and "multiply, fold back" (col2im)
// backward. Keeping gemm/im2col/col2im behind this small interface lets the
// same layer code run on plain C++ loops, on BLAS (Apple Accelerate uses the
// CPU's matrix units), or on a GPU backend later.
#pragma once
#include <memory>
#include <string>

namespace azgo {

// Where the arithmetic happens. Layers call only these operations, so a
// different backend (BLAS today, Metal on the GPU later) needs no layer changes.
class Backend {
public:
    virtual ~Backend() = default;
    virtual const char* name() const = 0;

    // Row-major C[M x N] = alpha * op(A) * op(B) + beta * C, tightly packed.
    // op(A) is M x K (A is stored K x M when transA); op(B) is K x N
    // (B is stored N x K when transB). beta == 0 means C is overwritten.
    virtual void gemm(bool transA, bool transB, int M, int N, int K, float alpha,
                      const float* A, const float* B, float beta, float* C) const = 0;

    // 'Same'-padded k x k patch unfolding (k odd). in: [C][B][n*n];
    // col: [C*k*k][B*n*n], row index (c*k + dy)*k + dx.
    virtual void im2col(const float* in, int C, int B, int n, int k, float* col) const;
    // Adjoint of im2col: OVERWRITES in with the sum of contributions.
    virtual void col2im(const float* col, int C, int B, int n, int k, float* in) const;
};

// Blocked loops, packing transposed operands first. threads > 1 splits the
// rows of C across std::threads (each element computed by exactly one thread,
// so results do not depend on the thread count).
class NaiveBackend : public Backend {
public:
    explicit NaiveBackend(int threads = 1);
    const char* name() const override { return "naive"; }
    void gemm(bool transA, bool transB, int M, int N, int K, float alpha,
              const float* A, const float* B, float beta, float* C) const override;
private:
    int threads_;
};

bool blasAvailable();  // true when built with -DAZ_USE_BLAS=ON

// "naive" or "blas". threads applies to naive only; BLAS is kept
// single-threaded (our worker threads supply the parallelism).
// Throws std::invalid_argument for an unknown name or unavailable BLAS.
std::unique_ptr<Backend> makeBackend(const std::string& name, int threads = 1);

} // namespace azgo
