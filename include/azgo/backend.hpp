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

class ThreadPool;   // azgo/parallel.hpp

// Where the arithmetic happens. Layers call only these operations, so a
// different backend (BLAS today, Metal on the GPU later) needs no layer changes.
class Backend {
public:
    virtual ~Backend() = default;
    virtual const char* name() const = 0;

    // Row-major C[M x N] = alpha * op(A) * op(B) + beta * C, tightly packed.
    // op(A) is M x K (A is stored K x M when transA); op(B) is K x N
    // (B is stored N x K when transB). beta == 0 means C is overwritten.
    void gemm(bool transA, bool transB, int M, int N, int K, float alpha,
              const float* A, const float* B, float beta, float* C) const {
        gemm(transA, transB, M, N, K, alpha, A, transA ? M : K, B, transB ? K : N, beta, C, N);
    }
    // The same with explicit leading dimensions, BLAS style: the length of
    // a stored row of A (lda), B (ldb) and C (ldc). gemmRowBlocks uses it to
    // compute a band of rows of C from the matching band of op(A).
    virtual void gemm(bool transA, bool transB, int M, int N, int K, float alpha,
                      const float* A, int lda, const float* B, int ldb, float beta,
                      float* C, int ldc) const = 0;

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
    using Backend::gemm;
    void gemm(bool transA, bool transB, int M, int N, int K, float alpha,
              const float* A, int lda, const float* B, int ldb, float beta,
              float* C, int ldc) const override;
private:
    int threads_;
};

bool blasAvailable();  // true when built with -DAZ_USE_BLAS=ON

// "naive" or "blas". threads applies to naive only; BLAS is kept
// single-threaded (our worker threads supply the parallelism).
// Throws std::invalid_argument for an unknown name or unavailable BLAS.
std::unique_ptr<Backend> makeBackend(const std::string& name, int threads = 1);

// The rows of C that gemmRowBlocks gives each single-threaded gemm call.
constexpr int kGemmRowBlock = 16;

// Packed gemm (as Backend::gemm) for the convolutions of a training step.
// pool == nullptr: one call, exactly Backend::gemm. Otherwise the rows of C
// are cut into bands of kGemmRowBlock rows (the last band also takes the
// M % kGemmRowBlock rows left over, so no band is thinner than the block;
// M < 2 * kGemmRowBlock stays one band) and each band is one single-threaded
// gemm call on the pool. The cut depends only on M, so the result is the
// same for any number of threads, including one. Each row of C still sums
// over the whole of K in one call, which is why the bands give exactly the
// unsplit call's bits: by construction for the naive backend, and for
// Accelerate as measured for every shape the network uses (tests/test_backend.cpp).
void gemmRowBlocks(const Backend& be, ThreadPool* pool, bool transA, bool transB, int M, int N, int K,
                   float alpha, const float* A, const float* B, float beta, float* C);

// be.im2col / be.col2im, one input channel per chunk on the pool (a plain
// call when pool is null). A channel's rows of col come only from that
// channel, so the bits are the same either way.
void im2colChannels(const Backend& be, ThreadPool* pool, const float* in, int C, int B, int n, int k,
                    float* col);
void col2imChannels(const Backend& be, ThreadPool* pool, const float* col, int C, int B, int n, int k,
                    float* in);

} // namespace azgo
