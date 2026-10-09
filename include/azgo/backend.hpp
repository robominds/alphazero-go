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
    // a stored row of A (lda), B (ldb) and C (ldc). gemmColumnBlock uses it
    // to compute a block of columns of C from the matching columns of op(B).
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

// How a convolution's gemms are spread over a ThreadPool. The forward
// product (y = w . columns) and the input gradient (dcolumns = w^T . dy)
// are cut into blocks of kGemmColumnBlock columns of C, i.e. of pixels,
// each one single-threaded gemm call over the whole of K; the cut depends
// only on the shape, so the result is the same for any number of threads,
// including one. Each column block also gives exactly the unsplit call's
// bits: by construction for the naive backend (each element sums over K in
// one fixed order), and for Accelerate as measured over ~11,000 conv
// shapes (boards 5-19, 8-256 filters, batches 1-256, blocks of 128-1024
// columns; tests/test_backend.cpp keeps a representative set). The weight
// gradient (dW = dy . columns^T, a sum over every pixel) is not cut: cut
// by rows it re-reads the whole column matrix per band and ran slower than
// one call, and cut by columns Accelerate's bits changed for some small
// batches. It runs as one call beside the input gradient's blocks.
constexpr int kGemmColumnBlock = 512;
// Fewer rows of C than this and the gemm stays one call: the 1x1 head
// convolutions' forward products (1 or 2 rows), for which Accelerate's bits
// did depend on the column split.
constexpr int kGemmMinSplitRows = 8;

// The number of column blocks of an M x N product: N / kGemmColumnBlock
// (the last block also takes the remainder, so none is narrower than
// kGemmColumnBlock), or 1 when N < 2 * kGemmColumnBlock or M < kGemmMinSplitRows.
int gemmColumnBlockCount(int M, int N);

// Block `block` of `blocks` (gemmColumnBlockCount, or 1) of a packed gemm,
// as Backend::gemm: those columns of C as one call with leading dimensions.
// With blocks == 1 it is exactly Backend::gemm.
void gemmColumnBlock(const Backend& be, int block, int blocks, bool transA, bool transB, int M, int N, int K,
                     float alpha, const float* A, const float* B, float beta, float* C);

// Every column block of a packed gemm on the pool; one plain Backend::gemm
// call when pool is null.
void gemmColumns(const Backend& be, ThreadPool* pool, bool transA, bool transB, int M, int N, int K,
                 float alpha, const float* A, const float* B, float beta, float* C);

// be.im2col / be.col2im, one input channel per chunk on the pool (a plain
// call when pool is null). A channel's rows of col come only from that
// channel, so the bits are the same either way.
void im2colChannels(const Backend& be, ThreadPool* pool, const float* in, int C, int B, int n, int k,
                    float* col);
void col2imChannels(const Backend& be, ThreadPool* pool, const float* col, int C, int B, int n, int k,
                    float* in);

} // namespace azgo
