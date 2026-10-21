// The numeric core. Almost all of a ResNet's time goes into one operation:
// general matrix multiply, C = alpha * A * B + beta * C. A convolution
// becomes a gemm once im2col copies every k x k input patch into a column,
// so a layer is "unfold, multiply" forward and "multiply, fold back" (col2im)
// backward. Keeping gemm/im2col/col2im behind this small interface lets the
// same layer code run on plain C++ loops or on BLAS (Apple Accelerate uses
// the CPU's matrix units). The GPU backend, metal, does not plug in here: a
// GPU only pays off on whole forward passes, so it has its own layers
// (src/metal/) behind the batching evaluator (azgo/inference_hub.hpp).
#pragma once
#include <memory>
#include <string>

namespace azgo {

class ThreadPool;   // azgo/parallel.hpp

// Where the arithmetic happens. Layers call only these operations, so a
// different CPU backend (naive loops, BLAS) needs no layer changes.
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

    // Optional: the packed gemm with the rows of C computed in parallel on
    // `pool`, in bands of kGemmRowBlock rows fixed by the shape, giving
    // exactly gemm's bits; for a long product that cannot be cut by columns
    // (a convolution's weight gradient). Returns false, computing nothing,
    // when the backend has no cheap way to do it (the default).
    virtual bool gemmRowsOnPool(ThreadPool& pool, bool transA, bool transB, int M, int N, int K, float alpha,
                                const float* A, const float* B, float beta, float* C) const;
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
    // Packs the transposed operands once, then runs the row bands: each
    // element's sum over K runs in the same fixed order as in one call.
    bool gemmRowsOnPool(ThreadPool& pool, bool transA, bool transB, int M, int N, int K, float alpha,
                        const float* A, const float* B, float beta, float* C) const override;
private:
    int threads_;
};

bool blasAvailable();  // true when built with -DAZ_USE_BLAS=ON

// The BLAS library's own build, where the OS version does not already pin
// its arithmetic: OpenBLAS's configuration line (version, kernel, flags),
// e.g. "OpenBLAS 0.3.26 NO_LAPACKE DYNAMIC_ARCH NO_AFFINITY Cooperlake
// MAX_THREADS=64", whose kernel name (Cooperlake) depends on the CPU. ""
// for Accelerate (it ships with macOS) and without BLAS.
std::string blasDescription();

// "naive" or "blas". threads applies to naive only; BLAS is kept
// single-threaded (our worker threads supply the parallelism).
// Throws std::invalid_argument for an unknown name or unavailable BLAS, and
// for "metal" and "cuda", which run whole forward passes on the GPU
// (WorkerEvaluators, azgo/inference_hub.hpp), not one operation at a time.
std::unique_ptr<Backend> makeBackend(const std::string& name, int threads = 1);

// The backend for the work that stays on the CPU: "blas" for "metal"
// (training, the bots, grow), any other name unchanged.
std::string cpuBackendName(const std::string& name);

// Throws std::invalid_argument unless `name` can run in this build on this
// machine: "backend 'blas' not available: rebuild with -DAZ_USE_BLAS=ON",
// "backend 'metal' not available: rebuild with -DAZ_USE_METAL=ON (macOS
// only)", "backend 'metal': no usable Metal GPU (<reason>)", "backend 'cuda'
// not available: rebuild with -DAZ_USE_CUDA=ON (Linux, NVIDIA GPU)" or
// "backend 'cuda': no usable CUDA GPU (<reason>)". For metal and cuda it
// creates the GPU context (Metal compiles its shaders), so a run can refuse
// before it writes anything.
void checkBackendAvailable(const std::string& name);

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
// gradient (dW = dy . columns^T, a sum over every pixel) is not cut by
// columns (Accelerate's bits changed for some small batches) and, with
// BLAS, not by rows either: each band re-reads the whole column matrix,
// and on Accelerate the bands ran slower than one call. It runs as one
// call beside the input gradient's blocks, except on a backend that
// offers Backend::gemmRowsOnPool (the naive one), where it is cut into
// bands of kGemmRowBlock rows of C (the last band also takes the remainder).
constexpr int kGemmColumnBlock = 512;
constexpr int kGemmRowBlock = 16;
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
