// Compiled only when CMake is configured with -DAZ_USE_BLAS=ON.
#include "azgo/backend.hpp"
#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <memory>
#include <string>

#ifdef __APPLE__
// Select Accelerate's current CBLAS headers (the legacy ones are deprecated
// since macOS 13.3); the calling convention is the standard CBLAS one.
// CMake already defines it for the azgo target (Task 1); the guard keeps a
// second definition from triggering -Wmacro-redefined.
#ifndef ACCELERATE_NEW_LAPACK
#define ACCELERATE_NEW_LAPACK
#endif
#include <Accelerate/Accelerate.h>
#else
#include <cblas.h>
extern "C" void openblas_set_num_threads(int);
extern "C" char* openblas_get_config(void);
#endif

namespace azgo {

namespace {

// Only gemm is delegated; im2col/col2im are memory shuffles that BLAS does
// not provide, so the Backend defaults are used.
class BlasBackend : public Backend {
public:
    const char* name() const override { return "blas"; }

    using Backend::gemm;
    void gemm(bool transA, bool transB, int M, int N, int K, float alpha, const float* A, int lda,
              const float* B, int ldb, float beta, float* C, int ldc) const override {
        if (M <= 0 || N <= 0) return;
        if (K <= 0) {
            // BLAS requires leading dimensions >= 1; the product is empty anyway.
            for (int i = 0; i < M; ++i) {
                float* c = C + static_cast<size_t>(i) * ldc;
                if (beta == 0.0f) std::fill(c, c + N, 0.0f);
                else for (int j = 0; j < N; ++j) c[j] *= beta;
            }
            return;
        }
        // Row-major leading dimensions are the STORED row lengths:
        // A is M x K (or K x M when transposed), B is K x N (or N x K), C is M x N.
        // Like reference BLAS, cblas_sgemm does not read C when beta == 0.
        cblas_sgemm(CblasRowMajor, transA ? CblasTrans : CblasNoTrans,
                    transB ? CblasTrans : CblasNoTrans, M, N, K, alpha,
                    A, lda, B, ldb, beta, C, ldc);
    }
};

// Called once, before the first BLAS call. Self-play and arena already run
// one worker thread per core; a multithreaded BLAS inside each worker would
// oversubscribe the cores and make timings erratic.
bool configureSingleThreaded() {
#ifdef __APPLE__
    // Accelerate reads this when it first spins up its thread pool. (On Apple
    // Silicon, sgemm runs on the AMX/SME matrix unit from a single thread
    // anyway, so this costs little.) setenv is not thread-safe, which is why
    // it happens once, from the first makeBackend("blas") call.
    setenv("VECLIB_MAXIMUM_THREADS", "1", 1);
#else
    openblas_set_num_threads(1);
#endif
    return true;
}

} // namespace

std::string blasDescription() {
#ifdef __APPLE__
    return "";   // Accelerate: part of macOS, so the OS version names it
#else
    const char* config = openblas_get_config();
    return config ? std::string(config) : std::string("OpenBLAS (no configuration)");
#endif
}

std::unique_ptr<Backend> makeBlasBackend() {
    static const bool configured = configureSingleThreaded();
    (void)configured;
    return std::make_unique<BlasBackend>();
}

} // namespace azgo
