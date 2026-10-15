// CPU mul_mat_id with negative ids: the expert is skipped and its output row is zero,
// the other rows are the same as with all the ids (used by the hybrid MoE cache)

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static int n_failed = 0;

static void cpu_set_proc(ggml_backend_t backend, const char * name, int value) {
    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(backend));
    if (strcmp(name, "ggml_backend_cpu_set_use_ref") == 0) {
        auto fn = (void (*)(ggml_backend_t, bool)) ggml_backend_reg_get_proc_address(reg, name);
        if (!fn) {
            fprintf(stderr, "%s not available\n", name);
            exit(1);
        }
        fn(backend, value != 0);
    } else {
        auto fn = (void (*)(ggml_backend_t, int)) ggml_backend_reg_get_proc_address(reg, name);
        if (!fn) {
            fprintf(stderr, "%s not available\n", name);
            exit(1);
        }
        fn(backend, value);
    }
}

// as: [k, m, n_as], b: [k, n_b, n_tokens] with n_b = 1 (broadcast, like gate/up) or n_used (like down)
static void test_skip(ggml_backend_t backend, ggml_type type, int64_t n_b, int n_threads, bool use_ref) {
    const int64_t k        = 256; // a multiple of the K-quant block size
    const int64_t m        = 48;
    const int64_t n_as     = 16;
    const int64_t n_used   = 4;
    const int64_t n_tokens = 6;

    srand(1234);
    std::vector<float> as_f32(k*m*n_as);
    std::vector<float> b_f32(k*n_b*n_tokens);
    for (float & x : as_f32) { x = (float) rand()/RAND_MAX - 0.5f; }
    for (float & x : b_f32)  { x = (float) rand()/RAND_MAX - 0.5f; }

    // distinct experts per token
    std::vector<int32_t> ids_full(n_used*n_tokens);
    for (int64_t t = 0; t < n_tokens; ++t) {
        for (int64_t i = 0; i < n_used; ++i) {
            ids_full[t*n_used + i] = (int32_t) ((t*5 + i*3) % n_as);
        }
    }
    // token 0 keeps all its experts, token 1 skips all of them, the others skip some
    std::vector<int32_t> ids_skip = ids_full;
    for (int64_t t = 1; t < n_tokens; ++t) {
        for (int64_t i = 0; i < n_used; ++i) {
            if (t == 1 || (t + i) % 2 == 0) {
                ids_skip[t*n_used + i] = -1;
            }
        }
    }

    ggml_init_params ip = { 16*1024*1024, nullptr, true };
    ggml_context * ctx = ggml_init(ip);

    ggml_tensor * as  = ggml_new_tensor_3d(ctx, type,           k, m,      n_as);
    ggml_tensor * b   = ggml_new_tensor_3d(ctx, GGML_TYPE_F32,  k, n_b,    n_tokens);
    ggml_tensor * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32,  n_used, n_tokens);
    ggml_tensor * out = ggml_mul_mat_id(ctx, as, b, ids); // [m, n_used, n_tokens]

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);

    if (type == GGML_TYPE_F32) {
        ggml_backend_tensor_set(as, as_f32.data(), 0, ggml_nbytes(as));
    } else {
        std::vector<uint8_t> q(ggml_nbytes(as));
        ggml_quantize_chunk(type, as_f32.data(), q.data(), 0, m*n_as, k, nullptr);
        ggml_backend_tensor_set(as, q.data(), 0, ggml_nbytes(as));
    }
    ggml_backend_tensor_set(b, b_f32.data(), 0, ggml_nbytes(b));

    cpu_set_proc(backend, "ggml_backend_set_n_threads", n_threads);
    cpu_set_proc(backend, "ggml_backend_cpu_set_use_ref", use_ref);

    std::vector<float> out_full(ggml_nelements(out));
    std::vector<float> out_skip(ggml_nelements(out), NAN); // the skipped rows must be written, not left as they were

    ggml_backend_tensor_set(ids, ids_full.data(), 0, ggml_nbytes(ids));
    ggml_backend_graph_compute(backend, gf);
    ggml_backend_tensor_get(out, out_full.data(), 0, ggml_nbytes(out));

    ggml_backend_tensor_set(out, out_skip.data(), 0, ggml_nbytes(out));
    ggml_backend_tensor_set(ids, ids_skip.data(), 0, ggml_nbytes(ids));
    ggml_backend_graph_compute(backend, gf);
    ggml_backend_tensor_get(out, out_skip.data(), 0, ggml_nbytes(out));

    // without the tiled kernel each row is computed the same way in both runs; the tiled kernel picks its
    // tiles from the number of rows per expert, so the kept rows may differ by rounding
    const float tol = use_ref ? 0.0f : 1e-4f;
    int n_bad_zero = 0;
    int n_bad_kept = 0;
    float max_err  = 0.0f;
    for (int64_t t = 0; t < n_tokens; ++t) {
        for (int64_t i = 0; i < n_used; ++i) {
            const float * row_full = out_full.data() + (t*n_used + i)*m;
            const float * row_skip = out_skip.data() + (t*n_used + i)*m;
            for (int64_t r = 0; r < m; ++r) {
                if (ids_skip[t*n_used + i] < 0) {
                    n_bad_zero += row_skip[r] != 0.0f;
                } else {
                    const float err = std::fabs(row_full[r] - row_skip[r]);
                    max_err = std::fmax(max_err, std::isnan(err) ? INFINITY : err);
                    n_bad_kept += !(err <= tol*std::fmax(1.0f, std::fabs(row_full[r])));
                }
            }
        }
    }

    const bool ok = n_bad_zero == 0 && n_bad_kept == 0;
    printf("mul_mat_id skip %-5s n_b=%lld threads=%d %-6s: %s (non-zero skipped values: %d, kept values off: %d, max err %g)\n",
        ggml_type_name(type), (long long) n_b, n_threads, use_ref ? "ref" : "tiled", ok ? "PASS" : "FAIL", n_bad_zero, n_bad_kept, max_err);
    n_failed += !ok;

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
}

int main() {
    ggml_backend_load_all();
    ggml_backend_t backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    if (!backend) {
        fprintf(stderr, "failed to initialize the CPU backend\n");
        return 1;
    }

    for (ggml_type type : { GGML_TYPE_F32, GGML_TYPE_Q4_K, GGML_TYPE_Q5_K, GGML_TYPE_Q6_K }) {
        for (int64_t n_b : { 1, 4 }) {
            for (int n_threads : { 1, 4 }) {
                for (bool use_ref : { true, false }) {
                    test_skip(backend, type, n_b, n_threads, use_ref);
                }
            }
        }
    }

    ggml_backend_free(backend);

    printf("%s\n", n_failed == 0 ? "OK" : "FAILED");
    return n_failed == 0 ? 0 : 1;
}
