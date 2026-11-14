// TEST ONLY: intentionally permits a causal context on an explicitly marked synthetic fixture.
// This is not a production inference executable and is never installed or shipped as glide-logits.
#include "llama.h"
#include "protocol.hpp"
#include <cmath>
#include <cstring>
#include <iostream>
#include <memory>
#include <numeric>
#include <vector>

int main(int argc, char ** argv) {
    try {
        if ((argc != 3 && argc != 4) || (std::strcmp(argv[2], "causal") != 0 && std::strcmp(argv[2], "noncausal") != 0))
            throw std::runtime_error("usage: glide-test-attention-control SYNTHETIC.gguf causal|noncausal [context-size]");
        const int capacity = argc == 4 ? std::stoi(argv[3]) : 64;
        if (capacity < 1 || capacity > 1024) throw std::runtime_error("invalid synthetic capacity");
        const bool causal = std::strcmp(argv[2], "causal") == 0;
        llama_backend_init();
        auto mp = llama_model_default_params();
        ggml_backend_dev_t devices[] = {nullptr};
        mp.devices = devices;
        mp.n_gpu_layers = 0;
        mp.load_mtp = false;
        std::unique_ptr<llama_model, decltype(&llama_model_free)> model(llama_model_load_from_file(argv[1], mp), llama_model_free);
        if (!model) throw std::runtime_error("fixture load failed");
        char marker[8]{};
        if (llama_model_meta_val_str(model.get(), "test.synthetic_fixture", marker, sizeof(marker)) != 4 ||
            std::strcmp(marker, "true") != 0 || llama_model_n_embd_out(model.get()) != 255)
            throw std::runtime_error("control executable accepts only explicitly marked 255-output synthetic fixtures");
        auto cp = llama_context_default_params();
        cp.n_ctx = cp.n_batch = cp.n_ubatch = cp.n_outputs_max = cp.n_outputs_max_per_seq = capacity;
        cp.n_seq_max = 1;
        cp.n_rs_seq = 0;
        cp.n_threads = cp.n_threads_batch = 2;
        cp.embeddings = true;
        cp.pooling_type = LLAMA_POOLING_TYPE_LAST;
        cp.attention_type = causal ? LLAMA_ATTENTION_TYPE_CAUSAL : LLAMA_ATTENTION_TYPE_NON_CAUSAL;
        cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
        cp.type_k = cp.type_v = GGML_TYPE_F16;
        cp.offload_kqv = cp.op_offload = false;
        std::unique_ptr<llama_context, decltype(&llama_free)> ctx(llama_init_from_model(model.get(), cp), llama_free);
        if (!ctx || llama_get_causal_attn(ctx.get()) != causal) throw std::runtime_error("wrong control context mode");
        std::vector<int32_t> readouts(255);
        std::iota(readouts.begin(), readouts.end(), 0);
        const auto * vocab = llama_model_get_vocab(model.get());
        std::string line;
        while (std::getline(std::cin, line)) {
            auto req = glide::validate_request(glide::parse_json(line), llama_vocab_n_tokens(vocab), capacity, readouts);
            auto memory = llama_get_memory(ctx.get());
            if (!memory) throw std::runtime_error("hybrid memory unavailable");
            llama_memory_clear(memory, true);
            const int32_t n = static_cast<int32_t>(req.token_ids.size());
            auto batch = llama_batch_init(n, 0, 1);
            batch.n_tokens = n;
            for (int32_t i = 0; i < n; ++i) {
                batch.token[i] = req.token_ids[i];
                batch.pos[i] = i;
                batch.n_seq_id[i] = 1;
                batch.seq_id[i][0] = 0;
                batch.logits[i] = 1;
            }
            const int status = llama_decode(ctx.get(), batch);
            llama_batch_free(batch);
            if (status != 0) throw std::runtime_error("control decode failed");
            const float * logits = llama_get_embeddings_seq(ctx.get(), 0);
            if (!logits) throw std::runtime_error("control readout unavailable");
            std::vector<float> values(logits, logits + 255);
            for (auto value : values) if (!std::isfinite(value)) throw std::runtime_error("nonfinite control readout");
            std::cout << glide::json({{"id", req.id}, {"logits", values}, {"causal", causal}}).dump() << '\n' << std::flush;
            llama_memory_clear(memory, true);
        }
        ctx.reset();
        model.reset();
        llama_backend_free();
        return 0;
    } catch (const std::exception & exc) {
        std::cerr << "Attention control failed: " << exc.what() << '\n';
        return 2;
    }
}
