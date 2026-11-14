// Unit tests of fail-closed evidence admission. Fabricated records are NOT runtime evidence.
#include "execution_evidence.hpp"
#include "llama.h"
#include <functional>
#include <iostream>
#include <stdexcept>

int main() {
    using glide::json;
    using glide::execution_observer;
    size_t count = 0;
    auto check = [&](bool yes, const char * msg) { if (!yes) throw std::runtime_error(msg); ++count; };
    auto reject = [&](execution_observer value, size_t expected = 1, bool cuda = false) {
        try { value.finish(3, expected, cuda); }
        catch (const glide::error & e) { check(e.code == "execution_evidence_failed", "wrong error"); return; }
        throw std::runtime_error("invalid execution evidence was accepted");
    };
    const json cpu = {{"backend", "CPU"}, {"type", "CPU"}, {"name", "unit-test CPU"}, {"buffer", "CPU"}};
    const json cuda = {{"backend", "CUDA"}, {"type", "GPU"}, {"name", "unit-test CUDA"}, {"buffer", "CUDA0"}};
    auto make = [&] {
        execution_observer value;
        value.begin(); value.decode_calls = 1; value.reset_before = value.reset_after = true;
        value.fa.push_back({{"name", "kqv_out-1"}, {"op", "FLASH_ATTN_EXT"}, {"output_device", cpu}, {"phase", "post_execution"}, {"executed", true}});
        value.classifier.push_back({{"name", "result_embd_proj"}, {"op", "MUL_MAT"}, {"weight_name", "cls.output.weight"},
            {"output_device", cpu}, {"weight_device", cpu}, {"phase", "post_execution"}, {"executed", true}});
        value.embedding.push_back({{"name", "lookup"}, {"op", "GET_ROWS"}, {"weight_name", "token_embd.weight"},
            {"output_device", cpu}, {"weight_device", cpu}, {"phase", "post_execution"}, {"executed", true}});
        return value;
    };
    auto good = make();
    check(good.finish(3, 1, false).at("requested_backend") == "cpu", "CPU unit admission failed");
    auto x = make(); x.reset_before = false; reject(x);
    x = make(); x.reset_after = false; reject(x);
    x = make(); x.decode_calls = 0; reject(x);
    x = make(); x.decode_calls = 2; reject(x);
    x = make(); x.callback_failed = true; reject(x);
    reject(make(), 2); reject(make(), 0); reject(make(), 1, true);
    x = make(); x.fa.push_back(x.fa[0]); reject(x, 2);
    x = make(); x.classifier.clear(); reject(x);
    x = make(); x.embedding.clear(); reject(x);
    x = make(); x.vocabulary.push_back(x.classifier[0]); reject(x);
    x = make(); x.embedding[0]["weight_device"] = cuda; reject(x);
    x = make(); x.fa[0]["output_device"] = cuda; reject(x);
    x = make(); x.classifier[0]["weight_device"] = cuda; reject(x);
    x = make(); x.fa[0]["output_device"] = cuda; x.classifier[0]["output_device"] = cuda; x.classifier[0]["weight_device"] = cuda;
    check(x.finish(3, 1, true).at("requested_backend") == "cuda", "CUDA admission logic unit failed");
    x = make(); x.fa[0]["output_device"] = cuda; x.classifier[0]["output_device"] = cuda; x.classifier[0]["weight_device"] = cuda;
    x.embedding[0]["output_device"]["backend"] = "Unknown"; reject(x, 1, true);
    // Merely entering a callback is insufficient: inactive graph branches and
    // pre-execution ask callbacks cannot fabricate successful node records.
    ggml_tensor node{}; node.op = GGML_OP_FLASH_ATTN_EXT; std::strcpy(node.name, "test-fa");
    x.begin();
    check(!execution_observer::callback(&node, true, &x), "inactive branch selected");
    execution_observer::callback(&node, false, &x);
    check(x.fa.empty() && !x.callback_failed, "inactive branch recorded as executed");
    node.flags = GGML_TENSOR_FLAG_COMPUTE;
    check(execution_observer::callback(&node, true, &x) && x.fa.empty(), "pre-execution callback recorded evidence");
    check(execution_observer::callback(&node, false, &x) && x.callback_failed, "unallocated node must fail without throwing");
    llama_backend_init();
    auto buffer = ggml_backend_buft_alloc_buffer(ggml_backend_cpu_buffer_type(), 64);
    node.buffer = buffer;
    auto record = execution_observer::device(&node);
    check(record["backend"] == "CPU" && record["type"] == "CPU" && record["buffer"] == "CPU", "actual CPU allocation not resolved");
    ggml_backend_buffer_free(buffer);
    alignas(64) char storage[64]{};
    buffer = ggml_backend_cpu_buffer_from_ptr(storage, sizeof(storage)); node.buffer = buffer;
    record = execution_observer::device(&node);
    check(record["backend"] == "CPU" && record["type"] == "CPU" && record["buffer"] == "CPU_Mapped", "actual CPU mapped allocation not resolved");
    ggml_backend_buffer_free(buffer);
    llama_backend_free();
    std::cout << count << " evidence admission checks passed; no CUDA execution claim\n";
}
