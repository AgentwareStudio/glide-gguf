#include "llama.h"
#include "gguf.h"
#include "protocol.hpp"
#include "execution_evidence.hpp"
#include "ggml-cpu.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <memory>
#include <set>
#include <string>
#include <thread>
#include <vector>

namespace {
using clock_type = std::chrono::steady_clock;
using glide::json;
using glide::error;
constexpr const char * kSourceRevision = "1a9a7ed94379c2ea6f324176c9bc22fb9b7b7d60";
constexpr const char * kTokenizerSha256 = "06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523";
constexpr const char * kDecisionConfigSha256 = "df40c569d6248a8562177ea8377d47c6fbe558521f60dd14e4e8a78d04e7ce10";
constexpr uint32_t kHardMaxContext = 131072;

struct options {
    std::string model_path;
    uint32_t ctx_size = 4096;
    int32_t threads = std::max(1u, std::min(8u, std::thread::hardware_concurrency()));
    uint32_t timeout_ms = 120000;
    int32_t gpu_layers = 0;
};

uint32_t positive_int(const std::string & text, uint32_t maximum, const char * key) {
    if (text.empty() || text.find_first_not_of("0123456789") != std::string::npos)
        throw error("invalid_argument", std::string(key) + " requires a positive integer");
    unsigned long long value;
    try { value = std::stoull(text); }
    catch (...) { throw error("invalid_argument", std::string(key) + " is too large"); }
    if (value == 0 || value > maximum) throw error("invalid_argument", std::string(key) + " is outside the supported range");
    return static_cast<uint32_t>(value);
}

options parse_options(int argc, char ** argv) {
    options opt;
    std::set<std::string> seen;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--help") {
            std::cout << "GLiDE raw logits runner (CPU by default) (no generation or softmax)\n"
                         "Usage: glide-logits-fa --model MODEL.gguf [--ctx-size 4096] [--threads N] [--timeout-ms 120000] [--gpu-layers 0]\n"
                         "Input: JSONL {id,token_ids,readout_token_ids}; exactly 255 canonical readout IDs.\n"
                         "Each prompt is decoded once, in one noncausal batch, with fresh KV/recurrent state.\n"
                         "--ctx-size: 1..131072 tokens, no truncation; --timeout-ms: 1..86400000 (cooperative CPU abort; GPU needs external watchdog).\n"
                         "--gpu-layers: 0 (CPU) or -1 (all CUDA); partial offload is rejected.\n";
            std::exit(0);
        }
        if (arg == "--version") {
            std::cout << "glide-logits-fa 1; profile glide-fa-hostemb-v1; llama.cpp " << GLIDE_LLAMA_REVISION << '\n';
            std::exit(0);
        }
        if (arg != "--model" && arg != "--ctx-size" && arg != "--threads" && arg != "--timeout-ms" && arg != "--gpu-layers")
            throw error("invalid_argument", "Unknown option: " + arg);
        if (!seen.insert(arg).second) throw error("invalid_argument", "Duplicate option: " + arg);
        if (++i == argc) throw error("invalid_argument", "Missing value for " + arg);
        const std::string value = argv[i];
        if (arg == "--model") opt.model_path = value;
        else if (arg == "--ctx-size") opt.ctx_size = positive_int(value, kHardMaxContext, "--ctx-size");
        else if (arg == "--threads") opt.threads = positive_int(value, 1024, "--threads");
        else if (arg == "--gpu-layers") opt.gpu_layers = value == "-1" ? -1 : (value == "0" ? 0 : positive_int(value, 10000, "--gpu-layers"));
        else opt.timeout_ms = positive_int(value, 86400000, "--timeout-ms");
    }
#ifndef GLIDE_CUDA_BUILD
    if (opt.gpu_layers != 0) throw error("invalid_argument", "Nonzero --gpu-layers requires a GGML_CUDA=ON build");
#endif
    if (opt.gpu_layers != 0 && opt.gpu_layers != -1) throw error("invalid_argument", "glide-fa-hostemb-v1 requires --gpu-layers 0 or -1; partial offload is not admitted");
    if (opt.model_path.empty()) throw error("invalid_argument", "--model is required");
    return opt;
}

bool starts_with(const std::string & value, const std::string & prefix) { return value.rfind(prefix, 0) == 0; }
bool ends_with(const std::string & value, const std::string & suffix) {
    return value.size() >= suffix.size() && value.compare(value.size() - suffix.size(), suffix.size(), suffix) == 0;
}

int64_t require_key(const gguf_context * gguf, const char * name, gguf_type type) {
    auto index = gguf_find_key(gguf, name);
    if (index < 0 || gguf_get_kv_type(gguf, index) != type)
        throw error("unsupported_model", std::string("Missing or incorrectly typed GGUF metadata: ") + name);
    return index;
}

std::string require_string(const gguf_context * gguf, const char * name, const char * expected = nullptr) {
    std::string value = gguf_get_val_str(gguf, require_key(gguf, name, GGUF_TYPE_STRING));
    if (expected && value != expected) throw error("unsupported_model", std::string("Unsupported GGUF metadata: ") + name);
    return value;
}

std::string require_sha256(const gguf_context * gguf, const char * name) {
    const auto value = require_string(gguf, name);
    if (value.size() != 64 || value.find_first_not_of("0123456789abcdef") != std::string::npos)
        throw error("unsupported_model", std::string("Expected lowercase SHA-256: ") + name);
    return value;
}

struct manifest {
    size_t expected_full_attention_layers = 0;
    std::vector<int32_t> readouts;
    std::string readouts_json;
    std::string tokenizer_sha256;
    std::string decision_config_sha256;
    std::string weight_source;
    std::string provenance_json;
};

manifest inspect_gguf(const std::string & path) {
    std::unique_ptr<gguf_context, decltype(&gguf_free)> gguf(
        gguf_init_from_file(path.c_str(), {true, nullptr}), gguf_free);
    if (!gguf) throw error("model_open_failed", "Could not parse the GGUF file");
    require_string(gguf.get(), "general.architecture", "qwen35");
    require_string(gguf.get(), "glide.attention_mode", "noncausal_full_attention");
    require_string(gguf.get(), "glide.prompt_format", "glide-v1");
    require_string(gguf.get(), "glide.pooling", "last");
    require_string(gguf.get(), "glide.source_revision", kSourceRevision);
    require_string(gguf.get(), "glide.readout_dtype", "BF16");
    require_string(gguf.get(), "glide.readout_layout", "cls.output");
    require_string(gguf.get(), "glide.llama_cpp_commit", GLIDE_LLAMA_REVISION);
    if (gguf_get_val_u32(gguf.get(), require_key(gguf.get(), "qwen35.embedding_length_out", GGUF_TYPE_UINT32)) != 255 ||
        gguf_get_val_u32(gguf.get(), require_key(gguf.get(), "qwen35.pooling_type", GGUF_TYPE_UINT32)) != LLAMA_POOLING_TYPE_LAST)
        throw error("unsupported_model", "GLiDE requires embedding_length_out=255 and pooling_type=LAST");
    if (gguf_get_val_u32(gguf.get(), require_key(gguf.get(), "glide.format_version", GGUF_TYPE_UINT32)) != 1)
        throw error("unsupported_model", "Unsupported GLiDE format version");
    if (!gguf_get_val_bool(gguf.get(), require_key(gguf.get(), "glide.text_only", GGUF_TYPE_BOOL)))
        throw error("unsupported_model", "Only GLiDE text-only models are supported");
    if (gguf_get_val_bool(gguf.get(), require_key(gguf.get(), "qwen35.attention.causal", GGUF_TYPE_BOOL)))
        throw error("unsupported_model", "Model declares causal attention; GLiDE requires noncausal full attention");
    const std::set<std::string> allowed_glide = {
        "glide.format_version", "glide.attention_mode", "glide.prompt_format", "glide.pooling",
        "glide.source_revision", "glide.text_only", "glide.readout_token_ids_json", "glide.readout_dtype",
        "glide.tokenizer_sha256", "glide.decision_config_sha256", "glide.readout_layout",
        "glide.llama_cpp_commit", "glide.weight_source", "glide.provenance_json"};
    for (int64_t i = 0; i < gguf_get_n_kv(gguf.get()); ++i) {
        const std::string key = gguf_get_key(gguf.get(), i);
        if ((starts_with(key, "glide.") && !allowed_glide.count(key)) || starts_with(key, "decision.") ||
            ends_with(key, ".decision.type") || starts_with(key, "clip.") || starts_with(key, "vision.") ||
            starts_with(key, "audio.") || starts_with(key, "multimodal."))
            throw error("unsupported_model", "Unsupported GGUF metadata key: " + key);
    }
    bool has_output = false;
    for (int64_t i = 0; i < gguf_get_n_tensors(gguf.get()); ++i) {
        const std::string name = gguf_get_tensor_name(gguf.get(), i);
        if (starts_with(name, "v.") || starts_with(name, "vision.") || starts_with(name, "audio.") ||
            starts_with(name, "mm.") || starts_with(name, "mmproj.") ||
            (starts_with(name, "cls.") && name != "cls.output.weight") || name == "output.weight")
            throw error("unsupported_model", "Unsupported multimodal/classification tensor: " + name);
        if (name == "cls.output.weight") {
            has_output = true;
            if (gguf_get_tensor_type(gguf.get(), i) != GGML_TYPE_BF16 || gguf_get_tensor_ne(gguf.get(), i)[1] != 255)
                throw error("unsupported_model", "GLiDE cls.output.weight must contain exactly 255 BF16 readout rows");
        }
    }
    if (!has_output) throw error("unsupported_model", "Small GLiDE cls.output.weight is required");
    manifest result;
    const auto recurrent_key = require_key(gguf.get(), "qwen35.attention.recurrent_layers", GGUF_TYPE_ARRAY);
    const auto block_count = gguf_get_val_u32(gguf.get(), require_key(gguf.get(), "qwen35.block_count", GGUF_TYPE_UINT32));
    if (gguf_get_arr_type(gguf.get(), recurrent_key) != GGUF_TYPE_BOOL || block_count == 0 || block_count > 8192 ||
        gguf_get_arr_n(gguf.get(), recurrent_key) != block_count)
        throw error("unsupported_model", "Recurrent-layer metadata must be a typed bool array matching block count");
    const auto * recurrent = static_cast<const uint8_t *>(gguf_get_arr_data(gguf.get(), recurrent_key));
    for (uint32_t i = 0; i < block_count; ++i) {
        if (recurrent[i] > 1) throw error("unsupported_model", "Invalid recurrent-layer boolean");
        if (!recurrent[i]) ++result.expected_full_attention_layers;
    }
    if (!result.expected_full_attention_layers)
        throw error("unsupported_model", "At least one full-attention layer is required");
    result.readouts_json = require_string(gguf.get(), "glide.readout_token_ids_json");
    if (result.readouts_json.size() > 16384) throw error("unsupported_model", "GLiDE readout metadata is too long");
    result.readouts = glide::parse_readouts(glide::parse_json(result.readouts_json), std::numeric_limits<int32_t>::max());
    if (json(result.readouts).dump() != result.readouts_json)
        throw error("unsupported_model", "glide.readout_token_ids_json must use canonical compact JSON");
    result.tokenizer_sha256 = require_sha256(gguf.get(), "glide.tokenizer_sha256");
    result.decision_config_sha256 = require_sha256(gguf.get(), "glide.decision_config_sha256");
    if (result.tokenizer_sha256 != kTokenizerSha256 || result.decision_config_sha256 != kDecisionConfigSha256)
        throw error("unsupported_model", "Tokenizer or decision config SHA-256 does not match the pinned GLiDE revision");
    result.weight_source = require_string(gguf.get(), "glide.weight_source");
    if (result.weight_source != "original-bf16" && result.weight_source != "reconstructed-packed-effective-bf16")
        throw error("unsupported_model", "Unsupported GLiDE weight source");
    result.provenance_json = require_string(gguf.get(), "glide.provenance_json");
    if (result.provenance_json.size() > 16384)
        throw error("unsupported_model", "GLiDE provenance metadata exceeds 16 KiB");
    json provenance;
    try { provenance = glide::parse_json(result.provenance_json); }
    catch (const std::exception &) { throw error("unsupported_model", "Invalid GLiDE provenance JSON"); }
    if (!provenance.is_object() || !provenance.contains("kind") || provenance.at("kind") != result.weight_source ||
        !provenance.contains("revision") || provenance.at("revision") != kSourceRevision)
        throw error("unsupported_model", "GLiDE provenance must be an object with matching kind and source revision");
    return result;
}

std::string model_metadata(const llama_model * model, const char * key) {
    const auto length = llama_model_meta_val_str(model, key, nullptr, 0);
    if (length < 0 || length > 16384) throw error("unsupported_model", std::string("Missing/oversized loaded-model metadata: ") + key);
    std::vector<char> value(static_cast<size_t>(length) + 1);
    if (llama_model_meta_val_str(model, key, value.data(), value.size()) != length)
        throw error("unsupported_model", "Loaded-model metadata changed unexpectedly");
    return std::string(value.data(), static_cast<size_t>(length));
}

void verify_loaded_metadata(const llama_model * model, const manifest & info) {
    const std::vector<std::pair<const char *, std::string>> expected = {
        {"general.architecture", "qwen35"}, {"glide.format_version", "1"},
        {"glide.attention_mode", "noncausal_full_attention"}, {"glide.prompt_format", "glide-v1"},
        {"glide.pooling", "last"}, {"glide.source_revision", kSourceRevision},
        {"glide.text_only", "true"}, {"qwen35.attention.causal", "false"},
        {"glide.readout_dtype", "BF16"}, {"glide.readout_layout", "cls.output"},
        {"qwen35.embedding_length_out", "255"}, {"qwen35.pooling_type", "3"}, {"glide.readout_token_ids_json", info.readouts_json},
        {"glide.tokenizer_sha256", info.tokenizer_sha256}, {"glide.decision_config_sha256", info.decision_config_sha256},
        {"glide.llama_cpp_commit", GLIDE_LLAMA_REVISION}, {"glide.weight_source", info.weight_source},
        {"glide.provenance_json", info.provenance_json}};
    for (const auto & item : expected) {
        if (model_metadata(model, item.first) != item.second)
            throw error("unsupported_model", std::string("Loaded model metadata mismatch: ") + item.first);
    }
}

std::set<int32_t> find_multimodal_tokens(const llama_vocab * vocab) {
    std::set<int32_t> result;
    for (int32_t id = 0; id < llama_vocab_n_tokens(vocab); ++id) {
        const char * raw = llama_vocab_get_text(vocab, id);
        if (!raw) continue;
        const std::string text(raw);
        if (starts_with(text, "<|vision_") || starts_with(text, "<|image_") || starts_with(text, "<|video_") ||
            starts_with(text, "<|audio") || starts_with(text, "<|text_audio") || starts_with(text, "<|spk_") ||
            text == "<|end_of_task|>" || text == "<|speech|>") result.insert(id);
    }
    return result;
}

struct backend_guard {
    backend_guard() { llama_backend_init(); }
    ~backend_guard() { llama_backend_free(); }
};

struct memory_guard {
    llama_memory_t memory;
    glide::execution_observer & observer;
    bool cleared_after = false;
    memory_guard(llama_context * ctx, glide::execution_observer & obs) : memory(llama_get_memory(ctx)), observer(obs) {
        if (!memory) throw error("unsupported_model", "Qwen3.5 hybrid memory is unavailable");
        llama_memory_clear(memory, true);
        observer.reset_before = true;
    }
    void clear_after() {
        if (!cleared_after) {
            llama_memory_clear(memory, true);
            cleared_after = true;
            observer.reset_after = true;
        }
        observer.active = false;
    }
    ~memory_guard() { clear_after(); }
};

struct deadline {
    clock_type::time_point end = clock_type::time_point::max();
    bool expired = false;
    static bool check(void * data) {
        auto & value = *static_cast<deadline *>(data);
        value.expired = clock_type::now() >= value.end;
        return value.expired;
    }
};

double elapsed_ms(clock_type::time_point start) {
    return std::chrono::duration<double, std::milli>(clock_type::now() - start).count();
}

json evaluate(llama_context * ctx, const glide::request & request, const options & opt, deadline & timer, glide::execution_observer & observer, size_t expected_layers) {
    const auto started = clock_type::now();
    observer.begin();
    memory_guard state(ctx, observer); // Clear both recurrent and KV storage, even after an exception/abort.
    if (llama_get_causal_attn(ctx)) throw error("unsafe_context", "Context became causal; refusing evaluation");
    if (request.token_ids.size() > llama_n_batch(ctx) || request.token_ids.size() > llama_n_ubatch(ctx) ||
        request.token_ids.size() > llama_n_ctx(ctx))
        throw error("unsafe_context", "Full noncausal prompt does not fit a single physical batch");

    // Own all batch storage; no per-request llama_batch_init allocation is needed.
    std::vector<llama_pos> positions(request.token_ids.size());
    std::vector<int32_t> counts(request.token_ids.size(), 1);
    std::vector<llama_seq_id> sequence_ids(request.token_ids.size(), 0);
    std::vector<llama_seq_id *> sequences(request.token_ids.size());
    std::vector<int8_t> outputs(request.token_ids.size(), 1); // LAST pooling needs every positional row.
    for (size_t i = 0; i < positions.size(); ++i) { positions[i] = static_cast<llama_pos>(i); sequences[i] = &sequence_ids[i]; }
    llama_batch batch{};
    batch.n_tokens = static_cast<int32_t>(request.token_ids.size());
    batch.token = const_cast<llama_token *>(request.token_ids.data());
    batch.pos = positions.data();
    batch.n_seq_id = counts.data();
    batch.seq_id = sequences.data();
    batch.logits = outputs.data();

    timer.end = clock_type::now() + std::chrono::milliseconds(opt.timeout_ms);
    timer.expired = false;
    const auto decode_start = clock_type::now();
    ++observer.decode_calls;
    const int status = llama_decode(ctx, batch); // EXACTLY ONE whole-prompt call. Never split or reuse prefix state.
    if (status != 0) {
        if (timer.expired) throw error("decode_timeout", "Cooperative CPU timeout reached; increase --timeout-ms if appropriate");
        throw error("decode_failed", "llama_decode returned status " + std::to_string(status));
    }
    float * raw_logits = llama_get_embeddings_seq(ctx, 0); // Raw small-head projection, not normalized embeddings.
    const double decode_ms = elapsed_ms(decode_start);
    if (!raw_logits) throw error("missing_logits", "Last-position logits are unavailable");
    std::vector<float> logits;
    logits.reserve(request.readout_token_ids.size());
    for (size_t i = 0; i < glide::kReadoutSize; ++i) {
        const float value = raw_logits[i];
        if (!std::isfinite(value)) throw error("nonfinite_logits", "Selected last-position logits contain NaN or infinity");
        logits.push_back(value);
    }
    state.clear_after();
    const auto evidence = observer.finish(request.token_ids.size(), expected_layers, opt.gpu_layers != 0);
    return {{"id", request.id}, {"ok", true}, {"execution_evidence", evidence}, {"logits", logits}, {"prompt_tokens", request.token_ids.size()},
        {"decode_ms", decode_ms}, {"total_ms", elapsed_ms(started)}};
}

int run(const options & opt) {
    const auto info = inspect_gguf(opt.model_path);
    backend_guard backend;
    auto params = llama_model_default_params();
    ggml_backend_dev_t cpu_only_devices[] = {nullptr};
    params.devices = opt.gpu_layers == 0 ? cpu_only_devices : nullptr;
    params.n_gpu_layers = opt.gpu_layers;
    params.split_mode = LLAMA_SPLIT_MODE_NONE;
    params.load_mtp = false;
    params.check_tensors = true;
    // Anchored override catches both token embedding and the otherwise-unused tied
    // vocabulary duplicate. Keep storage alive until after model destruction.
    llama_model_tensor_buft_override host_embedding[] = {
        {"^token_embd\\.weight$", ggml_backend_cpu_buffer_type()}, {nullptr, nullptr}};
    params.tensor_buft_overrides = host_embedding;
    std::unique_ptr<llama_model, decltype(&llama_model_free)> model(
        llama_model_load_from_file(opt.model_path.c_str(), params), llama_model_free);
    if (!model) throw error("model_load_failed", "llama.cpp could not load model weights");
    verify_loaded_metadata(model.get(), info);
    if (llama_model_has_encoder(model.get()) || !llama_model_has_decoder(model.get()))
        throw error("unsupported_model", "Only decoder-style Qwen3.5 GLiDE graphs are supported");
    if (opt.ctx_size > static_cast<uint32_t>(llama_model_n_ctx_train(model.get())))
        throw error("unsupported_context", "--ctx-size exceeds the exported model training context; extrapolation is not supported");
    const auto * vocab = llama_model_get_vocab(model.get());
    const int32_t vocab_size = llama_vocab_n_tokens(vocab);
    glide::parse_readouts(json(info.readouts), vocab_size);
    const auto multimodal_tokens = find_multimodal_tokens(vocab);

    deadline timer;
    glide::execution_observer observer;
    auto context_params = llama_context_default_params();
    context_params.n_ctx = opt.ctx_size;
    context_params.n_batch = opt.ctx_size;
    context_params.n_ubatch = opt.ctx_size;
    context_params.n_seq_max = 1;
    context_params.n_rs_seq = 0;
    context_params.n_outputs_max = opt.ctx_size;
    context_params.n_outputs_max_per_seq = opt.ctx_size;
    context_params.n_threads = opt.threads;
    context_params.n_threads_batch = opt.threads;
    context_params.pooling_type = LLAMA_POOLING_TYPE_LAST; // Only select the last raw 255-way projection; no normalization.
    context_params.attention_type = LLAMA_ATTENTION_TYPE_UNSPECIFIED; // Honor validated model metadata, then assert it.
    context_params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    context_params.type_k = context_params.type_v = GGML_TYPE_F16;
    context_params.cb_eval = glide::execution_observer::callback;
    context_params.cb_eval_user_data = &observer;
    context_params.embeddings = true;
    context_params.offload_kqv = opt.gpu_layers != 0;
    context_params.op_offload = opt.gpu_layers != 0;
    context_params.no_perf = false;
    context_params.abort_callback = deadline::check;
    context_params.abort_callback_data = &timer;
    std::unique_ptr<llama_context, decltype(&llama_free)> ctx(llama_init_from_model(model.get(), context_params), llama_free);
    if (!ctx) throw error("context_init_failed", "Could not initialize the full-batch context");
    if (llama_get_causal_attn(ctx.get()) || llama_pooling_type(ctx.get()) != LLAMA_POOLING_TYPE_LAST ||
        llama_model_n_embd_out(model.get()) != 255 || llama_n_batch(ctx.get()) < opt.ctx_size || llama_n_ubatch(ctx.get()) < opt.ctx_size)
        throw error("unsafe_context", "Context is causal or cannot hold the configured prompt in one physical batch");

    std::cerr << json({{"event", "ready"}, {"backend_profile", glide::kBackendProfile},
        {"flash_attention", "enabled"}, {"token_embedding_override", {{"pattern", "^token_embd\\.weight$"}, {"buffer_type", "CPU"}}},
        {"expected_full_attention_layers", info.expected_full_attention_layers},
        {"evidence_contract_sha256", GLIDE_EVIDENCE_CONTRACT_SHA256}, {"llama_revision", GLIDE_LLAMA_REVISION}, {"ctx_size", opt.ctx_size},
        {"n_batch", llama_n_batch(ctx.get())}, {"n_ubatch", llama_n_ubatch(ctx.get())}, {"causal_attention", false},
        {"timeout_ms", opt.timeout_ms}, {"threads", opt.threads}, {"backend", opt.gpu_layers == 0 ? "cpu" : "cuda"}, {"gpu_layers", opt.gpu_layers},
        {"tokenizer_sha256", info.tokenizer_sha256}, {"decision_config_sha256", info.decision_config_sha256}}).dump() << '\n';

    // Bound memory while consuming untrusted lines. Oversize errors are emitted immediately,
    // then the remainder of that same line is discarded until its newline/EOF.
    std::string line;
    line.reserve(65536);
    bool discarding = false;
    auto respond = [&](const std::string & input) {
        json id = nullptr;
        try {
            const auto value = glide::parse_json(input);
            id = glide::request_id(value);
            const auto request = glide::validate_request(value, vocab_size, opt.ctx_size, info.readouts, multimodal_tokens);
            std::cout << evaluate(ctx.get(), request, opt, timer, observer, info.expected_full_attention_layers).dump() << '\n' << std::flush;
        } catch (const error & e) {
            std::cout << glide::error_response(id, e.code, e.what()).dump() << '\n' << std::flush;
        } catch (const std::exception & e) {
            std::cerr << json({{"event", "request_exception"}, {"message", e.what()}}).dump() << '\n';
            std::cout << glide::error_response(id, "internal_error", "Request failed; see stderr diagnostics").dump() << '\n' << std::flush;
        }
    };
    char ch;
    while (std::cin.get(ch)) {
        if (ch == '\n') {
            if (!discarding) respond(line);
            line.clear();
            discarding = false;
        } else if (!discarding) {
            if (line.size() == glide::kMaxLineBytes) {
                std::cout << glide::error_response(nullptr, "request_too_large", "JSON line exceeds 4 MiB").dump() << '\n' << std::flush;
                line.clear();
                discarding = true;
            } else line.push_back(ch);
        }
    }
    if (!discarding && !line.empty()) respond(line);
    if (std::cin.bad()) throw error("input_failed", "Error reading standard input");
    return 0;
}
} // namespace

int main(int argc, char ** argv) {
    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);
    try { return run(parse_options(argc, argv)); }
    catch (const error & e) {
        std::cerr << json({{"event", "fatal"}, {"code", e.code}, {"message", e.what()}}).dump() << '\n';
    } catch (const std::exception & e) {
        std::cerr << json({{"event", "fatal"}, {"code", "internal_error"}, {"message", e.what()}}).dump() << '\n';
    }
    return 2;
}
