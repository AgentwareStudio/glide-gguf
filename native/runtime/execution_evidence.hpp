#pragma once
#include "ggml.h"
#include "ggml-backend.h"
#include "protocol.hpp"
#include <cstring>
#include <cstdio>
#include <set>
#include <string>

namespace glide {
constexpr const char * kBackendProfile = "glide-fa-hostemb-v1";

// Public cb_eval is called with ask=false only after a successful compute and
// backend synchronization. No pre-execution reservation is execution evidence.
struct execution_observer {
    bool active = false;
    bool callback_failed = false;
    char callback_error[256]{};
    bool reset_before = false;
    bool reset_after = false;
    int decode_calls = 0;
    json fa = json::array();
    json classifier = json::array();
    json embedding = json::array();
    json vocabulary = json::array();

    void begin() {
        active = false;
        callback_failed = reset_before = reset_after = false;
        callback_error[0] = 0;
        decode_calls = 0;
        fa = json::array(); classifier = json::array();
        embedding = json::array(); vocabulary = json::array();
        active = true;
    }

    static const char * device_type(enum ggml_backend_dev_type type) {
        switch (type) {
            case GGML_BACKEND_DEVICE_TYPE_CPU: return "CPU";
            case GGML_BACKEND_DEVICE_TYPE_GPU: return "GPU";
            case GGML_BACKEND_DEVICE_TYPE_IGPU: return "IGPU";
            case GGML_BACKEND_DEVICE_TYPE_ACCEL: return "ACCEL";
            default: return "UNKNOWN";
        }
    }

    static ggml_backend_buffer_type_t mapped_cpu_type() {
        // Pinned upstream leaves CPU and CPU_Mapped buft.device null. Obtain the
        // exact public API type identity, not a name-based fallback or request flag.
        static auto type = [] {
            auto probe = ggml_backend_cpu_buffer_from_ptr(nullptr, 0);
            if (!probe) throw error("execution_evidence_failed", "Could not identify mapped CPU buffer type");
            auto result = ggml_backend_buffer_get_type(probe);
            ggml_backend_buffer_free(probe);
            return result;
        }();
        return type;
    }

    static json device(const ggml_tensor * tensor) {
        if (!tensor) throw error("execution_evidence_failed", "Missing observed tensor");
        // Views use their owning allocation, never a requested/guessed backend.
        auto buffer = tensor->buffer;
        for (const auto * view = tensor->view_src; !buffer && view; view = view->view_src) buffer = view->buffer;
        if (!buffer) throw error("execution_evidence_failed", "Observed tensor has no allocated buffer");
        const auto buft = ggml_backend_buffer_get_type(buffer);
        auto dev = ggml_backend_buft_get_device(buft);
        if (!dev && (buft == ggml_backend_cpu_buffer_type() || buft == mapped_cpu_type()))
            dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
        if (!dev) throw error("execution_evidence_failed", "Observed allocation has no device");
        const auto reg = ggml_backend_dev_backend_reg(dev);
        if (!reg) throw error("execution_evidence_failed", "Observed device has no backend registry");
        if (std::strcmp(ggml_backend_reg_name(reg), "CUDA") == 0 && ggml_backend_buft_is_host(buft))
            throw error("execution_evidence_failed", "CUDA host allocation is not GPU execution evidence");
        return {{"backend", ggml_backend_reg_name(reg)}, {"type", device_type(ggml_backend_dev_type(dev))},
                {"name", ggml_backend_dev_name(dev)}, {"buffer", ggml_backend_buft_name(buft)}};
    }

    static const char * weight_name(const ggml_tensor * tensor) noexcept {
        if (!tensor) return nullptr;
        // Scheduler copies retain the canonical source name between # delimiters.
        // Reporting their actual copied allocation prevents misreporting CPU placement.
        for (const char * canonical : {"cls.output.weight", "token_embd.weight"}) {
            if (std::strcmp(tensor->name, canonical) == 0) return canonical;
            const char * sep = std::strchr(tensor->name, '#');
            if (sep && std::strncmp(sep + 1, canonical, std::strlen(canonical)) == 0 &&
                sep[1 + std::strlen(canonical)] == '#') return canonical;
        }
        return nullptr;
    }

    static bool wanted(const ggml_tensor * node) noexcept {
        // Scheduler callbacks also visit order-only/inactive select branches.
        // Backends skip these nodes; a callback alone is not proof of execution.
        if (!(node->flags & GGML_TENSOR_FLAG_COMPUTE)) return false;
        if (node->op == GGML_OP_FLASH_ATTN_EXT) return true;
        if (node->op != GGML_OP_MUL_MAT && node->op != GGML_OP_GET_ROWS) return false;
        return weight_name(node->src[0]) != nullptr;
    }

    static bool callback(ggml_tensor * node, bool ask, void * opaque) noexcept {
        auto & self = *static_cast<execution_observer *>(opaque);
        if (!self.active || !node) return false;
        if (ask) return wanted(node);
        try {
            if (!wanted(node)) return true;
            if (!node->name[0]) throw error("execution_evidence_failed", "Unnamed observed node");
            json record = {{"name", node->name}, {"op", ggml_op_name(node->op)},
                {"output_device", device(node)}, {"phase", "post_execution"}, {"executed", true}};
            if (node->op == GGML_OP_FLASH_ATTN_EXT) self.fa.push_back(std::move(record));
            else {
                const auto * weight = node->src[0];
                const auto * name = weight_name(weight);
                record["weight_name"] = name;
                record["weight_device"] = device(weight);
                if (std::strcmp(name, "cls.output.weight") == 0 && node->op == GGML_OP_MUL_MAT)
                    self.classifier.push_back(std::move(record));
                else if (std::strcmp(name, "token_embd.weight") == 0 && node->op == GGML_OP_GET_ROWS)
                    self.embedding.push_back(std::move(record));
                else if (std::strcmp(name, "token_embd.weight") == 0 && node->op == GGML_OP_MUL_MAT)
                    self.vocabulary.push_back(std::move(record));
                else self.callback_failed = true;
            }
        } catch (const std::exception & exc) {
            self.callback_failed = true;
            std::snprintf(self.callback_error, sizeof(self.callback_error), "%s (%s)", exc.what(), node->name);
        } catch (...) {
            // Never unwind into C or allocate memory while recording failure.
            self.callback_failed = true;
        }
        return true; // Do not truncate later computation on callback failure.
    }

    static bool cpu(const json & dev) { return dev.at("backend") == "CPU" && dev.at("type") == "CPU"; }
    static bool cuda(const json & dev) {
        return dev.at("backend") == "CUDA" && (dev.at("type") == "GPU" || dev.at("type") == "IGPU");
    }

    json finish(size_t prompt_tokens, size_t expected_layers, bool requested_cuda) {
        active = false;
        auto require = [](bool yes, const char * message) {
            if (!yes) throw error("execution_evidence_failed", message);
        };
        if (callback_failed) throw error("execution_evidence_failed", callback_error[0] ? callback_error : "Post-execution callback could not attest executed allocations");
        require(reset_before && reset_after && decode_calls == 1, "State reset or decode-count evidence missing");
        require(fa.size() == expected_layers && expected_layers > 0, "Unexpected number of executed Flash Attention nodes");
        std::set<std::string> names;
        for (const auto & node : fa) {
            require(names.insert(node.at("name").get<std::string>()).second, "Duplicate executed Flash Attention node");
            require(requested_cuda ? cuda(node.at("output_device")) : cpu(node.at("output_device")),
                    "Flash Attention executed on the wrong backend");
        }
        if (classifier.size() != 1 || embedding.size() != 1 || !vocabulary.empty())
            throw error("execution_evidence_failed", "Expected exactly one classifier and lookup with no vocabulary matmul; observed " +
                std::to_string(classifier.size()) + "/" + std::to_string(embedding.size()) + "/" + std::to_string(vocabulary.size()));
        for (const auto * field : {"output_device", "weight_device"})
            require(requested_cuda ? cuda(classifier[0].at(field)) : cpu(classifier[0].at(field)),
                    "Classifier did not execute with weights and output on the requested backend");
        require(cpu(embedding[0].at("weight_device")), "Token embedding weights are not on CPU");
        require(requested_cuda ? (cpu(embedding[0].at("output_device")) || cuda(embedding[0].at("output_device"))) : cpu(embedding[0].at("output_device")),
                "Token lookup output is not on an admitted backend");
        return {{"schema", "glide.native_execution_evidence.v1"}, {"backend_profile", kBackendProfile},
            {"requested_backend", requested_cuda ? "cuda" : "cpu"}, {"decode_calls", decode_calls},
            {"prompt_tokens", prompt_tokens}, {"expected_full_attention_layers", expected_layers},
            {"state_reset_before", reset_before}, {"state_reset_after", reset_after},
            {"flash_attention_nodes", fa}, {"classifier_nodes", classifier},
            {"token_embedding_nodes", embedding}, {"vocabulary_matmul_nodes", vocabulary}};
    }
};
} // namespace glide
