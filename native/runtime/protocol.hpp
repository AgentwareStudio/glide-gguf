#pragma once

#include "nlohmann/json.hpp"
#include <cstdint>
#include <limits>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace glide {
using json = nlohmann::json;
constexpr size_t kReadoutSize = 255;
constexpr size_t kMaxLineBytes = 4 * 1024 * 1024;

struct error : std::runtime_error {
    std::string code;
    error(std::string c, const std::string & m) : std::runtime_error(m), code(std::move(c)) {}
};

inline json parse_json(const std::string & line) {
    if (line.size() > kMaxLineBytes) throw error("request_too_large", "JSON line exceeds 4 MiB");
    std::set<std::string> keys;
    try {
        return json::parse(line, [&](int depth, json::parse_event_t event, json & parsed) {
            if (depth > 8) throw error("invalid_request", "JSON nesting exceeds protocol limit");
            if (event == json::parse_event_t::key && depth == 1 && !keys.insert(parsed.get<std::string>()).second)
                throw error("invalid_request", "Duplicate top-level JSON field");
            return true;
        });
    } catch (const json::exception &) {
        throw error("invalid_json", "Expected one valid JSON object per line");
    }
}

inline int32_t token_id(const json & item, int32_t vocab_size, const char * field) {
    if (!item.is_number_integer()) throw error("invalid_request", std::string(field) + " must contain integers, not booleans or floats");
    if ((item.is_number_unsigned() && item.get<uint64_t>() > uint64_t(std::numeric_limits<int32_t>::max())) ||
        (!item.is_number_unsigned() && (item.get<int64_t>() < 0 || item.get<int64_t>() > std::numeric_limits<int32_t>::max())))
        throw error("invalid_token", std::string(field) + " contains an out-of-range token ID");
    const auto id = item.get<int32_t>();
    if (id >= vocab_size) throw error("invalid_token", std::string(field) + " contains a token outside the model vocabulary");
    return id;
}

inline std::vector<int32_t> parse_readouts(const json & value, int32_t vocab_size) {
    if (!value.is_array() || value.size() != kReadoutSize)
        throw error("invalid_readouts", "readout_token_ids must contain exactly 255 entries");
    std::vector<int32_t> result;
    std::set<int32_t> seen;
    for (const auto & item : value) {
        auto id = token_id(item, vocab_size, "readout_token_ids");
        if (!seen.insert(id).second) throw error("invalid_readouts", "readout_token_ids contains duplicates");
        result.push_back(id);
    }
    return result;
}

inline json request_id(const json & value) {
    if (!value.is_object() || !value.contains("id")) return nullptr;
    const auto & id = value.at("id");
    if ((id.is_string() && id.get_ref<const std::string &>().size() <= 256) || id.is_number_integer()) return id;
    return nullptr;
}

struct request {
    json id;
    std::vector<int32_t> token_ids;
    std::vector<int32_t> readout_token_ids;
};

inline request validate_request(const json & value, int32_t vocab_size, uint32_t max_tokens,
        const std::vector<int32_t> & canonical_readouts, const std::set<int32_t> & multimodal_tokens = {}) {
    if (!value.is_object()) throw error("invalid_request", "Request must be an object");
    for (const auto & item : value.items()) {
        if (item.key() != "id" && item.key() != "token_ids" && item.key() != "readout_token_ids")
            throw error("unsupported_field", "Only id, token_ids, and readout_token_ids are supported; multimodal inputs are not supported");
    }
    if (!value.contains("id") || request_id(value).is_null())
        throw error("invalid_request", "id must be an integer or a string of at most 256 bytes");
    if (!value.contains("token_ids") || !value.at("token_ids").is_array())
        throw error("invalid_request", "token_ids must be an array");
    const auto & tokens = value.at("token_ids");
    if (tokens.empty()) throw error("empty_prompt", "token_ids must not be empty");
    if (tokens.size() > max_tokens)
        throw error("prompt_too_long", "Prompt exceeds --ctx-size; noncausal prompts must never be split");
    if (!value.contains("readout_token_ids")) throw error("invalid_readouts", "readout_token_ids is required");
    request req{value.at("id"), {}, parse_readouts(value.at("readout_token_ids"), vocab_size)};
    if (req.readout_token_ids != canonical_readouts)
        throw error("readout_mismatch", "readout_token_ids must exactly match the model's canonical order");
    req.token_ids.reserve(tokens.size());
    for (const auto & item : tokens) {
        auto id = token_id(item, vocab_size, "token_ids");
        if (multimodal_tokens.count(id)) throw error("multimodal_unsupported", "Prompt contains a multimodal control token");
        req.token_ids.push_back(id);
    }
    return req;
}

inline json error_response(const json & id, const std::string & code, const std::string & message) {
    return {{"id", id}, {"ok", false}, {"error", {{"code", code}, {"message", message}}}};
}
} // namespace glide
