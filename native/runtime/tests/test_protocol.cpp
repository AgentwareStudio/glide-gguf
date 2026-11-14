#include "protocol.hpp"
#include <functional>
#include <iostream>
#include <numeric>

int main() {
    using glide::json;
    size_t count = 0;
    auto check = [&](bool condition, const char * message) {
        if (!condition) throw std::runtime_error(message);
        ++count;
    };
    auto rejects = [&](const std::function<void()> & fn, const std::string & code) {
        try { fn(); } catch (const glide::error & e) { check(e.code == code, e.what()); return; }
        throw std::runtime_error("Expected rejection: " + code);
    };
    std::vector<int32_t> ids(255);
    std::iota(ids.begin(), ids.end(), 0);
    const json good = {{"id", "sample"}, {"token_ids", {1, 7, 9}}, {"readout_token_ids", ids}};
    auto valid = [&](const json & value) { return glide::validate_request(value, 1000, 16, ids, {900}); };
    auto req = valid(glide::parse_json(good.dump()));
    check(req.id == "sample" && req.token_ids == std::vector<int32_t>({1, 7, 9}) && req.readout_token_ids == ids, "Valid round trip");
    auto changed = good;
    changed["id"] = -5; check(valid(changed).id == -5, "Integer ID");
    changed["id"] = 18446744073709551615ULL; check(valid(changed).id.is_number_unsigned(), "Unsigned ID preserved");
    for (const auto & value : {json(), json(true), json(1.5), json::object(), json::array(), json(std::string(257, 'x'))}) {
        changed = good; changed["id"] = value;
        rejects([&] { valid(changed); }, "invalid_request");
    }
    changed = good; changed.erase("id"); rejects([&] { valid(changed); }, "invalid_request");
    changed = good; changed["token_ids"] = json::array(); rejects([&] { valid(changed); }, "empty_prompt");
    changed = good; changed["token_ids"] = std::vector<int>(17, 1); rejects([&] { valid(changed); }, "prompt_too_long");
    for (const auto & value : {json(-1), json(1000), json(18446744073709551615ULL)}) {
        changed = good; changed["token_ids"][0] = value; rejects([&] { valid(changed); }, "invalid_token");
    }
    for (const auto & value : {json(true), json(1.0), json("1"), json(nullptr)}) {
        changed = good; changed["token_ids"][0] = value; rejects([&] { valid(changed); }, "invalid_request");
    }
    changed = good; changed["token_ids"][0] = 900; rejects([&] { valid(changed); }, "multimodal_unsupported");
    changed = good; changed["images"] = json::array(); rejects([&] { valid(changed); }, "unsupported_field");
    changed = good; changed["readout_token_ids"][1] = 0; rejects([&] { valid(changed); }, "invalid_readouts");
    changed = good; changed["readout_token_ids"][0] = 999; rejects([&] { valid(changed); }, "readout_mismatch");
    changed = good; changed["readout_token_ids"].erase(0); rejects([&] { valid(changed); }, "invalid_readouts");
    changed = good; changed.erase("readout_token_ids"); rejects([&] { valid(changed); }, "invalid_readouts");
    rejects([&] { glide::parse_json("{bad}"); }, "invalid_json");
    rejects([&] { glide::parse_json("{\"id\":1,\"id\":2}"); }, "invalid_request");
    rejects([&] { glide::parse_json("[[[[[[[[[[0]]]]]]]]]]"); }, "invalid_request");
    rejects([&] { glide::parse_json(std::string(glide::kMaxLineBytes + 1, ' ')); }, "request_too_large");
    rejects([&] { valid(json::array()); }, "invalid_request");
    check(glide::request_id(json({{"id", true}})).is_null(), "Boolean ID cannot be reflected");
    check(glide::error_response("a", "x", "y")["error"]["code"] == "x", "Error shape");
    std::cout << count << " protocol assertions passed\n";
}
