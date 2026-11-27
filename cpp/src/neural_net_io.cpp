// SPDX-License-Identifier: MPL-2.0
// Copyright (C) 2026 Mark Higgins
//
// Binary weight persistence (legacy and versioned NGWT format) for NeuralNetwork.
#include "bgbot/neural_net.h"
#include "bgbot/encoding.h"
#include <cstdint>
#include <cstring>
#include <fstream>
#include <istream>
#include <ostream>
#include <string>
#include <vector>

namespace bgbot {

// ======================== Persistence ========================

namespace {

constexpr char WEIGHT_MAGIC[] = {'N', 'G', 'W', 'T'};
constexpr uint32_t WEIGHT_SCHEMA = 1;

uint32_t encoder_for_width(int width) {
    switch (width) {
        case TESAURO_INPUTS: return 1;
        case EXTENDED_CONTACT_INPUTS: return 2;
        case 214: return 3;  // legacy contact/crashed input prefix
        default: return 0;
    }
}

void write_word(std::ostream& out, uint32_t value) {
    const unsigned char bytes[] = {
        static_cast<unsigned char>(value), static_cast<unsigned char>(value >> 8),
        static_cast<unsigned char>(value >> 16), static_cast<unsigned char>(value >> 24),
    };
    out.write(reinterpret_cast<const char*>(bytes), sizeof(bytes));
}

uint32_t read_word(std::istream& in) {
    unsigned char bytes[4]{};
    in.read(reinterpret_cast<char*>(bytes), sizeof(bytes));
    return static_cast<uint32_t>(bytes[0]) | static_cast<uint32_t>(bytes[1]) << 8 |
           static_cast<uint32_t>(bytes[2]) << 16 | static_cast<uint32_t>(bytes[3]) << 24;
}

} // namespace

bool NeuralNetwork::save_weights(const std::string& filepath, bool versioned) const {
    const uint32_t encoder = encoder_for_width(n_inputs_);
    if (versioned && encoder == 0) return false;
    std::ofstream f(filepath, std::ios::binary);
    if (!f) return false;

    if (versioned) {
        f.write(WEIGHT_MAGIC, sizeof(WEIGHT_MAGIC));
        write_word(f, WEIGHT_SCHEMA);
        write_word(f, encoder);
    }
    write_word(f, n_hidden_);
    write_word(f, n_inputs_);
    write_word(f, NN_OUTPUTS);
    f.write(reinterpret_cast<const char*>(hidden_weights_.data()),
            hidden_weights_.size() * sizeof(float));
    f.write(reinterpret_cast<const char*>(output_weights_.data()),
            output_weights_.size() * sizeof(float));
    f.close();
    return f.good();
}

bool NeuralNetwork::load_weights(const std::string& filepath) {
    std::ifstream f(filepath, std::ios::binary);
    if (!f) return false;

    char prefix[sizeof(WEIGHT_MAGIC)];
    if (!f.read(prefix, sizeof(prefix))) return false;
    const bool versioned = std::memcmp(prefix, WEIGHT_MAGIC, sizeof(prefix)) == 0;
    if (versioned) {
        const auto schema = read_word(f);
        const auto encoder = read_word(f);
        if (schema != WEIGHT_SCHEMA || encoder == 0 ||
            encoder != encoder_for_width(n_inputs_)) return false;
    } else {
        f.seekg(0);
    }
    const auto hidden = read_word(f);
    const auto inputs = read_word(f);
    const auto outputs = read_word(f);
    if (!f || hidden != static_cast<uint32_t>(n_hidden_) ||
        inputs != static_cast<uint32_t>(n_inputs_) || outputs != NN_OUTPUTS) {
        return false;
    }

    // Complete validation before replacing a previously usable network.
    std::vector<float> hidden_weights(hidden_weights_.size());
    std::vector<float> output_weights(output_weights_.size());
    f.read(reinterpret_cast<char*>(hidden_weights.data()), hidden_weights.size() * sizeof(float));
    f.read(reinterpret_cast<char*>(output_weights.data()), output_weights.size() * sizeof(float));
    if (!f || (versioned && f.peek() != std::char_traits<char>::eof())) return false;
    hidden_weights_ = std::move(hidden_weights);
    output_weights_ = std::move(output_weights);
    build_transposed_weights();
    transposed_weights_valid_ = true;
    return true;
}


} // namespace bgbot
