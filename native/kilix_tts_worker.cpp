// Persistent, pipe-framed Pocket TTS worker for the Kilix Vulkan runtime.
//
// This source is built against the exact admitted llama.cpp closure.  It keeps
// the model resident but clears request-specific model, sampler, and audio
// state before every utterance.  Standard output is protocol-only; callers
// must launch with llama.cpp logging disabled.

#include "arg.h"
#include "common.h"
#include "log.h"
#include "sampling.h"
#include "llama.h"
#include "llama-cpp.h"
#include "mtmd.h"
#include "mtmd-helper.h"

#include <array>
#include <cerrno>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

#include <unistd.h>

namespace {

constexpr std::array<uint8_t, 8> READY = {'K', 'I', 'L', 'I', 'X', 'T', '1', '\n'};
constexpr std::array<uint8_t, 4> REQUEST_MAGIC = {'K', 'X', 'Q', '1'};
constexpr std::array<uint8_t, 4> RESPONSE_MAGIC = {'K', 'X', 'R', '1'};
constexpr uint32_t MAX_TEXT_BYTES = 4096;
constexpr uint64_t MAX_WAV_BYTES = 64ULL * 1024ULL * 1024ULL;
constexpr int32_t REQUIRED_SAMPLE_RATE = 24000;
constexpr int64_t MAX_AUDIO_SAMPLES = 120LL * REQUIRED_SAMPLE_RATE;

enum class read_result {
    ok,
    eof,
    error,
};

read_result read_exact(void * output, size_t length, bool allow_initial_eof) {
    auto * cursor = static_cast<uint8_t *>(output);
    size_t offset = 0;
    while (offset < length) {
        const ssize_t count = ::read(STDIN_FILENO, cursor + offset, length - offset);
        if (count > 0) {
            offset += static_cast<size_t>(count);
            continue;
        }
        if (count == 0) {
            return allow_initial_eof && offset == 0 ? read_result::eof : read_result::error;
        }
        if (errno == EINTR) {
            continue;
        }
        return read_result::error;
    }
    return read_result::ok;
}

bool write_exact(const void * input, size_t length) {
    const auto * cursor = static_cast<const uint8_t *>(input);
    size_t offset = 0;
    while (offset < length) {
        const ssize_t count = ::write(STDOUT_FILENO, cursor + offset, length - offset);
        if (count > 0) {
            offset += static_cast<size_t>(count);
            continue;
        }
        if (count < 0 && errno == EINTR) {
            continue;
        }
        return false;
    }
    return true;
}

uint32_t decode_u32(const uint8_t * value) {
    return static_cast<uint32_t>(value[0]) |
           (static_cast<uint32_t>(value[1]) << 8U) |
           (static_cast<uint32_t>(value[2]) << 16U) |
           (static_cast<uint32_t>(value[3]) << 24U);
}

void encode_u32(uint8_t * output, uint32_t value) {
    for (unsigned index = 0; index < 4; ++index) {
        output[index] = static_cast<uint8_t>((value >> (8U * index)) & 0xffU);
    }
}

void encode_u64(uint8_t * output, uint64_t value) {
    for (unsigned index = 0; index < 8; ++index) {
        output[index] = static_cast<uint8_t>((value >> (8U * index)) & 0xffU);
    }
}

bool valid_utf8(const std::string & value) {
    size_t index = 0;
    while (index < value.size()) {
        const uint8_t lead = static_cast<uint8_t>(value[index]);
        uint32_t codepoint = 0;
        size_t continuation = 0;
        if (lead <= 0x7fU) {
            if (lead == 0) {
                return false;
            }
            ++index;
            continue;
        }
        if (lead >= 0xc2U && lead <= 0xdfU) {
            codepoint = lead & 0x1fU;
            continuation = 1;
        } else if (lead >= 0xe0U && lead <= 0xefU) {
            codepoint = lead & 0x0fU;
            continuation = 2;
        } else if (lead >= 0xf0U && lead <= 0xf4U) {
            codepoint = lead & 0x07U;
            continuation = 3;
        } else {
            return false;
        }
        if (index + continuation >= value.size()) {
            return false;
        }
        for (size_t offset = 1; offset <= continuation; ++offset) {
            const uint8_t byte = static_cast<uint8_t>(value[index + offset]);
            if ((byte & 0xc0U) != 0x80U) {
                return false;
            }
            codepoint = (codepoint << 6U) | (byte & 0x3fU);
        }
        if ((continuation == 2 && codepoint < 0x800U) ||
            (continuation == 3 && codepoint < 0x10000U) ||
            (codepoint >= 0xd800U && codepoint <= 0xdfffU) ||
            codepoint > 0x10ffffU) {
            return false;
        }
        index += continuation + 1;
    }
    return true;
}

bool send_response(uint32_t status, const char * data, uint64_t length) {
    std::array<uint8_t, 16> header{};
    std::memcpy(header.data(), RESPONSE_MAGIC.data(), RESPONSE_MAGIC.size());
    encode_u32(header.data() + 4, status);
    encode_u64(header.data() + 8, length);
    return write_exact(header.data(), header.size()) &&
           (length == 0 || write_exact(data, static_cast<size_t>(length)));
}

uint32_t synthesize(
        const common_params & params,
        llama_model * model,
        mtmd_context * mctx,
        const std::string & prompt,
        std::vector<char> * output) {
    // Pocket's backbone has recurrent state whose device buffers must start at
    // zero.  A fresh context provides that guarantee without reloading either
    // model.  Reusing a context with metadata-only clearing leaks recurrent
    // state; zeroing the full device cache is slower than context recreation
    // on the qualified NVK tuple.
    llama_context_params context_params = common_context_params_to_llama(params);
    llama_context_ptr context(llama_init_from_model(model, context_params));
    if (!context) {
        return 19;
    }
    common_params_sampling sampling = params.sampling;
    common_sampler_ptr sampler(common_sampler_init(model, sampling));
    if (!sampler) {
        return 19;
    }
    mtmd_helper::gen_audio generator(context.get(), mctx);

    // The current mtmd ABI reseeds its persistent projector RNG only when the
    // numeric seed changes.  Prime it once with a guaranteed-different seed;
    // the first real generation step then restores the admitted seed.  The
    // priming graph is stateless apart from RNG/output scratch and avoids
    // cross-utterance stochastic drift without reloading the projector.
    std::vector<float> zero_hidden(static_cast<size_t>(llama_model_n_embd(model)), 0.0f);
    mtmd_gen_inp reseed = mtmd_gen_inp_default(mctx);
    reseed.type = MTMD_GEN_PROCESS_TYPE_GEN_CODE;
    reseed.embd = zero_hidden.data();
    reseed.seed = params.sampling.seed ^ 0xa5a5a5a5U;
    mtmd_gen_out ignored{};
    if (mtmd_gen_audio_process(mctx, &reseed, &ignored) != 0) {
        return 18;
    }

    mtmd_helper_gen_audio_inp input{};
    input.seq_id = 0;
    input.prompt = prompt.data();
    input.prompt_len = prompt.size();
    input.speaker_ref = nullptr;
    input.lang = params.tts_lang.c_str();
    input.top_k = params.sampling.top_k;
    input.top_p = params.sampling.top_p;
    input.seed = params.sampling.seed;
    input.out_type = MTMD_HELPER_GEN_AUDIO_OUTTYPE_WAV;
    if (generator.set_input(&input) != 0) {
        return 20;
    }

    for (;;) {
        const int32_t remaining = generator.step_prompt(params.n_batch);
        if (remaining < 0) {
            return 21;
        }
        if (remaining == 0) {
            break;
        }
    }

    auto sample_semantic_code = [&]() -> llama_token {
        const llama_token token = common_sampler_sample(sampler.get(), context.get(), -1);
        common_sampler_accept(sampler.get(), token, true);
        return token;
    };

    const int maximum_frames = params.n_predict > 0 ? params.n_predict : 512;
    int frames = 0;
    llama_token sampled = sample_semantic_code();
    const float * hidden_state = llama_get_embeddings_ith(context.get(), -1);
    if (hidden_state == nullptr) {
        return 22;
    }
    bool stop = false;
    while (!stop && frames < maximum_frames) {
        const float * next_state = nullptr;
        if (generator.step_gen(sampled, hidden_state, &next_state, &stop) != 0) {
            return 23;
        }
        if (next_state == nullptr) {
            break;
        }
        ++frames;
        hidden_state = next_state;
        sampled = sample_semantic_code();
    }

    int32_t sample_rate = 0;
    const char * data = nullptr;
    size_t data_length = 0;
    int64_t samples = 0;
    if (generator.get_output(&sample_rate, &data, &data_length, &samples) != 0) {
        return 24;
    }
    if (sample_rate != REQUIRED_SAMPLE_RATE || samples <= 0 ||
        samples > MAX_AUDIO_SAMPLES || data == nullptr || data_length <= 44 ||
        data_length > MAX_WAV_BYTES) {
        return 25;
    }
    output->assign(data, data + data_length);
    return 0;
}

void print_usage(int, char ** argv) {
    LOG("\nusage: %s --model MODEL --mmproj PROJECTOR [fixed runtime options]\n", argv[0]);
}

} // namespace

int main(int argc, char ** argv) {
    common_params params;
    common_init();
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_TTS, print_usage)) {
        return 2;
    }
    mtmd_helper_log_set(common_log_default_callback, nullptr);
    if (!params.prompt.empty() || params.mmproj.path.empty() ||
        !params.tts_speaker_file.empty()) {
        return 2;
    }

    params.embedding = true;
    llama_backend_init();
    llama_numa_init(params.numa);
    auto llama_init = common_init_from_params(params, true);
    if (!llama_init) {
        llama_backend_free();
        return 3;
    }
    llama_model * model = llama_init->model();
    if (model == nullptr) {
        llama_backend_free();
        return 3;
    }

    mtmd_context_params mtmd_params = mtmd_context_params_default();
    mtmd_params.use_gpu = params.mmproj_use_gpu;
    mtmd::context_ptr mctx(mtmd_init_from_file(params.mmproj.path.c_str(), model, mtmd_params));
    if (!mctx || mtmd_gen_audio_get_info(mctx.get()).type != MTMD_GEN_AUDIO_TYPE_POCKETTTS) {
        llama_backend_free();
        return 4;
    }
    if (!write_exact(READY.data(), READY.size())) {
        llama_backend_free();
        return 5;
    }

    for (;;) {
        std::array<uint8_t, 8> request{};
        const read_result result = read_exact(request.data(), request.size(), true);
        if (result == read_result::eof) {
            break;
        }
        if (result != read_result::ok ||
            std::memcmp(request.data(), REQUEST_MAGIC.data(), REQUEST_MAGIC.size()) != 0) {
            llama_backend_free();
            return 6;
        }
        const uint32_t length = decode_u32(request.data() + 4);
        if (length == 0) {
            break;
        }
        if (length > MAX_TEXT_BYTES) {
            send_response(10, nullptr, 0);
            llama_backend_free();
            return 7;
        }
        std::string prompt(length, '\0');
        if (read_exact(prompt.data(), prompt.size(), false) != read_result::ok ||
            !valid_utf8(prompt)) {
            send_response(11, nullptr, 0);
            llama_backend_free();
            return 7;
        }

        std::vector<char> output;
        const uint32_t status = synthesize(
            params,
            model,
            mctx.get(),
            prompt,
            &output);
        if (!send_response(
                status,
                output.empty() ? nullptr : output.data(),
                status == 0 ? static_cast<uint64_t>(output.size()) : 0)) {
            llama_backend_free();
            return 8;
        }
        if (status != 0) {
            llama_backend_free();
            return 9;
        }
    }

    llama_backend_free();
    return 0;
}
