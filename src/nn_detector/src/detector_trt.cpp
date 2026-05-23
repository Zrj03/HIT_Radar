#ifdef TRT
#include <NvOnnxParser.h>
#include <cuda.h>
#include <detector/detector_trt.h>

#include <fstream>
#include <functional>
#include <ios>
#include <memory>
#include <sstream>
#include <filesystem>

namespace {

std::string resolve_model_path(const toml::value &config, const std::string &share_dir,
                               const std::string &key, const std::string &fallback_path) {
    std::string model_path = fallback_path;
    if (config.contains(key)) {
        model_path = config.at(key).as_string();
    }

    std::filesystem::path path(model_path);
    if (path.is_absolute()) {
        return path.string();
    }
    return (std::filesystem::path(share_dir) / path).string();
}

const char* dtype_name(nvinfer1::DataType dtype) {
    switch (dtype) {
        case nvinfer1::DataType::kFLOAT:
            return "FLOAT";
        case nvinfer1::DataType::kHALF:
            return "HALF";
        case nvinfer1::DataType::kINT8:
            return "INT8";
        case nvinfer1::DataType::kINT32:
            return "INT32";
        case nvinfer1::DataType::kBOOL:
            return "BOOL";
#if NV_TENSORRT_MAJOR >= 10
        case nvinfer1::DataType::kUINT8:
            return "UINT8";
        case nvinfer1::DataType::kFP8:
            return "FP8";
        case nvinfer1::DataType::kINT64:
            return "INT64";
#endif
        default:
            return "UNKNOWN";
    }
}

const char *tensor_mode_name(nvinfer1::TensorIOMode mode) {
    switch (mode) {
        case nvinfer1::TensorIOMode::kINPUT:
            return "INPUT";
        case nvinfer1::TensorIOMode::kOUTPUT:
            return "OUTPUT";
        default:
            return "UNKNOWN";
    }
}

std::string dims_to_string(const nvinfer1::Dims &dims) {
    std::ostringstream oss;
    oss << "[";
    for (int i = 0; i < dims.nbDims; ++i) {
        if (i > 0) {
            oss << ", ";
        }
        oss << dims.d[i];
    }
    oss << "]";
    return oss.str();
}

bool has_half_output(const nvinfer1::ICudaEngine &engine) {
    const int tensor_count = engine.getNbIOTensors();
    for (int i = 0; i < tensor_count; ++i) {
        const char *tname = engine.getIOTensorName(i);
        if (engine.getTensorIOMode(tname) == nvinfer1::TensorIOMode::kOUTPUT &&
            engine.getTensorDataType(tname) == nvinfer1::DataType::kHALF) {
            return true;
        }
    }
    return false;
}

void mirror_engine_to_source(const std::string &engine_path, const rclcpp::Logger &logger) {
    namespace fs = std::filesystem;

    const fs::path cache_path(engine_path);
    if (!cache_path.has_filename()) {
        return;
    }

    // Mirror install-space TensorRT caches back into the source tree so the engine is
    // easy to inspect and reuse during local debugging.
    const std::string marker = "/install/nn_detector/share/nn_detector/models/";
    const std::string cache_str = cache_path.string();
    const auto marker_pos = cache_str.find(marker);
    if (marker_pos == std::string::npos) {
        return;
    }

    const fs::path workspace_root = cache_str.substr(0, marker_pos);
    const fs::path src_models_dir = workspace_root / "src/nn_detector/models";
    const fs::path src_engine_path = src_models_dir / cache_path.filename();

    std::error_code ec;
    fs::create_directories(src_models_dir, ec);
    if (ec) {
        RCLCPP_WARN(logger, "[TRT] failed to create source models dir %s: %s",
                    src_models_dir.c_str(), ec.message().c_str());
        return;
    }

    if (fs::equivalent(cache_path, src_engine_path, ec)) {
        return;
    }
    ec.clear();

    fs::copy_file(cache_path, src_engine_path, fs::copy_options::overwrite_existing, ec);
    if (ec) {
        RCLCPP_WARN(logger, "[TRT] failed to mirror engine to %s: %s", src_engine_path.c_str(),
                    ec.message().c_str());
        return;
    }

    RCLCPP_INFO(logger, "[TRT] mirrored engine to source tree: %s", src_engine_path.c_str());
}

}  // namespace

#ifndef CUDA_CHECK
#define CUDA_CHECK(callstr)                                                                   \
    {                                                                                         \
        cudaError_t error_code = callstr;                                                     \
        if (error_code != cudaSuccess) {                                                      \
            std::cerr << "CUDA error " << error_code << " at " << __FILE__ << ":" << __LINE__ \
                      << std::endl;                                                           \
            exit(0);                                                                          \
        }                                                                                     \
    }
#endif

void DetectorTRT::TrtLogger::log(nvinfer1::ILogger::Severity severity, const char *msg) noexcept {
    switch (severity) {
        case Severity::kINTERNAL_ERROR:
        case Severity::kERROR:
            RCLCPP_ERROR(logger, msg);
            break;
        case Severity::kWARNING:
            RCLCPP_WARN(logger, msg);
            break;
        case Severity::kINFO:
            RCLCPP_INFO(logger, msg);
            break;
        case Severity::kVERBOSE:
            RCLCPP_DEBUG(logger, msg);
            break;
    }
}

size_t DetectorTRT::tensor_dtype_size(nvinfer1::DataType dtype) {
    switch (dtype) {
        case nvinfer1::DataType::kFLOAT:
            return sizeof(float);
        case nvinfer1::DataType::kHALF:
            return sizeof(__half);
        case nvinfer1::DataType::kINT8:
            return sizeof(int8_t);
        case nvinfer1::DataType::kINT32:
            return sizeof(int32_t);
        case nvinfer1::DataType::kBOOL:
            return sizeof(bool);
#if NV_TENSORRT_MAJOR >= 10
        case nvinfer1::DataType::kUINT8:
            return sizeof(uint8_t);
        case nvinfer1::DataType::kINT64:
            return sizeof(int64_t);
#endif
        default:
            throw std::runtime_error("Unsupported TensorRT tensor dtype");
    }
}

void DetectorTRT::convert_input_to_tensor_dtype() {
    const size_t elem_count = input.size();
    input_raw_bytes.resize(elem_count * input_elem_size);
    auto *raw = input_raw_bytes.data();

    switch (input_type) {
        case nvinfer1::DataType::kFLOAT: {
            std::memcpy(raw, input.data(), elem_count * sizeof(float));
            break;
        }
        case nvinfer1::DataType::kHALF: {
            auto *dst = reinterpret_cast<__half *>(raw);
            for (size_t i = 0; i < elem_count; ++i) {
                dst[i] = __float2half(input[i]);
            }
            break;
        }
        default:
            throw std::runtime_error("Unsupported TensorRT input dtype");
    }
}

void DetectorTRT::convert_output_to_float(int output_idx) {
    outputs[output_idx].resize(out_size[output_idx]);
    const auto dtype = output_types[output_idx];
    const auto *raw = output_raw_bytes[output_idx].data();
    auto *dst = outputs[output_idx].data();

    switch (dtype) {
        case nvinfer1::DataType::kFLOAT: {
            std::memcpy(dst, raw, out_size[output_idx] * sizeof(float));
            break;
        }
        case nvinfer1::DataType::kHALF: {
            const auto *src = reinterpret_cast<const __half *>(raw);
            for (int j = 0; j < out_size[output_idx]; ++j) {
                dst[j] = __half2float(src[j]);
            }
            break;
        }
        case nvinfer1::DataType::kINT8: {
            const auto *src = reinterpret_cast<const int8_t *>(raw);
            for (int j = 0; j < out_size[output_idx]; ++j) {
                dst[j] = static_cast<float>(src[j]);
            }
            break;
        }
        case nvinfer1::DataType::kINT32: {
            const auto *src = reinterpret_cast<const int32_t *>(raw);
            for (int j = 0; j < out_size[output_idx]; ++j) {
                dst[j] = static_cast<float>(src[j]);
            }
            break;
        }
        case nvinfer1::DataType::kBOOL: {
            const auto *src = reinterpret_cast<const bool *>(raw);
            for (int j = 0; j < out_size[output_idx]; ++j) {
                dst[j] = src[j] ? 1.0f : 0.0f;
            }
            break;
        }
        default:
            throw std::runtime_error("Unsupported TensorRT tensor dtype during output conversion");
    }
}

DetectorTRT::DetectorTRT(const std::string &config_file, const std::string &share_dir,
                         const rclcpp::Logger &_logger, CUcontext *ctx)
    : NetDetector(config_file, share_dir, _logger), trt_logger(_logger), cuda_ctx(ctx) {
    std::string model_onnx = resolve_model_path(config, share_dir, "model_onnx",
                                                std::string(config.at("model_prefix").as_string()) + ".onnx");
    std::string model_cache = resolve_model_path(config, share_dir, "model_trt",
                                                 std::string(config.at("model_prefix").as_string()) + ".engine");
    bool use_fp16 = true;
    if (config.contains("use_fp16")) {
        use_fp16 = config.at("use_fp16").as_boolean();
    }

    RCLCPP_INFO(logger, "[TRT] loading engine/cache: %s", model_cache.c_str());
    runtime = std::shared_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(trt_logger));

    std::stringstream egsr;
    egsr.seekg(0, std::ios::beg);
    std::ifstream egf(model_cache);
    if (!egf.is_open()) {
        build_engine(model_onnx, model_cache);
    } else {
        egsr << egf.rdbuf();

        egsr.seekg(0, std::ios::end);
        const int size = egsr.tellg();
        egsr.seekg(0, std::ios::beg);

        char *mem = (char *)malloc(size);
        egsr.read(mem, size);
        eg = std::shared_ptr<nvinfer1::ICudaEngine>(runtime->deserializeCudaEngine(mem, size));
        if (eg != nullptr && !use_fp16 && has_half_output(*eg)) {
            RCLCPP_WARN(
                logger,
                "[TRT] cached engine has HALF outputs but use_fp16=false; rebuilding cache as FP32");
            eg.reset();
            std::error_code ec;
            std::filesystem::remove(model_cache, ec);
            if (ec) {
                RCLCPP_WARN(logger, "[TRT] failed to remove stale cache %s: %s",
                            model_cache.c_str(), ec.message().c_str());
            }
        }
        if (eg == nullptr) build_engine(model_onnx, model_cache);
        free(mem);
        egf.close();
    }

    if (!use_fp16 && has_half_output(*eg)) {
        RCLCPP_WARN(
            logger,
            "[TRT] engine still exposes HALF outputs after FP32 rebuild; ONNX may contain FP16 tensors");
    }

    mirror_engine_to_source(model_cache, logger);

    context = std::shared_ptr<nvinfer1::IExecutionContext>(eg->createExecutionContext());

    // input = new float[INPUT_H * INPUT_W * 3];
    input.resize(INPUT_H * INPUT_W * 3);

    int binding_num = eg->getNbIOTensors();
    // outputs = new float *[binding_num];
    outputs.resize(binding_num);
    output_raw_bytes.resize(binding_num);
    output_types.resize(binding_num);
    output_elem_sizes.resize(binding_num);
    out_size.resize(binding_num);
    layer_num = 0;

    RCLCPP_INFO(logger, "[TRT] engine I/O tensors=%d", binding_num);
    for (int i = 0; i < binding_num; ++i) {
        const char *tname = eg->getIOTensorName(i);
        const auto mode = eg->getTensorIOMode(tname);
        const auto dtype = eg->getTensorDataType(tname);
        const auto dims = eg->getTensorShape(tname);
        RCLCPP_INFO(logger, "[TRT] io[%d] name=%s mode=%s dtype=%s dims=%s", i, tname,
                    tensor_mode_name(mode), dtype_name(dtype), dims_to_string(dims).c_str());
        if (mode == nvinfer1::TensorIOMode::kINPUT) {
            input_type = dtype;
            input_elem_size = tensor_dtype_size(dtype);
            if (dims.nbDims != 4 || dims.d[0] != 1 || dims.d[1] != 3 || dims.d[2] != INPUT_H ||
                dims.d[3] != INPUT_W) {
                RCLCPP_WARN(logger,
                            "[TRT] input tensor %s dims=%s differs from configured [1, 3, %d, %d]",
                            tname, dims_to_string(dims).c_str(), INPUT_H, INPUT_W);
            }
        } else if (mode == nvinfer1::TensorIOMode::kOUTPUT) {
            // assert(dims.nbDims == 5);
            // out_size[layer_num] = dims.d[0] * dims.d[1] * dims.d[2] * dims.d[3] * dims.d[4];
            out_size[layer_num] = 1;
            for (int j = 0; j < dims.nbDims; ++j) {
                out_size[layer_num] *= dims.d[j];
            }
            output_types[layer_num] = dtype;
            output_elem_sizes[layer_num] = tensor_dtype_size(dtype);
            outputs[layer_num].resize(out_size[layer_num]);
            output_raw_bytes[layer_num].resize(out_size[layer_num] * output_elem_sizes[layer_num]);
            std::vector<size_t> standard_dims(dims.nbDims);
            std::transform(
                dims.d, dims.d + dims.nbDims, standard_dims.begin(),
                [](decltype(*dims.d) x) -> size_t {  // should be replaced by std::identity of C++20
                    return x;
                });

            decoder->set_layer_info(layer_num, standard_dims);
            RCLCPP_INFO(
                logger,
                "[TRT] output[%d] tensor=%s dtype=%s dims=%s elems=%d bytes=%zu",
                layer_num, tname, dtype_name(dtype), dims_to_string(dims).c_str(),
                out_size[layer_num],
                out_size[layer_num] * output_elem_sizes[layer_num]);
            ++layer_num;
        }
    }
    assert(layer_num + 1 == binding_num && "layer_num must equals binging_num-1");
    buffers.resize(binding_num, nullptr);
    input_raw_bytes.resize(INPUT_H * INPUT_W * 3 * input_elem_size);
    cudaMalloc(&input_buffer, input_raw_bytes.size());
    output_buffers.resize(layer_num, nullptr);
    for (int i = 0; i < layer_num; ++i) {
        cudaMalloc(&output_buffers[i], out_size[i] * output_elem_sizes[i]);
    }

    int output_idx = 0;
    for (int i = 0; i < binding_num; ++i) {
        const char *tname = eg->getIOTensorName(i);
        void *buffer = nullptr;
        if (eg->getTensorIOMode(tname) == nvinfer1::TensorIOMode::kOUTPUT) {
            buffer = output_buffers[output_idx++];
        } else {
            buffer = input_buffer;
        }
        buffers[i] = buffer;
        context->setTensorAddress(tname, buffer);
    }

    // cudaStreamCreate(&stream);

    RCLCPP_INFO(logger, "[TRT] engine loaded");
}

DetectorTRT::~DetectorTRT() {
    if (input_buffer) {
        cudaFree(input_buffer);
    }
    for (auto *buffer : output_buffers) {
        if (buffer) {
            cudaFree(buffer);
        }
    }
}

void DetectorTRT::build_engine(const std::string &model_onnx, const std::string &model_cache) {
    RCLCPP_INFO(logger, "[TRT] fail to load engine; generating one from onnx file...");

    std::shared_ptr<nvinfer1::IBuilder> builder{nvinfer1::createInferBuilder(trt_logger)};
    std::shared_ptr<nvinfer1::IBuilderConfig> config{builder->createBuilderConfig()};

    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 1 << 30);
    bool use_fp16 = true;
    if (this->config.contains("use_fp16")) {
        use_fp16 = this->config.at("use_fp16").as_boolean();
    }
    if (use_fp16) {
        config->setFlag(nvinfer1::BuilderFlag::kFP16);
        RCLCPP_INFO(logger, "[TRT] builder precision: FP16");
    } else {
        RCLCPP_INFO(logger, "[TRT] builder precision: FP32");
    }

    auto flag =
        1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    std::shared_ptr<nvinfer1::INetworkDefinition> network{builder->createNetworkV2(flag)};
    std::shared_ptr<nvonnxparser::IParser> parser{nvonnxparser::createParser(*network, trt_logger)};

    bool res = parser->parseFromFile(model_onnx.c_str(),
                                     static_cast<int>(nvinfer1::ILogger::Severity::kWARNING));
    assert(res && "parse onnx file failed");
    RCLCPP_INFO(logger, "[TRT] parsed network outputs=%d", network->getNbOutputs());
    for (int i = 0; i < network->getNbOutputs(); ++i) {
        const auto *output = network->getOutput(i);
        RCLCPP_INFO(logger, "[TRT] network output[%d] name=%s dtype=%s dims=%s", i,
                    output->getName(), dtype_name(output->getType()),
                    dims_to_string(output->getDimensions()).c_str());
    }

    std::shared_ptr<nvinfer1::IHostMemory> plan{builder->buildSerializedNetwork(*network, *config)};
    std::ofstream egf_o;
    egf_o.open(model_cache, std::ios::binary | std::ios::out);
    egf_o.write((const char *)plan->data(), plan->size());
    egf_o.close();
    eg = std::shared_ptr<nvinfer1::ICudaEngine>(
        runtime->deserializeCudaEngine(plan->data(), plan->size()));
}

std::vector<Armor> DetectorTRT::detect(cv::Mat img) {
    // PerfGuard perf_infer("Detector::Infer");

    float r = -1.;  // not resized
    if (img.cols != INPUT_W || img.rows != INPUT_H) {
        RCLCPP_WARN(logger, "[TRT INFER] resizing %dx%d to %dx%d", img.cols, img.rows, INPUT_W,
                    INPUT_H);
        r = std::min(INPUT_W / (img.cols * 1.0), INPUT_H / (img.rows * 1.0));
        img = static_resize(img, INPUT_H, INPUT_W);
    }

    assert(img.type() == CV_8UC3);
    assert(img.cols == INPUT_W && img.rows == INPUT_H);

    // RCLCPP_INFO(logger,"[TRT INFER] 1");

    img2blob(img, input);
    convert_input_to_tensor_dtype();
    // 解决多线程infer问题
    cuCtxPushCurrent(*cuda_ctx);

    CUDA_CHECK(cudaMemcpy(input_buffer, input_raw_bytes.data(), input_raw_bytes.size(),
                          cudaMemcpyHostToDevice));

    // bool result = context->enqueueV2(buffers, stream, nullptr);

    bool result = context->executeV2(&buffers[0]);
    if (!result) {
        RCLCPP_ERROR(logger, "[TRT INFER] forward failed");
        // logger.error();
        cuCtxPopCurrent(cuda_ctx);
        return {};
    }

    for (int i = 0; i < layer_num; ++i) {
        CUDA_CHECK(cudaMemcpy(output_raw_bytes[i].data(), output_buffers[i],
                              out_size[i] * output_elem_sizes[i],
                              cudaMemcpyDeviceToHost));
    }

    // 解决多线程infer问题
    cuCtxPopCurrent(cuda_ctx);

    for (int i = 0; i < layer_num; ++i) {
        convert_output_to_float(i);
    }

    std::vector<Armor> res;
    for (int i = 0; i < layer_num; ++i) {
        decoder->decode(i, outputs[i].data(), res);
    }

    // remove items out of the bounds
    auto last = std::remove_if(
        res.begin(), res.end(), [n = point_num, H = INPUT_H, W = INPUT_W](auto item) {
            return std::any_of(item.pts, item.pts + n,
                               [=](auto p) { return p.x < 0 || p.x > W || p.y < 0 || p.y > H; });
        });
    res.erase(last, res.end());

    res = do_merge_nms(res);

    if (r > 0) {
        for (auto &a : res) {
            a.rect.x /= r;
            a.rect.y /= r;
            a.rect.width /= r;
            a.rect.height /= r;
            for (auto &p : a.pts) {
                p.x /= r;
                p.y /= r;
            }
        }
    }

    return res;
}

#endif
