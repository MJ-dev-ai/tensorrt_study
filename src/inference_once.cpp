#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdio>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include "cv/cv.hpp"

#include <cuda_runtime.h>
#include "NvInfer.h"
#include "NvOnnxParser.h"

using namespace nvinfer1;

class Logger : public ILogger {
public:
    explicit Logger(const char* file_path) : log_file_(file_path, std::ios::app) {
        if (!log_file_) {
            throw std::runtime_error("Failed to open log file: " + std::string(file_path));
        }
    }

    void log(Severity severity, const char* msg) noexcept override {
        if (severity > Severity::kINFO || !msg) return;
        const char* level;
        switch (severity) {
            case Severity::kINTERNAL_ERROR: level = "INTERNAL_ERROR"; break;
            case Severity::kERROR: level = "ERROR"; break;
            case Severity::kWARNING: level = "WARNING"; break;
            case Severity::kINFO: level = "INFO"; break;
            case Severity::kVERBOSE: level = "VERBOSE"; break;
        }
        try {
            std::lock_guard<std::mutex> lock(mutex_);
            const std::time_t NOW = std::time(nullptr);
            std::tm local_time{};
            if (localtime_s(&local_time, &NOW) != 0) return;

            char time_buffer[16]{};
            if (std::strftime(time_buffer, sizeof(time_buffer), "[%H-%M-%S]", &local_time) == 0) return;

            log_file_ << time_buffer << " [" << level << "] " << msg << '\n';
            log_file_.flush();

            if (!log_file_) std::fputs("Failed to write log file.\n", stderr);
        } catch (...) {
            std::fputs("Logging failed.\n", stderr);
        }
    }

private:
    std::ofstream log_file_;
    std::mutex mutex_;
};

// ONNX를 엔진으로 빌드하고 파일과 벡터로 저장합니다.
std::vector<char> buildEngine(const char* onnx_path, const char* engine_path, Logger& logger, int batch_size) {
    std::unique_ptr<IBuilder> builder{createInferBuilder(logger)};
    if (!builder) throw std::runtime_error("Failed to create builder");

    const uint32_t FLAG = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kSTRONGLY_TYPED);
    std::unique_ptr<INetworkDefinition> network{builder->createNetworkV2(FLAG)};
    if (!network) throw std::runtime_error("Failed to create network");

    std::unique_ptr<nvonnxparser::IParser> parser{nvonnxparser::createParser(*network, logger)};
    if (!parser) throw std::runtime_error("Failed to create ONNX parser");

    if (!parser->parseFromFile(onnx_path, static_cast<int32_t>(ILogger::Severity::kWARNING))) {
        for (int32_t i = 0; i < parser->getNbErrors(); ++i) {
            std::cerr << parser->getError(i)->desc() << '\n';
        }
        throw std::runtime_error("Failed to parse ONNX file");
    }

    std::unique_ptr<IBuilderConfig> config{builder->createBuilderConfig()};
    if (!config) throw std::runtime_error("Failed to create builder config");
    config->setMemoryPoolLimit(MemoryPoolType::kWORKSPACE, 4ULL << 30);

    // 고정 ONNX는 같은 배치가 필요하고, 동적 ONNX는 선택한 배치로 최적화합니다.
    ITensor* input = network->getInput(0);
    Dims dims = input->getDimensions();
    if (dims.d[0] == -1) {
        dims.d[0] = batch_size;
        dims.d[2] = 2048;
        dims.d[3] = 2048;
        IOptimizationProfile* profile = builder->createOptimizationProfile();
        if (!profile || !profile->setDimensions(input->getName(), OptProfileSelector::kMIN, dims) ||
            !profile->setDimensions(input->getName(), OptProfileSelector::kOPT, dims) ||
            !profile->setDimensions(input->getName(), OptProfileSelector::kMAX, dims) ||
            config->addOptimizationProfile(profile) < 0) {
            throw std::runtime_error("Failed to configure batch profile");
        }
    } else if (dims.d[0] != batch_size) {
        throw std::runtime_error("ONNX batch does not match BATCH_SIZE. Export matching or dynamic ONNX first.");
    }

    std::unique_ptr<IHostMemory> serialized{builder->buildSerializedNetwork(*network, *config)};
    if (!serialized) throw std::runtime_error("Engine build failed");

    const char* data = static_cast<const char*>(serialized->data());
    std::ofstream file(engine_path, std::ios::binary);
    if (!file) throw std::runtime_error("Failed to open engine output file");

    file.write(data, static_cast<std::streamsize>(serialized->size()));
    file.close();
    if (!file) throw std::runtime_error("Failed to save engine file");

    return std::vector<char>(data, data + serialized->size());
}

// RGB 바이트를 정규화된 CHW 이미지로 변환한 뒤 cv::resize로 크기를 맞춥니다.
cv::CvImage preprocessImage(const char* image_path, const Dims& input_dims) {
    if (input_dims.nbDims != 4 || input_dims.d[0] != 1 || input_dims.d[1] != 3 ||
        input_dims.d[2] <= 0 || input_dims.d[3] <= 0 ||
        input_dims.d[2] > INT_MAX || input_dims.d[3] > INT_MAX) {
        throw std::runtime_error("Expected input shape [1, 3, H, W]");
    }

    const int HEIGHT = static_cast<int>(input_dims.d[2]);
    const int WIDTH = static_cast<int>(input_dims.d[3]);
    int width = 0, height = 0, channels = 0;

    std::unique_ptr<unsigned char, decltype(&stbi_image_free)> image(
        stbi_load(image_path, &width, &height, &channels, 3), stbi_image_free);
    if (!image) throw std::runtime_error("Failed to load image: " + std::string(image_path));

    const size_t PIXEL_COUNT = static_cast<size_t>(width) * height;
    cv::CvImage source{std::vector<float>(PIXEL_COUNT * 3), width, height};
    for (size_t i = 0; i < PIXEL_COUNT; ++i) {
        source.data[i] = image.get()[i * 3] / 255.0F;
        source.data[PIXEL_COUNT + i] = image.get()[i * 3 + 1] / 255.0F;
        source.data[2 * PIXEL_COUNT + i] = image.get()[i * 3 + 2] / 255.0F;
    }

    cv::CvImage resized;
    cv::resize(source, resized, {WIDTH, HEIGHT}, cv::Interpolation::Bicubic);
    return resized;
}

// 현재 YOLO 모델: FP32 입력 [N, 3, 2048, 2048], 출력 [N, 8, 86016].
// 성공 0, 실패 1. 실패 시에도 CUDA 자원을 정리합니다.
int runInference(const std::vector<char>& engine_data, const std::vector<float>& host_input,
                 const Dims& input_dims, std::vector<float>& host_output, Logger& logger) {
    host_output.clear();

    try {
        std::unique_ptr<IRuntime> runtime{createInferRuntime(logger)};
        if (!runtime) throw std::runtime_error("Failed to create runtime");

        std::unique_ptr<ICudaEngine> engine{
            runtime->deserializeCudaEngine(engine_data.data(), engine_data.size())};
        if (!engine) throw std::runtime_error("Failed to load engine");

        if (engine->getNbIOTensors() != 2) {
            throw std::runtime_error("Expected one input and one output tensor");
        }

        const char* input_name = nullptr;
        const char* output_name = nullptr;

        for (int32_t i = 0; i < engine->getNbIOTensors(); ++i) {
            const char* name = engine->getIOTensorName(i);
            if (engine->getTensorDataType(name) != DataType::kFLOAT ||
                engine->getTensorFormat(name) != TensorFormat::kLINEAR ||
                engine->getTensorLocation(name) != TensorLocation::kDEVICE) {
                throw std::runtime_error("Expected linear FP32 device tensors");
            }

            if (engine->getTensorIOMode(name) == TensorIOMode::kINPUT) input_name = name;
            else output_name = name;
        }

        if (!input_name || !output_name) throw std::runtime_error("Missing input/output tensor");

        std::unique_ptr<IExecutionContext> context{engine->createExecutionContext()};
        if (!context) throw std::runtime_error("Failed to create execution context");

        const size_t BATCH_SIZE = static_cast<size_t>(input_dims.d[0]);
        const size_t INPUT_COUNT = BATCH_SIZE * 3 * 2048 * 2048;
        if (host_input.size() != INPUT_COUNT) throw std::runtime_error("Input vector size mismatch");
        if (!context->setInputShape(input_name, input_dims)) {
            throw std::runtime_error("Engine input shape does not match BATCH_SIZE / image size");
        }

        const Dims OUTPUT_DIMS = context->getTensorShape(output_name);
        if (OUTPUT_DIMS.nbDims != 3 || OUTPUT_DIMS.d[0] != input_dims.d[0] ||
            OUTPUT_DIMS.d[1] != 8 || OUTPUT_DIMS.d[2] != 86016) {
            throw std::runtime_error("Expected output shape [N, 8, 86016]");
        }
        host_output.resize(BATCH_SIZE * 8 * 86016);

        // 함수 종료나 예외 발생 시 자동으로 정리합니다.
        struct CudaResources {
            void* input = nullptr;
            void* output = nullptr;
            cudaStream_t stream = nullptr;

            ~CudaResources() {
                if (stream) cudaStreamSynchronize(stream);
                if (input) cudaFree(input);
                if (output) cudaFree(output);
                if (stream) cudaStreamDestroy(stream);
            }
        } gpu;

        auto checkCuda = [](cudaError_t status) {
            if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
        };

        checkCuda(cudaStreamCreate(&gpu.stream));
        checkCuda(cudaMalloc(&gpu.input, host_input.size() * sizeof(float)));
        checkCuda(cudaMalloc(&gpu.output, host_output.size() * sizeof(float)));

        checkCuda(cudaMemcpyAsync(gpu.input, host_input.data(), host_input.size() * sizeof(float),
                                  cudaMemcpyHostToDevice, gpu.stream));

        if (!context->setTensorAddress(input_name, gpu.input) ||
            !context->setTensorAddress(output_name, gpu.output)) {
            throw std::runtime_error("Failed to set tensor addresses");
        }

        if (!context->enqueueV3(gpu.stream)) throw std::runtime_error("Failed to enqueue inference");

        checkCuda(cudaMemcpyAsync(host_output.data(), gpu.output, host_output.size() * sizeof(float),
                                  cudaMemcpyDeviceToHost, gpu.stream));
        checkCuda(cudaStreamSynchronize(gpu.stream));
        return 0;
    } catch (const std::exception& error) {
        host_output.clear();
        logger.log(ILogger::Severity::kERROR, error.what());
        std::cerr << "Inference error: " << error.what() << '\n';
        return 1;
    } catch (...) {
        host_output.clear();
        logger.log(ILogger::Severity::kERROR, "Unknown inference error");
        return 1;
    }
}

int main(int argc, char* argv[]) {
    const std::filesystem::path project_root{PROJECT_ROOT};
    std::string onnx_path;
    std::string engine_path;
    std::filesystem::path output_dir = project_root / "data" / "outputs";
    int batch_size = 4;
    std::vector<const char*> image_paths;

    try {
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            if (arg == "--help") {
                std::cout << "Usage: inference_once.exe [--onnx model.onnx] [--engine model.engine] [--batch N] [--output-dir DIR] [image paths...]\n";
                return 0;
            }
            if (arg == "--onnx" || arg == "--engine" || arg == "--batch" || arg == "--output-dir") {
                if (++i == argc) throw std::runtime_error("Missing value for " + arg);
                if (arg == "--onnx") onnx_path = argv[i];
                else if (arg == "--engine") engine_path = argv[i];
                else if (arg == "--output-dir") output_dir = argv[i];
                else {
                    size_t parsed = 0;
                    batch_size = std::stoi(argv[i], &parsed);
                    if (argv[i][parsed] != '\0' || batch_size <= 0)
                        throw std::runtime_error("Batch must be a positive integer");
                }
            } else if (arg.rfind("--", 0) == 0) {
                throw std::runtime_error("Unknown option: " + arg);
            } else {
                image_paths.push_back(argv[i]);
            }
        }
        if (!image_paths.empty() && image_paths.size() != 1 && image_paths.size() != static_cast<size_t>(batch_size))
            throw std::runtime_error("Provide one image to repeat, or one image per batch item");

        // Default model names follow the selected batch; explicit CLI paths stay unchanged.
        const std::string model_stem = "hyundai_emblem_b" + std::to_string(batch_size) + "_fp32";
        if (onnx_path.empty()) onnx_path = (project_root / "models" / (model_stem + ".onnx")).string();
        if (engine_path.empty()) engine_path = (project_root / "models" / (model_stem + ".engine")).string();

        // 현재 시간을 채운 다음 날짜 경로를 만듭니다.
        const std::time_t NOW = std::time(nullptr);
        std::tm local_time{};
        if (localtime_s(&local_time, &NOW) != 0) {
            throw std::runtime_error("Failed to get local time");
        }

        char log_file_path[64]{};
        if (std::strftime(log_file_path, sizeof(log_file_path), "%y-%m-%d/log.txt", &local_time) == 0) {
            throw std::runtime_error("Failed to format log path");
        }

        const auto log_path = project_root / "outputs" / "logs" / log_file_path;
        std::filesystem::create_directories(log_path.parent_path());
        Logger logger(log_path.string().c_str());

        const Dims INPUT_DIMS{4, {batch_size, 3, 2048, 2048}};
        const Dims IMAGE_DIMS{4, {1, 3, 2048, 2048}};
        const size_t IMAGE_COUNT = static_cast<size_t>(3) * 2048 * 2048;
        const size_t OUTPUT_COUNT = static_cast<size_t>(8) * 86016;

        std::vector<char> engine_data;

        if (std::filesystem::exists(engine_path)) {
            std::cout << "Loading existing engine..." << std::endl;
            std::ifstream engine_file(engine_path, std::ios::binary | std::ios::ate);
            if (!engine_file) throw std::runtime_error("Failed to open engine file");

            const std::streamoff ENGINE_SIZE = engine_file.tellg();
            if (ENGINE_SIZE <= 0) throw std::runtime_error("Engine file is empty or unreadable");

            engine_data.resize(static_cast<size_t>(ENGINE_SIZE));
            engine_file.seekg(0, std::ios::beg);

            if (!engine_file.read(engine_data.data(), static_cast<std::streamsize>(ENGINE_SIZE))) {
                throw std::runtime_error("Failed to read engine file");
            }
        } else {
            std::cout << "Building engine..." << std::endl;
            engine_data = buildEngine(onnx_path.c_str(), engine_path.c_str(), logger, batch_size);
            std::cout << "Engine build finished" << std::endl;
        }

        if (image_paths.empty()) {
            std::cout << "Engine ready: " << std::filesystem::absolute(engine_path) << std::endl;
            return 0;
        }

        std::vector<float> host_input;
        host_input.reserve(batch_size * IMAGE_COUNT);
        cv::CvImage image;
        for (int b = 0; b < batch_size; ++b) {
            const char* image_path = image_paths[image_paths.size() == 1 ? 0 : b];
            if (b == 0 || image_paths.size() != 1) image = preprocessImage(image_path, IMAGE_DIMS);
            host_input.insert(host_input.end(), image.data.begin(), image.data.end());
            std::cout << "Batch " << b << ": " << image_path << '\n';
        }
        std::cout << "Running inference..." << std::endl;
        std::vector<float> host_output;

        if (runInference(engine_data, host_input, INPUT_DIMS, host_output, logger) != 0) {
            return 1;
        }

        std::cout << "Output elements: " << host_output.size() << std::endl;
        const int WIDTH = static_cast<int>(IMAGE_DIMS.d[3]);
        const int HEIGHT = static_cast<int>(IMAGE_DIMS.d[2]);
        const size_t PIXEL_COUNT = static_cast<size_t>(WIDTH) * HEIGHT;
        std::vector<unsigned char> output_image(PIXEL_COUNT * 3);
        std::vector<float> output(OUTPUT_COUNT);

        std::filesystem::create_directories(output_dir);
        for (int b = 0; b < batch_size; ++b) {
            image.data.assign(host_input.begin() + b * IMAGE_COUNT, host_input.begin() + (b + 1) * IMAGE_COUNT);
            std::copy_n(host_output.begin() + b * OUTPUT_COUNT, OUTPUT_COUNT, output.begin());
            std::cout << "Batch " << b << " results:\n";
            cv::drawBbox(image, output, 4);
            for (size_t i = 0; i < PIXEL_COUNT; ++i) {
                for (size_t c = 0; c < 3; ++c) {
                    const float VALUE = std::clamp(image.data[c * PIXEL_COUNT + i], 0.0F, 1.0F);
                    output_image[i * 3 + c] = static_cast<unsigned char>(VALUE * 255.0F + 0.5F);
                }
            }
            const std::string OUTPUT_PATH = (output_dir / (batch_size == 1 ? "output.png" : "output_" + std::to_string(b) + ".png")).string();
            if (!stbi_write_png(OUTPUT_PATH.c_str(), WIDTH, HEIGHT, 3, output_image.data(), WIDTH * 3)) {
                throw std::runtime_error("Failed to save output image");
            }
            std::cout << "Saved: " << std::filesystem::absolute(OUTPUT_PATH) << std::endl;
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << std::endl;
        return 1;
    }
}