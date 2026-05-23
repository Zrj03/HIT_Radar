#ifndef CRH_2023_DETECTOR_TRT_HPP_
#define CRH_2023_DETECTOR_TRT_HPP_

#include <NvInfer.h>
#include <NvInferRuntime.h>
#include <cuda_fp16.h>
#include <detector/detector.h>
#include <cuda.h>

class DetectorTRT : public NetDetector {
   private:
    std::shared_ptr<nvinfer1::IRuntime> runtime;
    std::shared_ptr<nvinfer1::ICudaEngine> eg;
    std::shared_ptr<nvinfer1::IExecutionContext> context;
    // cudaStream_t stream;

    class TrtLogger : public nvinfer1::ILogger {
       private:
        rclcpp::Logger logger;

       public:
        explicit TrtLogger(const rclcpp::Logger &logger_) : logger(logger_) {}
        void log(nvinfer1::ILogger::Severity severity, const char *msg) noexcept override;
    };
    TrtLogger trt_logger;
    std::vector<float> input;
    std::vector<uint8_t> input_raw_bytes;
    std::vector<std::vector<float>> outputs;
    std::vector<std::vector<uint8_t>> output_raw_bytes;
    std::vector<void *> buffers;
    void *input_buffer = nullptr;
    std::vector<void *> output_buffers;
    nvinfer1::DataType input_type = nvinfer1::DataType::kFLOAT;
    size_t input_elem_size = sizeof(float);
    std::vector<nvinfer1::DataType> output_types;
    std::vector<size_t> output_elem_sizes;

    //解决回调函数不在一个线程的问题
    CUcontext* cuda_ctx;
    std::vector<int> out_size;
    void build_engine(const std::string &model_onnx, const std::string &model_engine);
    static size_t tensor_dtype_size(nvinfer1::DataType dtype);
    void convert_input_to_tensor_dtype();
    void convert_output_to_float(int output_idx);

   public:
    explicit DetectorTRT(const std::string &config_file, const std::string &share_dir,
                         const rclcpp::Logger &_logger, CUcontext *ctx);
    ~DetectorTRT();
    std::vector<Armor> detect(cv::Mat) override;
};

#endif
