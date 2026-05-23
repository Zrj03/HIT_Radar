#ifndef CRH_2023_DECODER_HPP_
#define CRH_2023_DECODER_HPP_

#include <utils/data.h>

#include <rclcpp/logging.hpp>
#include <rclcpp/rclcpp.hpp>
#include <toml.hpp>
#include <vector>

class NetDecoderBase {
   protected:
    int INPUT_W, INPUT_H, NUM_CLASSES, NUM_COLORS;
    float BBOX_CONF_THRESH;
    rclcpp::Logger logger;
    NetDecoderBase(toml::value &config, const rclcpp::Logger &_logger);

   public:
    virtual void decode(int layer_index, const float *prob, std::vector<Armor> &objects) = 0;
    virtual void set_layer_info(int layer_index, const std::vector<size_t> &dimensions) = 0;
    virtual bool check_num_outputs(int num_outputs) = 0;
};

class YOLOv5Decoder : public NetDecoderBase {
   protected:
    struct YOLOv5LayerInfo {
        int index, num_anchors, out_h, out_w, num_outputs, stride;
    };
    std::vector<std::vector<float>> anchors;
    std::vector<YOLOv5LayerInfo> layers;

   public:
    YOLOv5Decoder(toml::value &, const rclcpp::Logger &);
    virtual void decode(int layer_index, const float *prob, std::vector<Armor> &objects) override;
    virtual void set_layer_info(int, const std::vector<size_t> &) override;
    virtual bool check_num_outputs(int num_outputs) override;
};

class YOLOv5_1_Decoder : public YOLOv5Decoder {
   protected:
    int NUM_TSIZES;

   public:
    YOLOv5_1_Decoder(toml::value &, const rclcpp::Logger &);
    virtual void decode(int layer_index, const float *prob, std::vector<Armor> &objects) override;
    virtual bool check_num_outputs(int num_outputs) override;
};

class YOLOv5FlatDecoder : public NetDecoderBase {
   protected:
    struct YOLOv5FlatLayerInfo {
        int index, num_preds, num_outputs;
    };
    std::vector<YOLOv5FlatLayerInfo> layers;
    std::vector<int> class_color_map;
    std::vector<int> class_type_map;
    size_t decode_calls = 0;
    size_t debug_log_period = 30;

   public:
    YOLOv5FlatDecoder(toml::value &, const rclcpp::Logger &);
    virtual void decode(int layer_index, const float *prob, std::vector<Armor> &objects) override;
    virtual void set_layer_info(int, const std::vector<size_t> &) override;
    virtual bool check_num_outputs(int num_outputs) override;
};

class YOLOv8Decoder : public NetDecoderBase {
   protected:
    int NUM_KPTS, NUM_TSIZES;
    bool standard_detect = false;
    std::vector<int> class_color_map;
    std::vector<int> class_type_map;
    struct YOLOv8LayerInfo {
        int index, num_outputs, num_preds, stride;
        bool channels_first;
    };
    std::vector<YOLOv8LayerInfo> layers;

   public:
    YOLOv8Decoder(toml::value &, const rclcpp::Logger &);
    virtual void decode(int layer_index, const float *prob, std::vector<Armor> &objects) override;
    virtual void set_layer_info(int, const std::vector<size_t> &) override;
    virtual bool check_num_outputs(int num_outputs) override;
};

// DETR-based armor detector decoder
// Output format: [batch, num_queries, 4+NUM_CLASSES+NUM_COLORS]
//   [0:4]              = cx, cy, w, h (normalized 0-1, already sigmoid)
//   [4:4+NUM_CLASSES]  = class scores (already sigmoid)
//   [4+NUM_CLASSES:16] = color scores (already sigmoid)
class DETRDecoder : public NetDecoderBase {
   protected:
    float min_class_score = 0.0f;
    float min_color_score = 0.0f;
    float class_margin = 0.0f;
    float color_margin = 0.0f;
    struct DETRLayerInfo {
        int index;
        int num_queries;
        int num_outputs;
    };
    std::vector<DETRLayerInfo> layers;

   public:
    DETRDecoder(toml::value &, const rclcpp::Logger &);
    virtual void decode(int layer_index, const float *prob, std::vector<Armor> &objects) override;
    virtual void set_layer_info(int, const std::vector<size_t> &) override;
    virtual bool check_num_outputs(int num_outputs) override;
};

inline float sigmoid(float x) { return (1.0 / (1.0 + exp(-x))); }

/**
 * @brief calculate sigmoid values in an array
 *
 * @param src pointer of source array
 * @param dst pointer of destination array
 * @param length number of values
 */
inline void sigmoid(const float *src, float *dst, int length) {
    for (int i = 0; i < length; ++i) {
        dst[i] = (1.0 / (1.0 + exp(-src[i])));
    }
}

#endif  // CRH_2023_DECODER_HPP_
