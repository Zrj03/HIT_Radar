#include <hik_camera/hik_camera.h>
// #include <rm_utils/common.h>
// #include <rm_utils/frame_info.h>
#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <iostream>
// #include <rm_utils/perf.hpp>
#include <string>
// 尝试func, 如果返回值不是MV_OK(即0)则调用logger记录WARN日志
#define UPDBW(func)                                                                        \
    nRet = func;                                                                           \
    if (nRet != MV_OK) {                                                                   \
        RCLCPP_WARN(this->get_logger(), #func " failed!, error code: %x", (unsigned)nRet); \
    }

// 尝试func, 如果返回值不是MV_OK(即0)则调用logger记录FATAL日志
#define UPDBF(func)                                                                         \
    nRet = func;                                                                            \
    if (nRet != MV_OK) {                                                                    \
        RCLCPP_FATAL(this->get_logger(), #func " failed!, error code: %x", (unsigned)nRet); \
    }

// 对于不可恢复性错误重启相机节点
#define UPDBE(func)      \
    UPDBF(func)          \
    if (nRet != MV_OK) { \
        reset();         \
    }

using namespace hik_camera;

HikCameraNode::HikCameraNode(const rclcpp::NodeOptions& options) : Node("hik_camera", options) {
    RCLCPP_INFO(this->get_logger(), "Starting HikCameraNode!");
    // 1) 声明全部可配置参数，后续初始化和热更新都依赖这些参数。
    declare_params();
    bool use_sensor_data_qos = get_parameter("use_sensor_data_qos").as_bool();
    std::string camera_info_url = get_parameter("camera_info_url").as_string();
    // 2) 创建图像发布器。这里使用 image_transport::CameraPublisher 同步发布 image + camera_info。
    bool use_intra = options.use_intra_process_comms();
    if (!use_intra) {
        RCLCPP_WARN(get_logger(), "Not In Intra Process Mode");
    }
    if (!use_sensor_data_qos) {
        RCLCPP_WARN(get_logger(), "Not Use Sensor Data Qos");
    }
    auto qos = use_sensor_data_qos ? rmw_qos_profile_sensor_data : rmw_qos_profile_default;
    image_pub = image_transport::create_camera_publisher(this, "image", qos);
    fps_pub = this->create_publisher<std_msgs::msg::Float32>("publish_fps", rclcpp::QoS(10));

    // 3) 约定 frame_id：从命名空间末尾派生，避免多相机时 frame 名冲突。
    // 例如 /radar/hik_6mm -> hik_6mm_frame。
    auto ns = std::string_view(get_namespace());
    auto ns_pos = ns.rfind('/');
    if (ns_pos != std::string_view::npos && ns_pos + 1 < ns.size()) {
        frame_id = ns.substr(ns.rfind('/') + 1);
        frame_id.append("_frame");
    } else {
        frame_id = "default_camera_frame";
    }

    // 4) 加载相机内参。即使 URL 无效，节点仍可继续运行，只是不会附带有效内参。
    camera_info_manager =
        std::make_unique<camera_info_manager::CameraInfoManager>(this);
    if (camera_info_manager->validateURL(camera_info_url)) {
        camera_info_manager->loadCameraInfo(camera_info_url);
        camera_info_msg = camera_info_manager->getCameraInfo();
    } else {
        RCLCPP_WARN(this->get_logger(), "Invalid camera info URL: %s",
                    camera_info_url.c_str());
    }
    // 5) 初始化硬件并启动采集线程。
    init_camera();

    // 6) 启动监控线程：当采集线程标记 camera_failed 时执行 reset 自恢复。
    RCLCPP_WARN(get_logger(), "Starting Camera Monitor thread.");
    monitor_on = true;
    monitor_thread = std::thread(&HikCameraNode::monitor, this);

    param_client = std::make_shared<rclcpp::AsyncParametersClient>(
        this->get_node_base_interface(),
        this->get_node_topics_interface(),
        this->get_node_graph_interface(),
        this->get_node_services_interface());
    // 参数事件回调只做“置位”，实际参数写入在采集线程里完成，避免跨线程直接操作相机句柄。
    param_event_sub = param_client->on_parameter_event([this](const rcl_interfaces::msg::ParameterEvent::SharedPtr event) {
        // 仅处理本节点参数，避免被其他节点参数事件触发频繁重配。
        if (event->node != this->get_fully_qualified_name()) {
            return;
        }
        for (auto & new_param : event->new_parameters) {
            RCLCPP_INFO(this->get_logger(), "Param %s changed.", new_param.name.c_str());
        }
        for (auto & changed_param : event->changed_parameters) {
            RCLCPP_INFO(this->get_logger(), "Param %s changed.", changed_param.name.c_str());
        }
        param_changed = true;
    });
};

HikCameraNode::~HikCameraNode() {
    // sn: 目标相机序列号；为空时默认选第一台。
    monitor_on = false;
    if (monitor_thread.joinable()) {
        monitor_thread.join();
    }
    close_device();
};

void HikCameraNode::declare_params() {
    this->declare_parameter("sn", "");
    // https://github.com/ros-perception/image_common/blob/136807edb7ff13452214a296fb4819bc63b5b09e/image_transport/src/camera_common.cpp#L62
    this->declare_parameter("camera_info_url", "package://hik_camera/config/6mm.yaml");
    this->declare_parameter("exposure_time", 12000.0);
    this->declare_parameter("gain", 23.0);
    this->declare_parameter("digital_shift", 6.0);
    this->declare_parameter("frame_rate", 60.0);
    this->declare_parameter("enable_fps_log", true);
    this->declare_parameter("fps_log_period_sec", 5.0);
    this->declare_parameter("use_sensor_data_qos", false);
    rotate_180 = this->declare_parameter("rotate_180", false);
    if (rotate_180) {
        RCLCPP_WARN(this->get_logger(), "Rotate 180 degree");
    }
}

void HikCameraNode::init_camera() {
    MV_CC_DEVICE_INFO_LIST device_list;
    bool device_found = false;
    // 循环枚举直到找到目标设备（或节点退出）。
    while (!device_found && rclcpp::ok()) {
        // 枚举设备
        UPDBW(MV_CC_EnumDevices(MV_USB_DEVICE, &device_list));
        std::string sn_to_find = get_parameter("sn").as_string();
        char device_sn[INFO_MAX_BUFFER_SIZE];
        if (device_list.nDeviceNum > 0) {
            if (get_parameter("sn").as_string() == "") {
                // 未设置 sn 时选择第一台设备，适合单相机部署。
                RCLCPP_WARN(this->get_logger(), "Camera SN not set, use the first camera device");
                UPDBE(MV_CC_CreateHandle(&camera_handle, device_list.pDeviceInfo[0]));
                memcpy(device_sn,
                    device_list.pDeviceInfo[0]->SpecialInfo.stUsb3VInfo.chSerialNumber,
                    INFO_MAX_BUFFER_SIZE);
                device_sn[63] = '\0'; // 以防万一
                device_found = true;
            } else {
                // 指定 sn 时遍历设备列表做精确匹配。
                for (size_t i = 0; i < device_list.nDeviceNum; ++i) {
                    memcpy(device_sn,
                           device_list.pDeviceInfo[i]->SpecialInfo.stUsb3VInfo.chSerialNumber,
                           INFO_MAX_BUFFER_SIZE);
                    device_sn[63] = '\0';  // 以防万一
                    // RCLCPP_INFO(this->get_logger(), "Camera SN list: %s", device_sn);
                    if (std::strncmp(device_sn, sn_to_find.c_str(), 64U) == 0) {
                        UPDBE(MV_CC_CreateHandle(&camera_handle, device_list.pDeviceInfo[i]));
                        device_found = true;
                        break;
                    }
                }
            }
        }
        if (device_found) {
            RCLCPP_INFO(this->get_logger(), "Camera SN: %s", device_sn);
            break;
        } else {
            RCLCPP_WARN(this->get_logger(), "Camera SN: %s not found.", sn_to_find.c_str());
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }
    }
    if (device_found) {
        // 找到设备后按“开设备 -> 配参数 -> 开采集”固定流程启动。
        open_device();
        std::this_thread::sleep_for(std::chrono::milliseconds(120));
        set_hk_params();
        std::this_thread::sleep_for(std::chrono::milliseconds(80));
        start_grab();
    }
}

void HikCameraNode::monitor() {
    // 监控线程只负责故障恢复，不参与正常采集。
    while (rclcpp::ok() && monitor_on) {
        if (camera_failed) {
            RCLCPP_ERROR(this->get_logger(), "Camera failed! restarting...");
            reset();
        }
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }
}

void HikCameraNode::start_grab() {
    // 启动 SDK 采集，并拉起独立采集线程。
    constexpr int kStartRetry = 3;
    nRet = MV_OK;
    for (int i = 0; i < kStartRetry; ++i) {
        nRet = MV_CC_StartGrabbing(camera_handle);
        if (nRet == MV_OK) {
            break;
        }
        RCLCPP_WARN(this->get_logger(), "MV_CC_StartGrabbing failed (try %d/%d), error code: %x", i + 1, kStartRetry, (unsigned)nRet);
        MV_CC_StopGrabbing(camera_handle);
        std::this_thread::sleep_for(std::chrono::milliseconds(120));
    }
    if (nRet != MV_OK) {
        RCLCPP_ERROR(this->get_logger(), "MV_CC_StartGrabbing(camera_handle) failed after retries, error code: %x", (unsigned)nRet);
        camera_failed = true;
        grab_on = false;
        return;
    }
    // 开启采集线程
    grab_on = true;
    camera_failed = false;
    fail_cnt = 0;
    capture_thread = std::thread(&HikCameraNode::grab, this);
}

void HikCameraNode::stop_grab() {
    // 先停线程再停 SDK 采集，避免线程还在访问已停止的流。
    grab_on = false;
    if (capture_thread.joinable()) {
        capture_thread.join();
    }
    if (camera_handle) {
        MV_CC_StopGrabbing(camera_handle);
    }
}

std::pair<int, int> HikCameraNode::get_sensor_height_width() {
    // 获取max height/width
    MVCC_INTVALUE _max_height, _max_width;
    UPDBW(MV_CC_GetIntValue(camera_handle, "WidthMax", &_max_width));
    UPDBW(MV_CC_GetIntValue(camera_handle, "HeightMax", &_max_height));
    return std::pair{_max_height.nCurValue, _max_width.nCurValue};
}

void HikCameraNode::set_hk_params() {
    MV_CC_GetImageInfo(camera_handle, &img_info);
    if (hk_first_set_) {
        hk_first_set_ = false;
        // 仅首次设置的一次性参数：工作模式、触发模式、自动曝光等。
        UPDBW(MV_CC_SetEnumValue(camera_handle, "TriggerMode", MV_TRIGGER_MODE_OFF))
        UPDBW(MV_CC_SetEnumValue(camera_handle, "ExposureMode", MV_EXPOSURE_AUTO_MODE_OFF))
        UPDBW(MV_CC_SetEnumValue(camera_handle, "GainAuto", MV_GAIN_MODE_OFF))
        UPDBW(MV_CC_SetBoolValue(camera_handle, "BlackLevelEnable", false))
        UPDBW(MV_CC_SetEnumValue(camera_handle, "BalanceWhiteAuto", MV_BALANCEWHITE_AUTO_ONCE))
        UPDBW(MV_CC_SetEnumValue(camera_handle, "AcquisitionMode", MV_ACQ_MODE_CONTINUOUS))
        UPDBW(MV_CC_SetBoolValue(camera_handle, "AcquisitionFrameRateEnable", true))
    }
    // 每次可动态更新的参数：帧率、曝光、增益。
    const double requested_fps = std::max(1.0, get_parameter("frame_rate").as_double());
    const double requested_exposure_us = std::max(1.0, get_parameter("exposure_time").as_double());
    // 曝光时长必须小于单帧周期，否则相机会裁剪曝光，表现为“怎么拉曝光都不亮”。
    const double max_fps_by_exposure = 1e6 / requested_exposure_us;
    const double safe_fps = std::max(1.0, max_fps_by_exposure * 0.95);
    const double applied_fps = std::min(requested_fps, safe_fps);
    if (requested_fps > safe_fps) {
        RCLCPP_WARN(this->get_logger(),
                    "frame_rate %.2f too high for exposure %.1fus, limiting to %.2f fps.",
                    requested_fps, requested_exposure_us, applied_fps);
    }

    UPDBW(MV_CC_SetFloatValue(camera_handle, "AcquisitionFrameRate", applied_fps))
    UPDBW(MV_CC_SetFloatValue(camera_handle, "ExposureTime", requested_exposure_us))
    const double requested_gain = get_parameter("gain").as_double();
    nRet = MV_CC_SetFloatValue(camera_handle, "Gain", requested_gain);
    if (nRet != MV_OK) {
        MVCC_INTVALUE gain_raw_info{};
        int nRet_gain_raw_info = MV_CC_GetIntValue(camera_handle, "GainRaw", &gain_raw_info);
        if (nRet_gain_raw_info == MV_OK) {
            int64_t gain_raw_to_set = static_cast<int64_t>(std::llround(requested_gain));
            // 若用户按 dB 习惯输入(如 23)，而机型仅支持 GainRaw，做比例映射到 Raw 范围。
            if (requested_gain <= 32.0 && gain_raw_info.nMax > 32) {
                const double ratio = std::clamp(requested_gain / 32.0, 0.0, 1.0);
                gain_raw_to_set = gain_raw_info.nMin +
                    static_cast<int64_t>(std::llround(ratio * (gain_raw_info.nMax - gain_raw_info.nMin)));
            }
            gain_raw_to_set = std::clamp<int64_t>(gain_raw_to_set, gain_raw_info.nMin, gain_raw_info.nMax);
            nRet = MV_CC_SetIntValue(camera_handle, "GainRaw", gain_raw_to_set);
            if (nRet == MV_OK) {
                RCLCPP_WARN(this->get_logger(),
                            "Gain float unsupported, fallback to GainRaw=%ld (raw range: [%ld, %ld]).",
                            static_cast<long>(gain_raw_to_set),
                            static_cast<long>(gain_raw_info.nMin),
                            static_cast<long>(gain_raw_info.nMax));
            } else {
                RCLCPP_WARN(this->get_logger(),
                            "Gain and GainRaw both failed (Gain=%x, GainRaw=%x). Will retry next update.",
                            (unsigned)nRet_gain_raw_info,
                            (unsigned)nRet);
            }
        } else {
            RCLCPP_WARN(this->get_logger(),
                        "Gain not supported or invalid (set failed: %x). Will retry next update.",
                        (unsigned)nRet);
        }
    }
    const double digital_shift = get_parameter("digital_shift").as_double();
    if (digital_shift_available_ && digital_shift > 0.0) {
        // DigitalShift 并非所有机型都支持，失败后自动熔断避免刷屏。
        if (!digital_shift_enable_checked_) {
            nRet = MV_CC_SetBoolValue(camera_handle, "DigitalShiftEnable", true);
            if (nRet != MV_OK) {
                RCLCPP_WARN(this->get_logger(), "DigitalShift not supported (enable failed: %x), disabling it.", (unsigned)nRet);
                digital_shift_available_ = false;
            }
            digital_shift_enable_checked_ = true;
        }
        if (digital_shift_available_) {
            nRet = MV_CC_SetFloatValue(camera_handle, "DigitalShift", digital_shift);
            if (nRet != MV_OK) {
                RCLCPP_WARN(this->get_logger(), "DigitalShift not supported (set failed: %x), disabling it.", (unsigned)nRet);
                digital_shift_available_ = false;
            }
        }
    }

    fps_log_enable_ = get_parameter("enable_fps_log").as_bool();
    fps_log_period_sec_ = std::max(0.2, get_parameter("fps_log_period_sec").as_double());

    MVCC_FLOATVALUE exposure_value{};
    MVCC_FLOATVALUE gain_value{};
    MVCC_FLOATVALUE frame_rate_value{};
    MVCC_INTVALUE gain_raw_value{};
    const bool exposure_ok = (MV_CC_GetFloatValue(camera_handle, "ExposureTime", &exposure_value) == MV_OK);
    const bool fps_ok = (MV_CC_GetFloatValue(camera_handle, "AcquisitionFrameRate", &frame_rate_value) == MV_OK);
    const bool gain_ok = (MV_CC_GetFloatValue(camera_handle, "Gain", &gain_value) == MV_OK);
    const bool gain_raw_ok = (MV_CC_GetIntValue(camera_handle, "GainRaw", &gain_raw_value) == MV_OK);
    if (exposure_ok && fps_ok) {
        if (gain_ok && gain_raw_ok) {
            RCLCPP_INFO(this->get_logger(),
                        "Applied camera params: Exposure=%.1fus Gain=%.2f GainRaw=%ld FPS=%.2f",
                        exposure_value.fCurValue,
                        gain_value.fCurValue,
                        static_cast<long>(gain_raw_value.nCurValue),
                        frame_rate_value.fCurValue);
        } else if (gain_ok) {
            RCLCPP_INFO(this->get_logger(),
                        "Applied camera params: Exposure=%.1fus Gain=%.2f FPS=%.2f",
                        exposure_value.fCurValue,
                        gain_value.fCurValue,
                        frame_rate_value.fCurValue);
        } else if (gain_raw_ok) {
            RCLCPP_INFO(this->get_logger(),
                        "Applied camera params: Exposure=%.1fus GainRaw=%ld FPS=%.2f",
                        exposure_value.fCurValue,
                        static_cast<long>(gain_raw_value.nCurValue),
                        frame_rate_value.fCurValue);
        } else {
            RCLCPP_INFO(this->get_logger(),
                        "Applied camera params: Exposure=%.1fus FPS=%.2f (Gain unreadable)",
                        exposure_value.fCurValue,
                        frame_rate_value.fCurValue);
        }
    }
}

void HikCameraNode::grab() {
    MV_FRAME_OUT out_frame;
    RCLCPP_INFO(this->get_logger(), "Publishing image!");

    // Init convert param
    convert_param.nWidth = img_info.nWidthValue;
    convert_param.nHeight = img_info.nHeightValue;
    convert_param.enDstPixelType = PixelType_Gvsp_BGR8_Packed;

    // 复用一份 image_msg 缓冲，减少每帧动态分配开销。
    sensor_msgs::msg::Image image_msg;
    image_msg.encoding = "bgr8";
    image_msg.height = img_info.nHeightValue;
    image_msg.width = img_info.nWidthValue;
    image_msg.step = img_info.nWidthValue * 3;
    image_msg.data.resize(img_info.nWidthValue * img_info.nHeightValue * 3);

    auto fps_t0 = std::chrono::steady_clock::now();
    int fps_frame_cnt = 0;

    while (rclcpp::ok() && grab_on)
    {
        if (param_changed)
        {
            // 参数热更新在采集线程生效，确保和相机 SDK 调用同线程。
            set_hk_params();
            param_changed = false;
        }
        nRet = MV_CC_GetImageBuffer(camera_handle, &out_frame, 1000);
        if (MV_OK == nRet) {
            // 1) 把厂商原始像素格式转换到 ROS 常用的 BGR8。
            convert_param.pDstBuffer = image_msg.data.data();
            convert_param.nDstBufferSize = image_msg.data.size();
            convert_param.pSrcData = out_frame.pBufAddr;
            convert_param.nSrcDataLen = out_frame.stFrameInfo.nFrameLen;
            convert_param.enSrcPixelType = out_frame.stFrameInfo.enPixelType;
            MV_CC_ConvertPixelType(camera_handle, &convert_param);

            if (rotate_180)
            {
                // 2) 可选 180 度旋转：原地交换像素，避免额外图像对象拷贝。
                for (unsigned i = 0; i < img_info.nWidthValue * img_info.nHeightValue / 2; i++) {
                    std::swap(image_msg.data[i * 3], image_msg.data[(img_info.nWidthValue * img_info.nHeightValue - 1 - i) * 3]);
                    std::swap(image_msg.data[i * 3 + 1], image_msg.data[(img_info.nWidthValue * img_info.nHeightValue - 1 - i) * 3 + 1]);
                    std::swap(image_msg.data[i * 3 + 2], image_msg.data[(img_info.nWidthValue * img_info.nHeightValue - 1 - i) * 3 + 2]);
                }
            }

            // 3) 时间戳和 frame_id 统一赋值，再同步发布 image + camera_info。
            auto time_now = this->now();
            image_msg.header.stamp = this->now();
            image_msg.header.frame_id = frame_id;
            camera_info_msg.header.stamp = time_now;
            camera_info_msg.header.frame_id = frame_id;
            image_pub.publish(image_msg, camera_info_msg);
            ++fps_frame_cnt;

            if (fps_log_enable_) {
                auto fps_t1 = std::chrono::steady_clock::now();
                const double dt = std::chrono::duration<double>(fps_t1 - fps_t0).count();
                if (dt >= fps_log_period_sec_) {
                    const double publish_fps = static_cast<double>(fps_frame_cnt) / dt;
                    std_msgs::msg::Float32 fps_msg;
                    fps_msg.data = static_cast<float>(publish_fps);
                    fps_pub->publish(fps_msg);
                    RCLCPP_INFO(this->get_logger(), "Publish FPS: %.2f", publish_fps);
                    fps_t0 = fps_t1;
                    fps_frame_cnt = 0;
                }
            }

            // 4) 必须及时归还 SDK 缓冲，否则后续取帧会阻塞/丢帧。
            MV_CC_FreeImageBuffer(camera_handle, &out_frame);
            fail_cnt = 0;
        }
        else
        {
            // 连续失败计数用于触发监控线程重启。
            RCLCPP_WARN(this->get_logger(), "Get buffer failed! nRet: [%x]", nRet);
            // MV_CC_StopGrabbing(camera_handle);
            // MV_CC_StartGrabbing(camera_handle);
            fail_cnt++;
        }

        if (fail_cnt > 5)
        {
            // 标记故障后退出采集循环，由 monitor 线程执行 reset。
            RCLCPP_FATAL(this->get_logger(), "Camera failed!");
            grab_on = false;
            camera_failed = true;
        }
    }
}

void HikCameraNode::reset()
{
    // 故障恢复流程：完整关闭 -> 等待硬件稳定 -> 重新初始化。
    close_device();
    std::this_thread::sleep_for(std::chrono::seconds(1));
    init_camera();
}

void HikCameraNode::open_device()
{
    // 保持最小调用序：创建句柄后直接打开设备，避免额外状态切换。
    UPDBE(MV_CC_OpenDevice(camera_handle));

    // 每次重连后都要重新进行一次性参数配置。
    hk_first_set_ = true;
    gain_available_ = true;
    digital_shift_available_ = true;
    digital_shift_enable_checked_ = false;
}

void HikCameraNode::close_device()
{
    // 关闭顺序：先停采集线程，再关设备句柄。
    stop_grab();
    if (camera_handle)
    {
        MV_CC_CloseDevice(camera_handle);
        MV_CC_DestroyHandle(camera_handle);
        camera_handle = nullptr;
    }
    RCLCPP_INFO(this->get_logger(), "HikCamera closed!");
}

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(hik_camera::HikCameraNode)

// int main(int argc, char* argv[]) {
//     rclcpp::init(argc, argv);
//     auto node = std::make_shared<HikCameraNode>(rclcpp::NodeOptions());
//     rclcpp::spin(node);
//     rclcpp::shutdown();
//     return 0;
// }
