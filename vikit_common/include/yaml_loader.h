#ifndef VIKIT_CAMERA_LOADER_H_
#define VIKIT_CAMERA_LOADER_H_

#include <string>
#include <vector>
#include <yaml-cpp/yaml.h>
#include <vikit/abstract_camera.h>
#include <vikit/pinhole_camera.h>
#include <vikit/atan_camera.h>
#include <vikit/omni_camera.h>
#include <vikit/equidistant_camera.h>
#include <vikit/polynomial_camera.h>

namespace vk {
namespace camera_loader {

// 辅助函数：安全的 YAML 参数获取
template <typename T>
T getYamlParam(const YAML::Node& node, const std::string& key, T default_val = T()) {
  return node[key] ? node[key].as<T>() : default_val;
}

/// 从 YAML 节点加载单个相机
bool loadFromYaml(const YAML::Node& node, vk::AbstractCamera*& cam) {
  bool res = true;
  std::string cam_model = node["cam_model"].as<std::string>();
  
  if (cam_model == "Ocam") {
    cam = new vk::OmniCamera(node["cam_calib_file"].as<std::string>());
  }
  else if (cam_model == "Pinhole") {
    cam = new vk::PinholeCamera(
        node["cam_width"].as<int>(),
        node["cam_height"].as<int>(),
        getYamlParam(node, "scale", 1.0),
        node["cam_fx"].as<double>(),
        node["cam_fy"].as<double>(),
        node["cam_cx"].as<double>(),
        node["cam_cy"].as<double>(),
        getYamlParam(node, "cam_d0", 0.0),
        getYamlParam(node, "cam_d1", 0.0),
        getYamlParam(node, "cam_d2", 0.0),
        getYamlParam(node, "cam_d3", 0.0));
  }
  else if (cam_model == "EquidistantCamera") {
    cam = new vk::EquidistantCamera(
        node["cam_width"].as<int>(),
        node["cam_height"].as<int>(),
        getYamlParam(node, "scale", 1.0),
        node["cam_fx"].as<double>(),
        node["cam_fy"].as<double>(),
        node["cam_cx"].as<double>(),
        node["cam_cy"].as<double>(),
        getYamlParam(node, "k1", 0.0),
        getYamlParam(node, "k2", 0.0),
        getYamlParam(node, "k3", 0.0),
        getYamlParam(node, "k4", 0.0));
  }
  else if (cam_model == "PolynomialCamera") {
    cam = new vk::PolynomialCamera(
        node["cam_width"].as<int>(),
        node["cam_height"].as<int>(),
        node["cam_fx"].as<double>(),
        node["cam_fy"].as<double>(),
        node["cam_cx"].as<double>(),
        node["cam_cy"].as<double>(),
        getYamlParam(node, "cam_skew", 0.0),
        getYamlParam(node, "k2", 0.0),
        getYamlParam(node, "k3", 0.0),
        getYamlParam(node, "k4", 0.0),
        getYamlParam(node, "k5", 0.0),
        getYamlParam(node, "k6", 0.0),
        getYamlParam(node, "k7", 0.0));
  }
  else if (cam_model == "ATAN") {
    cam = new vk::ATANCamera(
        node["cam_width"].as<int>(),
        node["cam_height"].as<int>(),
        node["cam_fx"].as<double>(),
        node["cam_fy"].as<double>(),
        node["cam_cx"].as<double>(),
        node["cam_cy"].as<double>(),
        node["cam_d0"].as<double>());
  }
  else {
    cam = nullptr;
    res = false;
  }
  return res;
}

/// 从 YAML 文件加载多个相机
bool loadFromYamlFile(const std::string& yaml_file, std::vector<vk::AbstractCamera*>& cam_list) {
  try {
    YAML::Node config = YAML::LoadFile(yaml_file);
    int cam_num = config["cam_num"].as<int>();
    
    for (int i = 0; i < cam_num; ++i) {
      std::string cam_key = "cam_" + std::to_string(i);
      YAML::Node cam_node = config[cam_key];
      
      vk::AbstractCamera* camera = nullptr;
      if (loadFromYaml(cam_node, camera)) {
        cam_list.push_back(camera);
      } else {
        // 清理已创建相机
        for (auto& cam : cam_list) delete cam;
        cam_list.clear();
        return false;
      }
    }
    return true;
  } catch (const YAML::Exception& e) {
    std::cerr << "YAML Error: " << e.what() << std::endl;
    return false;
  }
}

} // namespace camera_loader
} // namespace vk

#endif // VIKIT_CAMERA_LOADER_H_