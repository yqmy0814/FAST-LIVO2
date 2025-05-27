#include "LIVMapper.h"
#include "glogger.h"

DEFINE_string(camera_config, "config/camera_fisheye_HILTI22.yaml", "camera config file.");

int main(int argc, char **argv) {
  // 方便报错时找到出错的位置
  GLogger glogger(argc, argv, "", "");
  ros::init(argc, argv, "laserMapping");
  ros::NodeHandle nh;
  image_transport::ImageTransport it(nh);
  LIVMapper mapper(nh,FLAGS_camera_config);
  mapper.InitializeSubscribersAndPublishers(nh, it);
  mapper.Run();
  return 0;
}