#include "LIVMapper.h"
#include "glogger.h"

int main(int argc, char **argv) {
  // 方便报错时找到出错的位置
  GLogger glogger(argc, argv, "", "");
  ros::init(argc, argv, "laserMapping");
  ros::NodeHandle nh;
  image_transport::ImageTransport it(nh);
  LIVMapper mapper(nh);
  mapper.initializeSubscribersAndPublishers(nh, it);
  mapper.run();
  return 0;
}