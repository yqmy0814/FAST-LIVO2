/*
This file is part of FAST-LIVO2: Fast, Direct LiDAR-Inertial-Visual Odometry.

Developer: Chunran Zheng <zhengcr@connect.hku.hk>

For commercial use, please contact me at <zhengcr@connect.hku.hk> or
Prof. Fu Zhang at <fuzhang@hku.hk>.

This file is subject to the terms and conditions outlined in the 'LICENSE' file,
which is included as part of this source code package.
*/

#ifndef PREPROCESS_H_
#define PREPROCESS_H_

#include <livox_ros_driver/CustomMsg.h>
#include <pcl_conversions/pcl_conversions.h>

#include "common_lib.h"

#define IS_VALID(a) ((abs(a) > 1e8) ? true : false)

enum LiDARFeature {
  Nor,
  Poss_Plane,
  Real_Plane,
  Edge_Jump,
  Edge_Plane,
  Wire,
  ZeroPoint
};
enum Surround { Prev, Next };
enum E_jump { Nr_nor, Nr_zero, Nr_180, Nr_inf, Nr_blind };

struct orgtype {
  double range;
  double dista;
  double angle[2];
  double intersect;
  E_jump edj[2];
  LiDARFeature ftype;
  orgtype() {
    range = 0;
    edj[Prev] = Nr_nor;
    edj[Next] = Nr_nor;
    ftype = Nor;
    intersect = 2;
  }
};

/*** Velodyne ***/
namespace velodyne_ros {
struct EIGEN_ALIGN16 Point {
  PCL_ADD_POINT4D;
  float intensity;
  float time;
  std::uint16_t ring;
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};
}  // namespace velodyne_ros
POINT_CLOUD_REGISTER_POINT_STRUCT(
    velodyne_ros::Point,
    (float, x, x)(float, y, y)(float, z, z)(float, intensity, intensity)(
        float, time, time)(std::uint16_t, ring, ring))
/****************/

/*** Ouster ***/
namespace ouster_ros {
struct EIGEN_ALIGN16 Point {
  PCL_ADD_POINT4D;
  float intensity;
  std::uint32_t t;
  std::uint16_t reflectivity;
  uint8_t ring;
  std::uint16_t ambient;
  std::uint32_t range;
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};
}  // namespace ouster_ros
POINT_CLOUD_REGISTER_POINT_STRUCT(
    ouster_ros::Point,
    (float, x, x)(float, y, y)(float, z, z)(float, intensity, intensity)(
        std::uint32_t, t, t)(std::uint16_t, reflectivity, reflectivity)(
        std::uint8_t, ring, ring)(std::uint16_t, ambient,
                                  ambient)(std::uint32_t, range, range))
/****************/

/*** Hesai_XT32 ***/
namespace xt32_ros {
struct EIGEN_ALIGN16 Point {
  PCL_ADD_POINT4D;
  float intensity;
  double timestamp;
  std::uint16_t ring;
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};
}  // namespace xt32_ros
POINT_CLOUD_REGISTER_POINT_STRUCT(
    xt32_ros::Point,
    (float, x, x)(float, y, y)(float, z, z)(float, intensity, intensity)(
        double, timestamp, timestamp)(std::uint16_t, ring, ring))
/*****************/

/*** Hesai_Pandar128 ***/
namespace Pandar128_ros {
struct EIGEN_ALIGN16 Point {
  PCL_ADD_POINT4D;
  uint8_t intensity;
  double timestamp;
  uint16_t ring;
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};
}  // namespace Pandar128_ros
POINT_CLOUD_REGISTER_POINT_STRUCT(
    Pandar128_ros::Point,
    (float, x, x)(float, y, y)(float, z, z)(std::uint8_t, intensity, intensity)(
        double, timestamp, timestamp)(std::uint16_t, ring, ring))
/*****************/

/*** Robosense_Airy ***/
namespace robosense_ros {
struct EIGEN_ALIGN16 Point {
  PCL_ADD_POINT4D;
  float intensity;
  double timestamp;
  uint16_t ring;
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};
}  // namespace robosense_ros
POINT_CLOUD_REGISTER_POINT_STRUCT(
    robosense_ros::Point,
    (float, x, x)(float, y, y)(float, z, z)(float, intensity, intensity)(
        double, timestamp, timestamp)(std::uint16_t, ring, ring))
/*****************/

class Preprocess {
 public:
  //   EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  Preprocess();
  ~Preprocess();

  void Process(const livox_ros_driver::CustomMsg::ConstPtr &msg,
               PointCloudXYZIN::Ptr &pcl_out);
  void Process(const sensor_msgs::PointCloud2::ConstPtr &msg,
               PointCloudXYZIN::Ptr &pcl_out);
  void Set(bool feat_en, int lid_type, double bld, int pfilt_num);

  // sensor_msgs::PointCloud2::ConstPtr pointcloud;
  PointCloudXYZIN pl_full_, pl_corn_, pl_surf_;
  PointCloudXYZIN pl_buff_[128];  // maximum 128 line lidar
  std::vector<orgtype> typess_[128];   // maximum 128 line lidar
  int lidar_type_, point_filter_num_, n_scans_;

  double blind_, blind_sqr_;
  bool feature_enabled_, given_offset_time_;
  ros::Publisher pub_full_, pub_surf_, pub_corn_;

 private:
  void AviaHandler(const livox_ros_driver::CustomMsg::ConstPtr &msg);
  void Oust64Handler(const sensor_msgs::PointCloud2::ConstPtr &msg);
  void VelodyneHandler(const sensor_msgs::PointCloud2::ConstPtr &msg);
  void Xt32Handler(const sensor_msgs::PointCloud2::ConstPtr &msg);
  void Pandar128Handler(const sensor_msgs::PointCloud2::ConstPtr &msg);
  void RobosenseHandler(const sensor_msgs::PointCloud2::ConstPtr &msg);
  void L515Handler(const sensor_msgs::PointCloud2::ConstPtr &msg);
  void GiveFeature(PointCloudXYZIN &pl, std::vector<orgtype> &types);
  void PubFunc(PointCloudXYZIN &pl, const ros::Time &ct);
  int PlaneJudge(const PointCloudXYZIN &pl, std::vector<orgtype> &types, uint i,
                  uint &i_nex, Eigen::Vector3d &curr_direct);
  bool EdgeJumpJudge(const PointCloudXYZIN &pl, std::vector<orgtype> &types,
                       uint i, Surround nor_dir);

  int group_size_;
  double dis_a_, dis_b_, inf_bound_;
  double limit_maxmid_, limit_midmin_, limit_maxmin_;
  double p2l_ratio_;
  double jump_up_limit_, jump_down_limit_;
  double cos160_;
  double edge_a_, edge_b_;
  double smallp_intersect_, smallp_ratio_;
  double vx_, vy_, vz_;
};
typedef std::shared_ptr<Preprocess> PreprocessPtr;

#endif  // PREPROCESS_H_