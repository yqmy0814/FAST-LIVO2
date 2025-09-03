/* 
This file is part of FAST-LIVO2: Fast, Direct LiDAR-Inertial-Visual Odometry.

Developer: Chunran Zheng <zhengcr@connect.hku.hk>

For commercial use, please contact me at <zhengcr@connect.hku.hk> or
Prof. Fu Zhang at <fuzhang@hku.hk>.

This file is subject to the terms and conditions outlined in the 'LICENSE' file,
which is included as part of this source code package.
*/

#ifndef IMU_PROCESSING_H
#define IMU_PROCESSING_H

#include <Eigen/Eigen>
#include "common_lib.h"
#include <condition_variable>
#include <nav_msgs/Odometry.h>
#include <utils/so3_math.h>

#include <fstream>
const bool time_list(PointXYZIN &x, PointXYZIN &y) { return (x.curvature < y.curvature); }

/// *************IMU Process and undistortion
class ImuProcess
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  ImuProcess();
  ~ImuProcess();

  void Reset();
  void SetExtrinsic(const V3D &transl, const M3D &rot);
  void SetExtrinsic(const V3D &transl);
  void SetExtrinsic(const MD(4, 4) & T);
  void SetGyrCovScale(const V3D &scaler);
  void SetAccCovScale(const V3D &scaler);
  void SetGyrBiasCov(const V3D &b_g);
  void SetAccBiasCov(const V3D &b_a);
  void SetInvExpoCov(const double &inv_expo);
  void SetImuInitFrameNum(const int &num);
  void DisableGravityEst();
  void DisableBiasEst();
  void DisableExposureEst();
  void Process(LidarMeasureGroup &lidar_meas, StatesGroup &stat, PointCloudXYZIN::Ptr cur_pcl_un_);
  void UndistortPcl(LidarMeasureGroup &lidar_meas, StatesGroup &state_inout, PointCloudXYZIN &pcl_out);

  std::ofstream fout_imu_;
  double imu_mean_acc_norm_;
  V3D unbiased_gyr_;

  V3D cov_acc_;
  V3D cov_gyr_;
  V3D cov_bias_gyr_;
  V3D cov_bias_acc_;
  double cov_inv_expo_;
  double first_lidar_time_;
  bool imu_time_init_ = false;
  bool imu_need_init_ = true;
  M3D eye3d_;
  V3D zero3d_;
  int lidar_type_;

private:
  void ImuInit(const MeasureGroup &meas, StatesGroup &state, int &N);
  PointCloudXYZIN pcl_wait_proc_;
  sensor_msgs::ImuConstPtr last_imu_;
  PointCloudXYZIN::Ptr cur_pcl_un_;
  std::vector<Pose6D> imu_pose_;
  M3D lidar_rot_to_imu_;
  V3D lidar_offset_to_imu_;
  V3D mean_acc_;
  V3D mean_gyr_;
  V3D angvel_last_;
  V3D acc_s_last_;
  double last_prop_end_time_;
  double time_last_scan_;
  int init_iter_num_ = 1, max_ini_count_ = 20;
  bool b_first_frame_ = true;
  bool gravity_est_en_ = true;
  bool ba_bg_est_en_ = true;
  bool exposure_estimate_en_ = true;
};
typedef std::shared_ptr<ImuProcess> ImuProcessPtr;
#endif