/*
This file is part of FAST-LIVO2: Fast, Direct LiDAR-Inertial-Visual Odometry.

Developer: Chunran Zheng <zhengcr@connect.hku.hk>

For commercial use, please contact me at <zhengcr@connect.hku.hk> or
Prof. Fu Zhang at <fuzhang@hku.hk>.

This file is subject to the terms and conditions outlined in the 'LICENSE' file,
which is included as part of this source code package.
*/

#include "preprocess.h"

#include "omp.h"

#define RETURN0 0x00
#define RETURN0AND1 0x10

Preprocess::Preprocess()
    : feature_enabled_(0), lidar_type_(AVIA), blind_(0.01), point_filter_num_(1) {
  inf_bound_ = 10;
  n_scans_ = 6;
  group_size_ = 8;
  dis_a_ = 0.01;
  dis_a_ = 0.1;  // B?
  p2l_ratio_ = 225;
  limit_maxmid_ = 6.25;
  limit_midmin_ = 6.25;
  limit_maxmin_ = 3.24;
  jump_up_limit_ = 170.0;
  jump_down_limit_ = 8.0;
  cos160_ = 160.0;
  edge_a_ = 2;
  edge_b_ = 0.1;
  smallp_intersect_ = 172.5;
  smallp_ratio_ = 1.2;
  given_offset_time_ = false;

  jump_up_limit_ = cos(jump_up_limit_ / 180 * M_PI);
  jump_down_limit_ = cos(jump_down_limit_ / 180 * M_PI);
  cos160_ = cos(cos160_ / 180 * M_PI);
  smallp_intersect_ = cos(smallp_intersect_ / 180 * M_PI);
}

Preprocess::~Preprocess() {}

void Preprocess::Set(bool feat_en, int lid_type, double bld, int pfilt_num) {
  feature_enabled_ = feat_en;
  lidar_type_ = lid_type;
  blind_ = bld;
  point_filter_num_ = pfilt_num;
}

void Preprocess::Process(const livox_ros_driver::CustomMsg::ConstPtr &msg,
                         PointCloudXYZIN::Ptr &pcl_out) {
  AviaHandler(msg);
  *pcl_out = pl_surf_;
}

void Preprocess::Process(const sensor_msgs::PointCloud2::ConstPtr &msg,
                         PointCloudXYZIN::Ptr &pcl_out) {
  switch (lidar_type_) {
    case OUST64:
      Oust64Handler(msg);
      break;

    case VELO16:
      VelodyneHandler(msg);
      break;

    case L515:
      L515Handler(msg);
      break;

    case XT32:
      Xt32Handler(msg);
      break;

    case PANDAR128:
      Pandar128Handler(msg);
      break;

    case ROBOSENSE:
      RobosenseHandler(msg);
      break;

    default:
      printf("Error LiDAR Type: %d \n", lidar_type_);
      break;
  }
  *pcl_out = pl_surf_;
}

void Preprocess::AviaHandler(
    const livox_ros_driver::CustomMsg::ConstPtr &msg) {
  pl_surf_.clear();
  pl_corn_.clear();
  pl_full_.clear();
  double t1 = omp_get_wtime();
  int plsize = msg->point_num;
  printf("[ Preprocess ] Input point number: %d \n", plsize);
  // printf("point_filter_num: %d\n", point_filter_num);

  pl_corn_.reserve(plsize);
  pl_surf_.reserve(plsize);
  pl_full_.resize(plsize);

  for (int i = 0; i < n_scans_; i++) {
    pl_buff_[i].clear();
    pl_buff_[i].reserve(plsize);
  }
  uint valid_num = 0;

  if (feature_enabled_) {
    for (uint i = 1; i < plsize; i++) {
      if ((msg->points[i].line < n_scans_) &&
          ((msg->points[i].tag & 0x30) == 0x10)) {
        pl_full_[i].x = msg->points[i].x;
        pl_full_[i].y = msg->points[i].y;
        pl_full_[i].z = msg->points[i].z;
        pl_full_[i].intensity = msg->points[i].reflectivity;
        pl_full_[i].curvature =
            msg->points[i].offset_time /
            float(1000000);  // use curvature as time of each laser points

        bool is_new = false;
        if ((abs(pl_full_[i].x - pl_full_[i - 1].x) > 1e-7) ||
            (abs(pl_full_[i].y - pl_full_[i - 1].y) > 1e-7) ||
            (abs(pl_full_[i].z - pl_full_[i - 1].z) > 1e-7)) {
          pl_buff_[msg->points[i].line].push_back(pl_full_[i]);
        }
      }
    }
    static int count = 0;
    static double time = 0.0;
    count++;
    double t0 = omp_get_wtime();
    for (int j = 0; j < n_scans_; j++) {
      if (pl_buff_[j].size() <= 5) continue;
      pcl::PointCloud<PointXYZIN> &pl = pl_buff_[j];
      plsize = pl.size();
      vector<orgtype> &types = typess_[j];
      types.clear();
      types.resize(plsize);
      plsize--;
      for (uint i = 0; i < plsize; i++) {
        types[i].range = pl[i].x * pl[i].x + pl[i].y * pl[i].y;
        vx_ = pl[i].x - pl[i + 1].x;
        vy_ = pl[i].y - pl[i + 1].y;
        vz_ = pl[i].z - pl[i + 1].z;
        types[i].dista = vx_ * vx_ + vy_ * vy_ + vz_ * vz_;
      }
      types[plsize].range =
          pl[plsize].x * pl[plsize].x + pl[plsize].y * pl[plsize].y;
      GiveFeature(pl, types);
      // pl_surf += pl;
    }
    time += omp_get_wtime() - t0;
    printf("Feature extraction time: %lf \n", time / count);
  } else {
    for (uint i = 0; i < plsize; i++) {
      if ((msg->points[i].line <
           n_scans_))  // && ((msg->points[i].tag & 0x30) == 0x10))
      {
        valid_num++;

        pl_full_[i].x = msg->points[i].x;
        pl_full_[i].y = msg->points[i].y;
        pl_full_[i].z = msg->points[i].z;
        pl_full_[i].intensity = msg->points[i].reflectivity;
        pl_full_[i].curvature =
            msg->points[i].offset_time /
            float(1000000);  // use curvature as time of each laser points
        if (msg->points[i].offset_time > 1000000000)
          std::cout << std::fixed
                    << "[ROSIO] point time :" << msg->points[i].offset_time;

        if (i == 0)
          pl_full_[i].curvature =
              fabs(pl_full_[i].curvature) < 1.0 ? pl_full_[i].curvature : 0.0;
        else {
          // if(fabs(pl_full[i].curvature - pl_full[i - 1].curvature) > 1.0)
          // ROS_ERROR("time jump: %f", fabs(pl_full[i].curvature - pl_full[i -
          // 1].curvature));
          pl_full_[i].curvature =
              fabs(pl_full_[i].curvature - pl_full_[i - 1].curvature) < 1.0
                  ? pl_full_[i].curvature
                  : pl_full_[i - 1].curvature +
                        0.004166667f;  // float(100/24000)
        }

        if (valid_num % point_filter_num_ == 0) {
          if (pl_full_[i].x * pl_full_[i].x + pl_full_[i].y * pl_full_[i].y +
                  pl_full_[i].z * pl_full_[i].z >=
              blind_sqr_) {
            pl_surf_.push_back(pl_full_[i]);
            // if (i % 100 == 0 || i == 0) printf("pl_full[i].curvature: %f \n",
            // pl_full[i].curvature);
          }
        }
      }
    }
  }
  printf("[ Preprocess ] Output point number: %zu \n", pl_surf_.points.size());
}

void Preprocess::L515Handler(const sensor_msgs::PointCloud2::ConstPtr &msg) {
  pl_surf_.clear();
  pl_corn_.clear();
  pl_full_.clear();
  pcl::PointCloud<pcl::PointXYZRGB> pl_orig;
  pcl::fromROSMsg(*msg, pl_orig);
  int plsize = pl_orig.size();
  pl_corn_.reserve(plsize);
  pl_surf_.reserve(plsize);

  double time_stamp = msg->header.stamp.toSec();
  // cout << "===================================" << endl;
  // printf("Pt size = %d, N_SCANS = %d\r\n", plsize, N_SCANS);
  for (int i = 0; i < pl_orig.points.size(); i++) {
    if (i % point_filter_num_ != 0) continue;

    double range = pl_orig.points[i].x * pl_orig.points[i].x +
                   pl_orig.points[i].y * pl_orig.points[i].y +
                   pl_orig.points[i].z * pl_orig.points[i].z;

    if (range < blind_sqr_) continue;

    Eigen::Vector3d pt_vec;
    PointXYZIN added_pt;
    added_pt.x = pl_orig.points[i].x;
    added_pt.y = pl_orig.points[i].y;
    added_pt.z = pl_orig.points[i].z;
    added_pt.normal_x = pl_orig.points[i].r;
    added_pt.normal_y = pl_orig.points[i].g;
    added_pt.normal_z = pl_orig.points[i].b;

    added_pt.curvature = 0.0;
    pl_surf_.points.push_back(added_pt);
  }

  std::cout << "pl size:: " << pl_orig.points.size() << endl;
  // pub_func(pl_surf, pub_full, msg->header.stamp);
  // pub_func(pl_surf, pub_corn, msg->header.stamp);
}

void Preprocess::Oust64Handler(const sensor_msgs::PointCloud2::ConstPtr &msg) {
  pl_surf_.clear();
  pl_corn_.clear();
  pl_full_.clear();
  pcl::PointCloud<ouster_ros::Point> pl_orig;
  pcl::fromROSMsg(*msg, pl_orig);
  int plsize = pl_orig.size();
  pl_corn_.reserve(plsize);
  pl_surf_.reserve(plsize);
  if (feature_enabled_) {
    for (int i = 0; i < n_scans_; i++) {
      pl_buff_[i].clear();
      pl_buff_[i].reserve(plsize);
    }

    for (uint i = 0; i < plsize; i++) {
      double range = pl_orig.points[i].x * pl_orig.points[i].x +
                     pl_orig.points[i].y * pl_orig.points[i].y +
                     pl_orig.points[i].z * pl_orig.points[i].z;
      if (range < blind_sqr_) continue;
      Eigen::Vector3d pt_vec;
      PointXYZIN added_pt;
      added_pt.x = pl_orig.points[i].x;
      added_pt.y = pl_orig.points[i].y;
      added_pt.z = pl_orig.points[i].z;
      added_pt.intensity = pl_orig.points[i].intensity;
      added_pt.normal_x = 0;
      added_pt.normal_y = 0;
      added_pt.normal_z = 0;
      double yaw_angle = atan2(added_pt.y, added_pt.x) * 57.3;
      if (yaw_angle >= 180.0) yaw_angle -= 360.0;
      if (yaw_angle <= -180.0) yaw_angle += 360.0;

      added_pt.curvature = pl_orig.points[i].t / 1e6;
      if (pl_orig.points[i].ring < n_scans_) {
        pl_buff_[pl_orig.points[i].ring].push_back(added_pt);
      }
    }

    for (int j = 0; j < n_scans_; j++) {
      PointCloudXYZIN &pl = pl_buff_[j];
      int linesize = pl.size();
      vector<orgtype> &types = typess_[j];
      types.clear();
      types.resize(linesize);
      linesize--;
      for (uint i = 0; i < linesize; i++) {
        types[i].range = sqrt(pl[i].x * pl[i].x + pl[i].y * pl[i].y);
        vx_ = pl[i].x - pl[i + 1].x;
        vy_ = pl[i].y - pl[i + 1].y;
        vz_ = pl[i].z - pl[i + 1].z;
        types[i].dista = vx_ * vx_ + vy_ * vy_ + vz_ * vz_;
      }
      types[linesize].range = sqrt(pl[linesize].x * pl[linesize].x +
                                   pl[linesize].y * pl[linesize].y);
      GiveFeature(pl, types);
    }
  } else {
    double time_stamp = msg->header.stamp.toSec();
    // cout << "===================================" << endl;
    // printf("Pt size = %d, N_SCANS = %d\r\n", plsize, N_SCANS);
    for (int i = 0; i < pl_orig.points.size(); i++) {
      if (i % point_filter_num_ != 0) continue;

      double range = pl_orig.points[i].x * pl_orig.points[i].x +
                     pl_orig.points[i].y * pl_orig.points[i].y +
                     pl_orig.points[i].z * pl_orig.points[i].z;

      if (range < blind_sqr_) continue;

      Eigen::Vector3d pt_vec;
      PointXYZIN added_pt;
      added_pt.x = pl_orig.points[i].x;
      added_pt.y = pl_orig.points[i].y;
      added_pt.z = pl_orig.points[i].z;
      added_pt.intensity = pl_orig.points[i].intensity;
      added_pt.normal_x = 0;
      added_pt.normal_y = 0;
      added_pt.normal_z = 0;
      double yaw_angle = atan2(added_pt.y, added_pt.x) * 57.3;
      if (yaw_angle >= 180.0) yaw_angle -= 360.0;
      if (yaw_angle <= -180.0) yaw_angle += 360.0;

      added_pt.curvature = pl_orig.points[i].t / 1e6;

      // cout<<added_pt.curvature<<endl;

      pl_surf_.points.push_back(added_pt);
    }
    std::sort(pl_surf_.points.begin(), pl_surf_.points.end(),
              [](const PointXYZIN &a, const PointXYZIN &b) {
                return a.curvature < b.curvature;
              });
  }
  // pub_func(pl_surf, pub_full, msg->header.stamp);
  // pub_func(pl_surf, pub_corn, msg->header.stamp);
}

#define MAX_LINE_NUM 64

void Preprocess::VelodyneHandler(
    const sensor_msgs::PointCloud2::ConstPtr &msg) {
  pl_surf_.clear();
  pl_corn_.clear();
  pl_full_.clear();

  pcl::PointCloud<velodyne_ros::Point> pl_orig;
  pcl::fromROSMsg(*msg, pl_orig);
  int plsize = pl_orig.points.size();
  if (plsize == 0) return;
  pl_surf_.reserve(plsize);

  bool is_first[MAX_LINE_NUM];
  double yaw_fp[MAX_LINE_NUM] = {0};      // yaw of first scan point
  double omega_l = 3.61;                  // scan angular velocity
  float yaw_last[MAX_LINE_NUM] = {0.0};   // yaw of last scan point
  float time_last[MAX_LINE_NUM] = {0.0};  // last offset time

  if (pl_orig.points[plsize - 1].time > 0) {
    given_offset_time_ = true;
  } else {
    given_offset_time_ = false;
    memset(is_first, true, sizeof(is_first));
    double yaw_first =
        atan2(pl_orig.points[0].y, pl_orig.points[0].x) * 57.29578;
    double yaw_end = yaw_first;
    int layer_first = pl_orig.points[0].ring;
    for (uint i = plsize - 1; i > 0; i--) {
      if (pl_orig.points[i].ring == layer_first) {
        yaw_end = atan2(pl_orig.points[i].y, pl_orig.points[i].x) * 57.29578;
        break;
      }
    }
  }

  if (feature_enabled_) {
    for (int i = 0; i < n_scans_; i++) {
      pl_buff_[i].clear();
      pl_buff_[i].reserve(plsize);
    }

    for (int i = 0; i < plsize; i++) {
      PointXYZIN added_pt;
      added_pt.normal_x = 0;
      added_pt.normal_y = 0;
      added_pt.normal_z = 0;
      int layer = pl_orig.points[i].ring;
      if (layer >= n_scans_) continue;
      added_pt.x = pl_orig.points[i].x;
      added_pt.y = pl_orig.points[i].y;
      added_pt.z = pl_orig.points[i].z;
      added_pt.intensity = pl_orig.points[i].intensity;
      added_pt.curvature = pl_orig.points[i].time / 1000.0;  // units: ms

      if (!given_offset_time_) {
        double yaw_angle = atan2(added_pt.y, added_pt.x) * 57.2957;
        if (is_first[layer]) {
          // printf("layer: %d; is first: %d", layer, is_first[layer]);
          yaw_fp[layer] = yaw_angle;
          is_first[layer] = false;
          added_pt.curvature = 0.0;
          yaw_last[layer] = yaw_angle;
          time_last[layer] = added_pt.curvature;
          continue;
        }

        if (yaw_angle <= yaw_fp[layer]) {
          added_pt.curvature = (yaw_fp[layer] - yaw_angle) / omega_l;
        } else {
          added_pt.curvature = (yaw_fp[layer] - yaw_angle + 360.0) / omega_l;
        }

        if (added_pt.curvature < time_last[layer])
          added_pt.curvature += 360.0 / omega_l;

        yaw_last[layer] = yaw_angle;
        time_last[layer] = added_pt.curvature;
      }

      pl_buff_[layer].points.push_back(added_pt);
    }

    for (int j = 0; j < n_scans_; j++) {
      PointCloudXYZIN &pl = pl_buff_[j];
      int linesize = pl.size();
      if (linesize < 2) continue;
      vector<orgtype> &types = typess_[j];
      types.clear();
      types.resize(linesize);
      linesize--;
      for (uint i = 0; i < linesize; i++) {
        types[i].range = sqrt(pl[i].x * pl[i].x + pl[i].y * pl[i].y);
        vx_ = pl[i].x - pl[i + 1].x;
        vy_ = pl[i].y - pl[i + 1].y;
        vz_ = pl[i].z - pl[i + 1].z;
        types[i].dista = vx_ * vx_ + vy_ * vy_ + vz_ * vz_;
      }
      types[linesize].range = sqrt(pl[linesize].x * pl[linesize].x +
                                   pl[linesize].y * pl[linesize].y);
      GiveFeature(pl, types);
    }
  } else {
    for (int i = 0; i < plsize; i++) {
      PointXYZIN added_pt;
      // cout<<"!!!!!!"<<i<<" "<<plsize<<endl;

      added_pt.normal_x = 0;
      added_pt.normal_y = 0;
      added_pt.normal_z = 0;
      added_pt.x = pl_orig.points[i].x;
      added_pt.y = pl_orig.points[i].y;
      added_pt.z = pl_orig.points[i].z;
      added_pt.intensity = pl_orig.points[i].intensity;
      added_pt.curvature = pl_orig.points[i].time / 1000.0;

      if (!given_offset_time_) {
        int layer = pl_orig.points[i].ring;
        double yaw_angle = atan2(added_pt.y, added_pt.x) * 57.2957;

        if (is_first[layer]) {
          // printf("layer: %d; is first: %d", layer, is_first[layer]);
          yaw_fp[layer] = yaw_angle;
          is_first[layer] = false;
          added_pt.curvature = 0.0;
          yaw_last[layer] = yaw_angle;
          time_last[layer] = added_pt.curvature;
          continue;
        }

        // compute offset time
        if (yaw_angle <= yaw_fp[layer]) {
          added_pt.curvature = (yaw_fp[layer] - yaw_angle) / omega_l;
        } else {
          added_pt.curvature = (yaw_fp[layer] - yaw_angle + 360.0) / omega_l;
        }

        if (added_pt.curvature < time_last[layer])
          added_pt.curvature += 360.0 / omega_l;

        // added_pt.curvature = pl_orig.points[i].t;

        yaw_last[layer] = yaw_angle;
        time_last[layer] = added_pt.curvature;
      }

      // if(i==(plsize-1))  printf("index: %d layer: %d, yaw: %lf, offset-time:
      // %lf, condition: %d\n", i, layer, yaw_angle, added_pt.curvature,
      // prints);
      if (i % point_filter_num_ == 0) {
        if (added_pt.x * added_pt.x + added_pt.y * added_pt.y +
                added_pt.z * added_pt.z >
            blind_sqr_) {
          pl_surf_.points.push_back(added_pt);
          // printf("time mode: %d time: %d \n", given_offset_time,
          // pl_orig.points[i].t);
        }
      }
    }
  }
  // pub_func(pl_surf, pub_full, msg->header.stamp);
  // pub_func(pl_surf, pub_surf, msg->header.stamp);
  // pub_func(pl_surf, pub_corn, msg->header.stamp);
}

void Preprocess::Pandar128Handler(
    const sensor_msgs::PointCloud2::ConstPtr &msg) {
  pl_surf_.clear();

  pcl::PointCloud<Pandar128_ros::Point> pl_orig;
  pcl::fromROSMsg(*msg, pl_orig);
  int plsize = pl_orig.points.size();
  pl_surf_.reserve(plsize);

  double time_head = pl_orig.points[0].timestamp;
  for (int i = 0; i < plsize; i++) {
    PointXYZIN added_pt;

    added_pt.normal_x = 0;
    added_pt.normal_y = 0;
    added_pt.normal_z = 0;
    added_pt.x = pl_orig.points[i].x;
    added_pt.y = pl_orig.points[i].y;
    added_pt.z = pl_orig.points[i].z;
    added_pt.intensity =
        static_cast<float>(pl_orig.points[i].intensity) / 255.0f;
    added_pt.curvature = (pl_orig.points[i].timestamp - time_head) * 1000.f;

    if (i % point_filter_num_ == 0) {
      if (added_pt.x * added_pt.x + added_pt.y * added_pt.y +
              added_pt.z * added_pt.z >
          blind_sqr_) {
        pl_surf_.points.push_back(added_pt);
        // printf("time mode: %d time: %d \n", given_offset_time,
        // pl_orig.points[i].t);
      }
    }
  }

  // define a lambda function for the comparison
  auto comparePoints = [](const PointXYZIN &a, const PointXYZIN &b) -> bool {
    return a.curvature < b.curvature;
  };

  // sort the points using the comparison function
  std::sort(pl_surf_.points.begin(), pl_surf_.points.end(), comparePoints);

  // cout << GREEN << "pl_surf.points[0].timestamp: " <<
  // pl_surf.points[0].curvature << RESET << endl; cout << GREEN <<
  // "pl_surf.points[1000].timestamp: " << pl_surf.points[1000].curvature <<
  // RESET << endl; cout << GREEN << "pl_surf.points[5000].timestamp: " <<
  // pl_surf.points[5000].curvature << RESET << endl; cout << GREEN <<
  // "pl_surf.points[10000].timestamp: " << pl_surf.points[10000].curvature <<
  // RESET << endl; cout << GREEN << "pl_surf.points[20000].timestamp: " <<
  // pl_surf.points[20000].curvature << RESET << endl; cout << GREEN <<
  // "pl_surf.points[30000].timestamp: " << pl_surf.points[30000].curvature <<
  // RESET << endl; cout << GREEN << "pl_surf.points[31000].timestamp: " <<
  // pl_surf.points[31000].curvature << RESET << endl;
}

void Preprocess::Xt32Handler(const sensor_msgs::PointCloud2::ConstPtr &msg) {
  pl_surf_.clear();
  pl_corn_.clear();
  pl_full_.clear();

  pcl::PointCloud<xt32_ros::Point> pl_orig;
  pcl::fromROSMsg(*msg, pl_orig);
  int plsize = pl_orig.points.size();
  pl_surf_.reserve(plsize);

  bool is_first[MAX_LINE_NUM];
  double yaw_fp[MAX_LINE_NUM] = {0};      // yaw of first scan point
  double omega_l = 3.61;                  // scan angular velocity
  float yaw_last[MAX_LINE_NUM] = {0.0};   // yaw of last scan point
  float time_last[MAX_LINE_NUM] = {0.0};  // last offset time

  if (pl_orig.points[plsize - 1].timestamp > 0) {
    given_offset_time_ = true;
  } else {
    given_offset_time_ = false;
    memset(is_first, true, sizeof(is_first));
    double yaw_first =
        atan2(pl_orig.points[0].y, pl_orig.points[0].x) * 57.29578;
    double yaw_end = yaw_first;
    int layer_first = pl_orig.points[0].ring;
    for (uint i = plsize - 1; i > 0; i--) {
      if (pl_orig.points[i].ring == layer_first) {
        yaw_end = atan2(pl_orig.points[i].y, pl_orig.points[i].x) * 57.29578;
        break;
      }
    }
  }

  double time_head = pl_orig.points[0].timestamp;

  if (feature_enabled_) {
    for (int i = 0; i < n_scans_; i++) {
      pl_buff_[i].clear();
      pl_buff_[i].reserve(plsize);
    }

    for (int i = 0; i < plsize; i++) {
      PointXYZIN added_pt;
      added_pt.normal_x = 0;
      added_pt.normal_y = 0;
      added_pt.normal_z = 0;
      int layer = pl_orig.points[i].ring;
      if (layer >= n_scans_) continue;
      added_pt.x = pl_orig.points[i].x;
      added_pt.y = pl_orig.points[i].y;
      added_pt.z = pl_orig.points[i].z;
      added_pt.intensity = pl_orig.points[i].intensity;
      added_pt.curvature = pl_orig.points[i].timestamp / 1000.0;  // units: ms

      if (!given_offset_time_) {
        double yaw_angle = atan2(added_pt.y, added_pt.x) * 57.2957;
        if (is_first[layer]) {
          // printf("layer: %d; is first: %d", layer, is_first[layer]);
          yaw_fp[layer] = yaw_angle;
          is_first[layer] = false;
          added_pt.curvature = 0.0;
          yaw_last[layer] = yaw_angle;
          time_last[layer] = added_pt.curvature;
          continue;
        }

        if (yaw_angle <= yaw_fp[layer]) {
          added_pt.curvature = (yaw_fp[layer] - yaw_angle) / omega_l;
        } else {
          added_pt.curvature = (yaw_fp[layer] - yaw_angle + 360.0) / omega_l;
        }

        if (added_pt.curvature < time_last[layer])
          added_pt.curvature += 360.0 / omega_l;

        yaw_last[layer] = yaw_angle;
        time_last[layer] = added_pt.curvature;
      }

      pl_buff_[layer].points.push_back(added_pt);
    }

    for (int j = 0; j < n_scans_; j++) {
      PointCloudXYZIN &pl = pl_buff_[j];
      int linesize = pl.size();
      if (linesize < 2) continue;
      vector<orgtype> &types = typess_[j];
      types.clear();
      types.resize(linesize);
      linesize--;
      for (uint i = 0; i < linesize; i++) {
        types[i].range = sqrt(pl[i].x * pl[i].x + pl[i].y * pl[i].y);
        vx_ = pl[i].x - pl[i + 1].x;
        vy_ = pl[i].y - pl[i + 1].y;
        vz_ = pl[i].z - pl[i + 1].z;
        types[i].dista = vx_ * vx_ + vy_ * vy_ + vz_ * vz_;
      }
      types[linesize].range = sqrt(pl[linesize].x * pl[linesize].x +
                                   pl[linesize].y * pl[linesize].y);
      GiveFeature(pl, types);
    }
  } else {
    for (int i = 0; i < plsize; i++) {
      PointXYZIN added_pt;
      // cout<<"!!!!!!"<<i<<" "<<plsize<<endl;

      added_pt.normal_x = 0;
      added_pt.normal_y = 0;
      added_pt.normal_z = 0;
      added_pt.x = pl_orig.points[i].x;
      added_pt.y = pl_orig.points[i].y;
      added_pt.z = pl_orig.points[i].z;
      added_pt.intensity = pl_orig.points[i].intensity;
      added_pt.curvature = (pl_orig.points[i].timestamp - time_head) * 1000.f;

      // printf("added_pt.curvature: %lf %lf \n", added_pt.curvature,
      // pl_orig.points[i].timestamp);

      // if(i==(plsize-1))  printf("index: %d layer: %d, yaw: %lf, offset-time:
      // %lf, condition: %d\n", i, layer, yaw_angle, added_pt.curvature,
      // prints);
      if (i % point_filter_num_ == 0) {
        if (added_pt.x * added_pt.x + added_pt.y * added_pt.y +
                added_pt.z * added_pt.z >
            blind_sqr_) {
          pl_surf_.points.push_back(added_pt);
          // printf("time mode: %d time: %d \n", given_offset_time,
          // pl_orig.points[i].t);
        }
      }
    }
  }
  // pub_func(pl_surf, pub_full, msg->header.stamp);
  // pub_func(pl_surf, pub_surf, msg->header.stamp);
  // pub_func(pl_surf, pub_corn, msg->header.stamp);
}

void Preprocess::RobosenseHandler(
    const sensor_msgs::PointCloud2::ConstPtr &msg) {
  pl_surf_.clear();

  pcl::PointCloud<robosense_ros::Point> pl_orig;
  pcl::fromROSMsg(*msg, pl_orig);
  int plsize = pl_orig.size();
  pl_surf_.reserve(plsize);

  double time_head = pl_orig.points[0].timestamp;
  for (int i = 0; i < plsize; ++i) {
    if (i % point_filter_num_ != 0) continue;

    const auto &pt = pl_orig.points[i];
    const double x = pt.x, y = pt.y, z = pt.z;
    const double dist_sqr = x * x + y * y + z * z;
    const bool is_valid = (dist_sqr >= blind_sqr_) && !std::isnan(x) &&
                          !std::isnan(y) && !std::isnan(z);
    if (!is_valid) continue;

    PointXYZIN added_pt;
    added_pt.normal_x = 0;
    added_pt.normal_y = 0;
    added_pt.normal_z = 0;
    added_pt.x = pt.x;
    added_pt.y = pt.y;
    added_pt.z = pt.z;
    added_pt.intensity = pt.intensity;
    added_pt.curvature = (pt.timestamp - time_head) * 1000.0;
    pl_surf_.points.push_back(added_pt);
  }
  std::sort(pl_surf_.points.begin(), pl_surf_.points.end(),
            [](const PointXYZIN &a, const PointXYZIN &b) {
              return a.curvature < b.curvature;
            });
}

void Preprocess::GiveFeature(pcl::PointCloud<PointXYZIN> &pl,
                              vector<orgtype> &types) {
  int plsize = pl.size();
  int plsize2;
  if (plsize == 0) {
    printf("something wrong\n");
    return;
  }
  uint head = 0;

  while (types[head].range < blind_sqr_) {
    head++;
  }

  // Surf
  plsize2 = (plsize > group_size_) ? (plsize - group_size_) : 0;

  Eigen::Vector3d curr_direct(Eigen::Vector3d::Zero());
  Eigen::Vector3d last_direct(Eigen::Vector3d::Zero());

  uint i_nex = 0, i2;
  uint last_i = 0;
  uint last_i_nex = 0;
  int last_state = 0;
  int plane_type;

  for (uint i = head; i < plsize2; i++) {
    if (types[i].range < blind_sqr_) {
      continue;
    }

    i2 = i;

    plane_type = PlaneJudge(pl, types, i, i_nex, curr_direct);

    if (plane_type == 1) {
      for (uint j = i; j <= i_nex; j++) {
        if (j != i && j != i_nex) {
          types[j].ftype = Real_Plane;
        } else {
          types[j].ftype = Poss_Plane;
        }
      }

      // if(last_state==1 && fabs(last_direct.sum())>0.5)
      if (last_state == 1 && last_direct.norm() > 0.1) {
        double mod = last_direct.transpose() * curr_direct;
        if (mod > -0.707 && mod < 0.707) {
          types[i].ftype = Edge_Plane;
        } else {
          types[i].ftype = Real_Plane;
        }
      }

      i = i_nex - 1;
      last_state = 1;
    } else  // if(plane_type == 2)
    {
      i = i_nex;
      last_state = 0;
    }
    // else if(plane_type == 0)
    // {
    //   if(last_state == 1)
    //   {
    //     uint i_nex_tem;
    //     uint j;
    //     for(j=last_i+1; j<=last_i_nex; j++)
    //     {
    //       uint i_nex_tem2 = i_nex_tem;
    //       Eigen::Vector3d curr_direct2;

    //       uint ttem = plane_judge(pl, types, j, i_nex_tem, curr_direct2);

    //       if(ttem != 1)
    //       {
    //         i_nex_tem = i_nex_tem2;
    //         break;
    //       }
    //       curr_direct = curr_direct2;
    //     }

    //     if(j == last_i+1)
    //     {
    //       last_state = 0;
    //     }
    //     else
    //     {
    //       for(uint k=last_i_nex; k<=i_nex_tem; k++)
    //       {
    //         if(k != i_nex_tem)
    //         {
    //           types[k].ftype = Real_Plane;
    //         }
    //         else
    //         {
    //           types[k].ftype = Poss_Plane;
    //         }
    //       }
    //       i = i_nex_tem-1;
    //       i_nex = i_nex_tem;
    //       i2 = j-1;
    //       last_state = 1;
    //     }

    //   }
    // }

    last_i = i2;
    last_i_nex = i_nex;
    last_direct = curr_direct;
  }

  plsize2 = plsize > 3 ? plsize - 3 : 0;
  for (uint i = head + 3; i < plsize2; i++) {
    if (types[i].range < blind_sqr_ || types[i].ftype >= Real_Plane) {
      continue;
    }

    if (types[i - 1].dista < 1e-16 || types[i].dista < 1e-16) {
      continue;
    }

    Eigen::Vector3d vec_a(pl[i].x, pl[i].y, pl[i].z);
    Eigen::Vector3d vecs[2];

    for (int j = 0; j < 2; j++) {
      int m = -1;
      if (j == 1) {
        m = 1;
      }

      if (types[i + m].range < blind_sqr_) {
        if (types[i].range > inf_bound_) {
          types[i].edj[j] = Nr_inf;
        } else {
          types[i].edj[j] = Nr_blind;
        }
        continue;
      }

      vecs[j] = Eigen::Vector3d(pl[i + m].x, pl[i + m].y, pl[i + m].z);
      vecs[j] = vecs[j] - vec_a;

      types[i].angle[j] = vec_a.dot(vecs[j]) / vec_a.norm() / vecs[j].norm();
      if (types[i].angle[j] < jump_up_limit_) {
        types[i].edj[j] = Nr_180;
      } else if (types[i].angle[j] > jump_down_limit_) {
        types[i].edj[j] = Nr_zero;
      }
    }

    types[i].intersect =
        vecs[Prev].dot(vecs[Next]) / vecs[Prev].norm() / vecs[Next].norm();
    if (types[i].edj[Prev] == Nr_nor && types[i].edj[Next] == Nr_zero &&
        types[i].dista > 0.0225 && types[i].dista > 4 * types[i - 1].dista) {
      if (types[i].intersect > cos160_) {
        if (EdgeJumpJudge(pl, types, i, Prev)) {
          types[i].ftype = Edge_Jump;
        }
      }
    } else if (types[i].edj[Prev] == Nr_zero && types[i].edj[Next] == Nr_nor &&
               types[i - 1].dista > 0.0225 &&
               types[i - 1].dista > 4 * types[i].dista) {
      if (types[i].intersect > cos160_) {
        if (EdgeJumpJudge(pl, types, i, Next)) {
          types[i].ftype = Edge_Jump;
        }
      }
    } else if (types[i].edj[Prev] == Nr_nor && types[i].edj[Next] == Nr_inf) {
      if (EdgeJumpJudge(pl, types, i, Prev)) {
        types[i].ftype = Edge_Jump;
      }
    } else if (types[i].edj[Prev] == Nr_inf && types[i].edj[Next] == Nr_nor) {
      if (EdgeJumpJudge(pl, types, i, Next)) {
        types[i].ftype = Edge_Jump;
      }
    } else if (types[i].edj[Prev] > Nr_nor && types[i].edj[Next] > Nr_nor) {
      if (types[i].ftype == Nor) {
        types[i].ftype = Wire;
      }
    }
  }

  plsize2 = plsize - 1;
  double ratio;
  for (uint i = head + 1; i < plsize2; i++) {
    if (types[i].range < blind_sqr_ || types[i - 1].range < blind_sqr_ ||
        types[i + 1].range < blind_sqr_) {
      continue;
    }

    if (types[i - 1].dista < 1e-8 || types[i].dista < 1e-8) {
      continue;
    }

    if (types[i].ftype == Nor) {
      if (types[i - 1].dista > types[i].dista) {
        ratio = types[i - 1].dista / types[i].dista;
      } else {
        ratio = types[i].dista / types[i - 1].dista;
      }

      if (types[i].intersect < smallp_intersect_ && ratio < smallp_ratio_) {
        if (types[i - 1].ftype == Nor) {
          types[i - 1].ftype = Real_Plane;
        }
        if (types[i + 1].ftype == Nor) {
          types[i + 1].ftype = Real_Plane;
        }
        types[i].ftype = Real_Plane;
      }
    }
  }

  int last_surface = -1;
  for (uint j = head; j < plsize; j++) {
    if (types[j].ftype == Poss_Plane || types[j].ftype == Real_Plane) {
      if (last_surface == -1) {
        last_surface = j;
      }

      if (j == uint(last_surface + point_filter_num_ - 1)) {
        PointXYZIN ap;
        ap.x = pl[j].x;
        ap.y = pl[j].y;
        ap.z = pl[j].z;
        ap.curvature = pl[j].curvature;
        pl_surf_.push_back(ap);

        last_surface = -1;
      }
    } else {
      if (types[j].ftype == Edge_Jump || types[j].ftype == Edge_Plane) {
        pl_corn_.push_back(pl[j]);
      }
      if (last_surface != -1) {
        PointXYZIN ap;
        for (uint k = last_surface; k < j; k++) {
          ap.x += pl[k].x;
          ap.y += pl[k].y;
          ap.z += pl[k].z;
          ap.curvature += pl[k].curvature;
        }
        ap.x /= (j - last_surface);
        ap.y /= (j - last_surface);
        ap.z /= (j - last_surface);
        ap.curvature /= (j - last_surface);
        pl_surf_.push_back(ap);
      }
      last_surface = -1;
    }
  }
}

void Preprocess::PubFunc(PointCloudXYZIN &pl, const ros::Time &ct) {
  pl.height = 1;
  pl.width = pl.size();
  sensor_msgs::PointCloud2 output;
  pcl::toROSMsg(pl, output);
  output.header.frame_id = "livox";
  output.header.stamp = ct;
}

int Preprocess::PlaneJudge(const PointCloudXYZIN &pl, vector<orgtype> &types,
                            uint i_cur, uint &i_nex,
                            Eigen::Vector3d &curr_direct) {
  double group_dis = dis_a_ * types[i_cur].range + dis_b_;
  group_dis = group_dis * group_dis;
  // i_nex = i_cur;

  double two_dis;
  vector<double> disarr;
  disarr.reserve(20);

  for (i_nex = i_cur; i_nex < i_cur + group_size_; i_nex++) {
    if (types[i_nex].range < blind_sqr_) {
      curr_direct.setZero();
      return 2;
    }
    disarr.push_back(types[i_nex].dista);
  }

  for (;;) {
    if ((i_cur >= pl.size()) || (i_nex >= pl.size())) break;

    if (types[i_nex].range < blind_sqr_) {
      curr_direct.setZero();
      return 2;
    }
    vx_ = pl[i_nex].x - pl[i_cur].x;
    vy_ = pl[i_nex].y - pl[i_cur].y;
    vz_ = pl[i_nex].z - pl[i_cur].z;
    two_dis = vx_ * vx_ + vy_ * vy_ + vz_ * vz_;
    if (two_dis >= group_dis) {
      break;
    }
    disarr.push_back(types[i_nex].dista);
    i_nex++;
  }

  double leng_wid = 0;
  double v1[3], v2[3];
  for (uint j = i_cur + 1; j < i_nex; j++) {
    if ((j >= pl.size()) || (i_cur >= pl.size())) break;
    v1[0] = pl[j].x - pl[i_cur].x;
    v1[1] = pl[j].y - pl[i_cur].y;
    v1[2] = pl[j].z - pl[i_cur].z;

    v2[0] = v1[1] * vz_ - vy_ * v1[2];
    v2[1] = v1[2] * vx_ - v1[0] * vz_;
    v2[2] = v1[0] * vy_ - vx_ * v1[1];

    double lw = v2[0] * v2[0] + v2[1] * v2[1] + v2[2] * v2[2];
    if (lw > leng_wid) {
      leng_wid = lw;
    }
  }

  if ((two_dis * two_dis / leng_wid) < p2l_ratio_) {
    curr_direct.setZero();
    return 0;
  }

  uint disarrsize = disarr.size();
  for (uint j = 0; j < disarrsize - 1; j++) {
    for (uint k = j + 1; k < disarrsize; k++) {
      if (disarr[j] < disarr[k]) {
        leng_wid = disarr[j];
        disarr[j] = disarr[k];
        disarr[k] = leng_wid;
      }
    }
  }

  if (disarr[disarr.size() - 2] < 1e-16) {
    curr_direct.setZero();
    return 0;
  }

  if (lidar_type_ == AVIA) {
    double dismax_mid = disarr[0] / disarr[disarrsize / 2];
    double dismid_min = disarr[disarrsize / 2] / disarr[disarrsize - 2];

    if (dismax_mid >= limit_maxmid_ || dismid_min >= limit_midmin_) {
      curr_direct.setZero();
      return 0;
    }
  } else {
    double dismax_min = disarr[0] / disarr[disarrsize - 2];
    if (dismax_min >= limit_maxmin_) {
      curr_direct.setZero();
      return 0;
    }
  }

  curr_direct << vx_, vy_, vz_;
  curr_direct.normalize();
  return 1;
}

bool Preprocess::EdgeJumpJudge(const PointCloudXYZIN &pl,
                                 vector<orgtype> &types, uint i,
                                 Surround nor_dir) {
  if (nor_dir == 0) {
    if (types[i - 1].range < blind_sqr_ || types[i - 2].range < blind_sqr_) {
      return false;
    }
  } else if (nor_dir == 1) {
    if (types[i + 1].range < blind_sqr_ || types[i + 2].range < blind_sqr_) {
      return false;
    }
  }
  double d1 = types[i + nor_dir - 1].dista;
  double d2 = types[i + 3 * nor_dir - 2].dista;
  double d;

  if (d1 < d2) {
    d = d1;
    d1 = d2;
    d2 = d;
  }

  d1 = sqrt(d1);
  d2 = sqrt(d2);

  if (d1 > edge_a_ * d2 || (d1 - d2) > edge_b_) {
    return false;
  }

  return true;
}