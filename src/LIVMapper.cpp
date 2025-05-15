/*
This file is part of FAST-LIVO2: Fast, Direct LiDAR-Inertial-Visual Odometry.

Developer: Chunran Zheng <zhengcr@connect.hku.hk>

For commercial use, please contact me at <zhengcr@connect.hku.hk> or
Prof. Fu Zhang at <fuzhang@hku.hk>.

This file is subject to the terms and conditions outlined in the 'LICENSE' file,
which is included as part of this source code package.
*/

#include "LIVMapper.h"

#include <glog/logging.h>
#include <yaml-cpp/yaml.h>

LIVMapper::LIVMapper(ros::NodeHandle &nh)
    : ext_t_(0, 0, 0), ext_r_(M3D::Identity()) {
  extrin_t_.assign(3, 0.0);
  extrin_r_.assign(9, 0.0);
  cameraextrin_t_.assign(3, 0.0);
  cameraextrin_r_.assign(9, 0.0);

  p_pre_.reset(new Preprocess());
  p_imu_.reset(new ImuProcess());

  readParameters(nh);
  VoxelMapConfig voxel_config;
  loadVoxelConfig(nh, voxel_config);

  visual_sub_map_.reset(new PointCloudXYZIN());
  feats_undistort_.reset(new PointCloudXYZIN());
  feats_down_body_.reset(new PointCloudXYZIN());
  feats_down_world_.reset(new PointCloudXYZIN());
  pcl_w_wait_pub_.reset(new PointCloudXYZIN());
  pcl_wait_pub_.reset(new PointCloudXYZIN());
  pcl_wait_save_.reset(new PointCloudXYZRGB());
  pcl_wait_save_intensity_.reset(new PointCloudXYZIN());
  voxelmap_manager_.reset(new VoxelMapManager(voxel_config, voxel_map_));
  vio_manager_.reset(new VIOManager());
  root_dir_ = ROOT_DIR;
  initializeFiles();
  initializeComponents();
  path_.header.stamp = ros::Time::now();
  path_.header.frame_id = "camera_init";
}

LIVMapper::~LIVMapper() {}

void LIVMapper::readParameters(ros::NodeHandle &nh) {
  nh.param<string>("common/lid_topic", lid_topic_, "/livox/lidar");
  nh.param<string>("common/imu_topic", imu_topic_, "/livox/imu");
  nh.param<bool>("common/ros_driver_bug_fix", ros_driver_fix_en_, false);
  nh.param<int>("common/img_en", img_en_, 1);
  nh.param<int>("common/lidar_en", lidar_en_, 1);
  nh.param<string>("common/img_topic", img_topic_, "/left_camera/image");

  nh.param<bool>("vio/normal_en", normal_en_, true);
  nh.param<bool>("vio/inverse_composition_en", inverse_composition_en_, false);
  nh.param<int>("vio/max_iterations", max_iterations_, 5);
  nh.param<double>("vio/img_point_cov", img_point_cov_, 100);
  nh.param<bool>("vio/raycast_en", raycast_en_, false);
  nh.param<bool>("vio/exposure_estimate_en", exposure_estimate_en_, true);
  nh.param<double>("vio/inv_expo_cov", inv_expo_cov_, 0.2);
  nh.param<int>("vio/grid_size", grid_size_, 5);
  nh.param<int>("vio/grid_n_height", grid_n_height_, 17);
  nh.param<int>("vio/patch_pyrimid_level", patch_pyrimid_level_, 3);
  nh.param<int>("vio/patch_size", patch_size_, 8);
  nh.param<double>("vio/outlier_threshold", outlier_threshold_, 1000);

  nh.param<double>("time_offset/exposure_time_init", exposure_time_init_, 0.0);
  nh.param<double>("time_offset/img_time_offset", img_time_offset_, 0.0);
  nh.param<double>("time_offset/imu_time_offset", imu_time_offset_, 0.0);
  nh.param<double>("time_offset/lidar_time_offset", lidar_time_offset_, 0.0);
  nh.param<bool>("uav/imu_rate_odom", imu_prop_enable_, false);
  nh.param<bool>("uav/gravity_align_en", gravity_align_en_, false);

  nh.param<string>("evo/seq_name", seq_name_, "01");
  nh.param<bool>("evo/pose_output_en", pose_output_en_, false);
  nh.param<double>("imu/gyr_cov", gyr_cov_, 1.0);
  nh.param<double>("imu/acc_cov", acc_cov_, 1.0);
  nh.param<int>("imu/imu_int_frame", imu_int_frame_, 3);
  nh.param<bool>("imu/imu_en", imu_en_, false);
  nh.param<bool>("imu/gravity_est_en", gravity_est_en_, true);
  nh.param<bool>("imu/ba_bg_est_en", ba_bg_est_en_, true);

  nh.param<double>("preprocess/blind", p_pre_->blind, 0.01);
  nh.param<double>("preprocess/filter_size_surf", filter_size_surf_min_, 0.5);
  nh.param<int>("preprocess/lidar_type", p_pre_->lidar_type, AVIA);
  nh.param<int>("preprocess/scan_line", p_pre_->N_SCANS, 6);
  nh.param<int>("preprocess/point_filter_num", p_pre_->point_filter_num, 3);
  nh.param<bool>("preprocess/feature_extract_enabled", p_pre_->feature_enabled,
                 false);

  nh.param<int>("pcd_save/interval", pcd_save_interval_, -1);
  nh.param<bool>("pcd_save/pcd_save_en", pcd_save_en_, false);
  nh.param<bool>("pcd_save/colmap_output_en", colmap_output_en_, false);
  nh.param<double>("pcd_save/filter_size_pcd", filter_size_pcd_, 0.5);
  nh.param<vector<double>>("extrin_calib/extrinsic_T", extrin_t_,
                           vector<double>());
  nh.param<vector<double>>("extrin_calib/extrinsic_R", extrin_r_,
                           vector<double>());
  nh.param<vector<double>>("extrin_calib/Pcl", cameraextrin_t_,
                           vector<double>());
  nh.param<vector<double>>("extrin_calib/Rcl", cameraextrin_r_,
                           vector<double>());
  nh.param<double>("debug/plot_time", plot_time_, -10);
  nh.param<int>("debug/frame_cnt", frame_cnt_, 6);

  nh.param<double>("publish/blind_rgb_points", blind_rgb_points_, 0.01);
  nh.param<int>("publish/pub_scan_num", pub_scan_num_, 1);
  nh.param<bool>("publish/pub_effect_point_en", pub_effect_point_en_, false);
  nh.param<bool>("publish/dense_map_en", dense_map_en, false);

  p_pre_->blind_sqr = p_pre_->blind * p_pre_->blind;
}

void LIVMapper::initializeComponents() {
  downSize_filter_surf_.setLeafSize(
      filter_size_surf_min_, filter_size_surf_min_, filter_size_surf_min_);
  ext_t_ << VEC_FROM_ARRAY(extrin_t_);
  ext_r_ << MAT_FROM_ARRAY(extrin_r_);

  voxelmap_manager_->extT_ << VEC_FROM_ARRAY(extrin_t_);
  voxelmap_manager_->extR_ << MAT_FROM_ARRAY(extrin_r_);
  // 载入相机参数
  // if (!vk::camera_loader::loadFromRosNs("laserMapping", vio_manager_->cam_))
  //   throw std::runtime_error("Camera model not correctly specified.");
  std::string file_name = std::string(ROOT_DIR) + "config/camera_pinhole.yaml";
  YAML::Node camera_config = YAML::LoadFile(file_name);
  if (!vk::camera_loader::loadFromYaml(camera_config, vio_manager_->cam_))
    throw std::runtime_error("Camera model not correctly specified.");
  // 视觉里程计初始化
  vio_manager_->grid_size_ = grid_size_;
  vio_manager_->patch_size_ = patch_size_;
  vio_manager_->outlier_threshold_ = outlier_threshold_;
  vio_manager_->SetImuToLidarExtrinsic(ext_t_, ext_r_);
  vio_manager_->SetLidarToCameraExtrinsic(cameraextrin_r_, cameraextrin_t_);
  // vio_manager的state_和LIVMapper的state_和state_propagat_是一致的，但是voxel_map的需要手动同步
  vio_manager_->state_ = &state_;
  vio_manager_->state_propagat_ = &state_propagat_;
  vio_manager_->max_iterations_ = max_iterations_;
  vio_manager_->img_point_cov_ = img_point_cov_;
  vio_manager_->normal_en_ = normal_en_;
  vio_manager_->inverse_composition_en_ = inverse_composition_en_;
  vio_manager_->raycast_en_ = raycast_en_;
  vio_manager_->grid_n_width_ = grid_n_width_;
  vio_manager_->grid_n_height_ = grid_n_height_;
  vio_manager_->patch_pyrimid_level_ = patch_pyrimid_level_;
  vio_manager_->exposure_estimate_en_ = exposure_estimate_en_;
  vio_manager_->colmap_output_en_ = colmap_output_en_;
  vio_manager_->InitializeVIO();

  p_imu_->set_extrinsic(ext_t_, ext_r_);
  p_imu_->set_gyr_cov_scale(V3D(gyr_cov_, gyr_cov_, gyr_cov_));
  p_imu_->set_acc_cov_scale(V3D(acc_cov_, acc_cov_, acc_cov_));
  p_imu_->set_inv_expo_cov(inv_expo_cov_);
  p_imu_->set_gyr_bias_cov(V3D(0.0001, 0.0001, 0.0001));
  p_imu_->set_acc_bias_cov(V3D(0.0001, 0.0001, 0.0001));
  p_imu_->set_imu_init_frame_num(imu_int_frame_);

  if (!imu_en_) p_imu_->disable_imu();
  if (!gravity_est_en_) p_imu_->disable_gravity_est();
  if (!ba_bg_est_en_) p_imu_->disable_bias_est();
  if (!exposure_estimate_en_) p_imu_->disable_exposure_est();

  slam_mode_ = (img_en_ && lidar_en_) ? LIVO : imu_en_ ? ONLY_LIO : ONLY_LO;
}

void LIVMapper::initializeFiles() {
  if (pcd_save_en_ && colmap_output_en_) {
    const std::string folderPath =
        std::string(ROOT_DIR) + "/scripts/colmap_output.sh";

    std::string chmodCommand = "chmod +x " + folderPath;

    int chmodRet = system(chmodCommand.c_str());
    if (chmodRet != 0) {
      std::cerr << "Failed to set execute permissions for the script."
                << std::endl;
      return;
    }

    int executionRet = system(folderPath.c_str());
    if (executionRet != 0) {
      std::cerr << "Failed to execute the script." << std::endl;
      return;
    }
  }
  if (colmap_output_en_)
    fout_points_.open(
        std::string(ROOT_DIR) + "Log/Colmap/sparse/0/points3D.txt",
        std::ios::out);
  if (pcd_save_interval_ > 0)
    fout_pcd_pos_.open(std::string(ROOT_DIR) + "Log/PCD/scans_pos.json",
                       std::ios::out);
  fout_pre_.open(DEBUG_FILE_DIR("mat_pre.txt"), std::ios::out);
  fout_out_.open(DEBUG_FILE_DIR("mat_out.txt"), std::ios::out);
}

void LIVMapper::initializeSubscribersAndPublishers(
    ros::NodeHandle &nh, image_transport::ImageTransport &it) {
  sub_pcl_ =
      p_pre_->lidar_type == AVIA
          ? nh.subscribe(lid_topic_, 200000, &LIVMapper::livox_pcl_cbk, this)
          : nh.subscribe(lid_topic_, 200000, &LIVMapper::standard_pcl_cbk,
                         this);
  sub_imu_ = nh.subscribe(imu_topic_, 200000, &LIVMapper::imu_cbk, this);
  sub_img_ = nh.subscribe(img_topic_, 200000, &LIVMapper::img_cbk, this);

  pubLaser_cloud_full_res_ =
      nh.advertise<sensor_msgs::PointCloud2>("/cloud_registered", 100);
  pub_normal_ = nh.advertise<visualization_msgs::MarkerArray>(
      "visualization_marker", 100);
  pub_sub_visual_map_ = nh.advertise<sensor_msgs::PointCloud2>(
      "/cloud_visual_sub_map_before", 100);
  pub_laser_cloud_effect_ =
      nh.advertise<sensor_msgs::PointCloud2>("/cloud_effected", 100);
  pub_laser_cloud_map_ =
      nh.advertise<sensor_msgs::PointCloud2>("/Laser_map", 100);
  pub_odom_aft_mapped_ =
      nh.advertise<nav_msgs::Odometry>("/aft_mapped_to_init", 10);
  pub_path_ = nh.advertise<nav_msgs::Path>("/path", 10);
  plane_pub_ = nh.advertise<visualization_msgs::Marker>("/planner_normal", 1);
  voxel_pub_ = nh.advertise<visualization_msgs::MarkerArray>("/voxels", 1);
  pub_laser_cloud_dyn_ =
      nh.advertise<sensor_msgs::PointCloud2>("/dyn_obj", 100);
  pub_laser_cloud_dyn_rmed_ =
      nh.advertise<sensor_msgs::PointCloud2>("/dyn_obj_removed", 100);
  pub_laser_cloud_dyn_dbg_ =
      nh.advertise<sensor_msgs::PointCloud2>("/dyn_obj_dbg_hist", 100);
  mavros_pose_publisher_ =
      nh.advertise<geometry_msgs::PoseStamped>("/mavros/vision_pose/pose", 10);
  pub_image_ = it.advertise("/rgb_img", 1);
  pub_imu_prop_odom_ =
      nh.advertise<nav_msgs::Odometry>("/LIVO2/imu_propagate", 10000);
  imu_prop_timer_ =
      nh.createTimer(ros::Duration(0.004), &LIVMapper::imu_prop_callback, this);
  voxelmap_manager_->voxel_map_pub_ =
      nh.advertise<visualization_msgs::MarkerArray>("/planes", 10000);
}

void LIVMapper::handleFirstFrame() {
  if (!first_frame_finished_) {
    first_lidar_time_ = Lidar_measures_.last_lio_update_time;
    p_imu_->first_lidar_time_ = first_lidar_time_;  // Only for IMU data log
    first_frame_finished_ = true;
    cout << "FIRST LIDAR FRAME!" << endl;
  }
}

void LIVMapper::gravityAlignment() {
  if (!p_imu_->imu_need_init_ && !gravity_align_finished_) {
    std::cout << "Gravity Alignment Starts" << std::endl;
    V3D ez(0, 0, -1), gz(state_.gravity);
    Quaterniond G_q_I0 = Quaterniond::FromTwoVectors(gz, ez);
    M3D G_R_I0 = G_q_I0.toRotationMatrix();

    state_.pos_end = G_R_I0 * state_.pos_end;
    state_.rot_end = G_R_I0 * state_.rot_end;
    state_.vel_end = G_R_I0 * state_.vel_end;
    state_.gravity = G_R_I0 * state_.gravity;
    gravity_align_finished_ = true;
    std::cout << "Gravity Alignment Finished" << std::endl;
  }
}

void LIVMapper::processImu() {
  p_imu_->Process2(Lidar_measures_, state_, feats_undistort_);

  if (gravity_align_en_) gravityAlignment();

  state_propagat_ = state_;
  voxelmap_manager_->state_ = state_;
  voxelmap_manager_->feats_undistort_ = feats_undistort_;
}

void LIVMapper::stateEstimationAndMapping() {
  switch (Lidar_measures_.lio_vio_flg) {
    case VIO:
      handleVIO();
      break;
    case LIO:
    case LO:
      handleLIO();
      break;
  }
}

void LIVMapper::handleVIO() {
  euler_cur_ = RotMtoEuler(state_.rot_end);
  fout_pre_ << std::setw(20)
            << Lidar_measures_.last_lio_update_time - first_lidar_time_ << " "
            << euler_cur_.transpose() * 57.3 << " "
            << state_.pos_end.transpose() << " " << state_.vel_end.transpose()
            << " " << state_.bias_g.transpose() << " "
            << state_.bias_a.transpose() << " "
            << V3D(state_.inv_expo_time, 0, 0).transpose() << std::endl;

  if (pcl_w_wait_pub_->empty() || (pcl_w_wait_pub_ == nullptr)) {
    std::cout << "[ VIO ] No point!!!" << std::endl;
    return;
  }

  std::cout << "[ VIO ] Raw feature num: " << pcl_w_wait_pub_->points.size()
            << std::endl;

  if (fabs((Lidar_measures_.last_lio_update_time - first_lidar_time_) -
           plot_time_) < (frame_cnt_ / 2 * 0.1)) {
    vio_manager_->plot_flag_ = true;
  } else {
    // 实际上只走这个分支
    vio_manager_->plot_flag_ = false;
  }

  vio_manager_->ProcessFrame(
      Lidar_measures_.measures.back().img, pv_list_,
      voxelmap_manager_->voxel_map_,
      Lidar_measures_.last_lio_update_time - first_lidar_time_);

  if (imu_prop_enable_) {
    ekf_finish_once_ = true;
    latest_ekf_state_ = state_;
    latest_ekf_time_ = Lidar_measures_.last_lio_update_time;
    state_update_flg_ = true;
  }

  // int size_sub_map = vio_manager->visual_sub_map_cur.size();
  // visual_sub_map->reserve(size_sub_map);
  // for (int i = 0; i < size_sub_map; i++)
  // {
  //   PointType temp_map;
  //   temp_map.x = vio_manager->visual_sub_map_cur[i]->pos_[0];
  //   temp_map.y = vio_manager->visual_sub_map_cur[i]->pos_[1];
  //   temp_map.z = vio_manager->visual_sub_map_cur[i]->pos_[2];
  //   temp_map.intensity = 0.;
  //   visual_sub_map->push_back(temp_map);
  // }

  publish_frame_world(pubLaser_cloud_full_res_, vio_manager_);
  publish_img_rgb(pub_image_, vio_manager_);

  euler_cur_ = RotMtoEuler(state_.rot_end);
  fout_out_ << std::setw(20)
            << Lidar_measures_.last_lio_update_time - first_lidar_time_ << " "
            << euler_cur_.transpose() * 57.3 << " "
            << state_.pos_end.transpose() << " " << state_.vel_end.transpose()
            << " " << state_.bias_g.transpose() << " "
            << state_.bias_a.transpose() << " "
            << V3D(state_.inv_expo_time, 0, 0).transpose() << " "
            << feats_undistort_->points.size() << std::endl;
}

void LIVMapper::handleLIO() {
  euler_cur_ = RotMtoEuler(state_.rot_end);
  fout_pre_ << setw(20)
            << Lidar_measures_.last_lio_update_time - first_lidar_time_ << " "
            << euler_cur_.transpose() * 57.3 << " "
            << state_.pos_end.transpose() << " " << state_.vel_end.transpose()
            << " " << state_.bias_g.transpose() << " "
            << state_.bias_a.transpose() << " "
            << V3D(state_.inv_expo_time, 0, 0).transpose() << endl;

  if (feats_undistort_->empty() || (feats_undistort_ == nullptr)) {
    std::cout << "[ LIO ]: No point!!!" << std::endl;
    return;
  }

  double t0 = omp_get_wtime();

  downSize_filter_surf_.setInputCloud(feats_undistort_);
  downSize_filter_surf_.filter(*feats_down_body_);

  double t_down = omp_get_wtime();

  feats_down_size_ = feats_down_body_->points.size();
  voxelmap_manager_->feats_down_body_ = feats_down_body_;
  transformLidar(state_.rot_end, state_.pos_end, feats_down_body_,
                 feats_down_world_);
  voxelmap_manager_->feats_down_world_ = feats_down_world_;
  voxelmap_manager_->feats_down_size_ = feats_down_size_;

  if (!lidar_map_inited_) {
    // 第一帧，建立VoxelMap
    lidar_map_inited_ = true;
    voxelmap_manager_->BuildVoxelMap();
  }

  double t1 = omp_get_wtime();
  // 位姿估计
  voxelmap_manager_->StateEstimation(state_propagat_);
  state_ = voxelmap_manager_->state_;
  pv_list_ = voxelmap_manager_->pv_list_;

  double t2 = omp_get_wtime();

  if (imu_prop_enable_) {
    ekf_finish_once_ = true;
    latest_ekf_state_ = state_;
    latest_ekf_time_ = Lidar_measures_.last_lio_update_time;
    state_update_flg_ = true;
  }

  if (pose_output_en_) {
    static bool pos_opend = false;
    static int ocount = 0;
    std::ofstream outFile, evoFile;
    if (!pos_opend) {
      evoFile.open(std::string(ROOT_DIR) + "Log/result/" + seq_name_ + ".txt",
                   std::ios::out);
      pos_opend = true;
      if (!evoFile.is_open()) ROS_ERROR("open fail\n");
    } else {
      evoFile.open(std::string(ROOT_DIR) + "Log/result/" + seq_name_ + ".txt",
                   std::ios::app);
      if (!evoFile.is_open()) ROS_ERROR("open fail\n");
    }
    Eigen::Matrix4d outT;
    Eigen::Quaterniond q(state_.rot_end);
    evoFile << std::fixed;
    evoFile << Lidar_measures_.last_lio_update_time << " " << state_.pos_end[0]
            << " " << state_.pos_end[1] << " " << state_.pos_end[2] << " "
            << q.x() << " " << q.y() << " " << q.z() << " " << q.w()
            << std::endl;
  }

  euler_cur_ = RotMtoEuler(state_.rot_end);
  geo_quat_ = tf::createQuaternionMsgFromRollPitchYaw(
      euler_cur_(0), euler_cur_(1), euler_cur_(2));
  publish_odometry(pub_odom_aft_mapped_);

  double t3 = omp_get_wtime();

  // 更新VoxelMap
  PointCloudXYZIN::Ptr world_lidar(new PointCloudXYZIN());
  transformLidar(state_.rot_end, state_.pos_end, feats_down_body_, world_lidar);
  for (size_t i = 0; i < world_lidar->points.size(); i++) {
    voxelmap_manager_->pv_list_[i].point_w << world_lidar->points[i].x,
        world_lidar->points[i].y, world_lidar->points[i].z;
    M3D point_crossmat = voxelmap_manager_->cross_mat_list_[i];
    M3D var = voxelmap_manager_->body_cov_list_[i];
    var = (state_.rot_end * ext_r_) * var *
              (state_.rot_end * ext_r_).transpose() +
          (-point_crossmat) * state_.cov.block<3, 3>(0, 0) *
              (-point_crossmat).transpose() +
          state_.cov.block<3, 3>(3, 3);
    voxelmap_manager_->pv_list_[i].var = var;
  }
  voxelmap_manager_->UpdateVoxelMap(voxelmap_manager_->pv_list_);
  std::cout << "[ LIO ] Update Voxel Map" << std::endl;
  pv_list_ = voxelmap_manager_->pv_list_;

  double t4 = omp_get_wtime();

  if (voxelmap_manager_->config_setting_.map_sliding_en) {
    voxelmap_manager_->mapSliding();
  }

  PointCloudXYZIN::Ptr laserCloudFullRes(dense_map_en ? feats_undistort_
                                                      : feats_down_body_);
  int size = laserCloudFullRes->points.size();
  PointCloudXYZIN::Ptr laserCloudWorld(new PointCloudXYZIN(size, 1));

  for (int i = 0; i < size; i++) {
    RGBpointBodyToWorld(&laserCloudFullRes->points[i],
                        &laserCloudWorld->points[i]);
  }
  *pcl_w_wait_pub_ = *laserCloudWorld;

  if (!img_en_) publish_frame_world(pubLaser_cloud_full_res_, vio_manager_);
  if (pub_effect_point_en_)
    publish_effect_world(pub_laser_cloud_effect_,
                         voxelmap_manager_->ptpl_list_);
  if (voxelmap_manager_->config_setting_.is_pub_plane_map_)
    voxelmap_manager_->pubVoxelMap();
  publish_path(pub_path_);
  publish_mavros(mavros_pose_publisher_);

  frame_num_++;
  aver_time_consu_ =
      aver_time_consu_ * (frame_num_ - 1) / frame_num_ + (t4 - t0) / frame_num_;

  // aver_time_icp = aver_time_icp * (frame_num - 1) / frame_num + (t2 - t1) /
  // frame_num; aver_time_map_inre = aver_time_map_inre * (frame_num - 1) /
  // frame_num + (t4 - t3) / frame_num; aver_time_solve = aver_time_solve *
  // (frame_num - 1) / frame_num + (solve_time_) / frame_num;
  // aver_time_const_H_time = aver_time_const_H_time * (frame_num - 1) /
  // frame_num + solve_const_H_time_ / frame_num; printf("[ mapping time ]: per
  // scan: propagation %0.6f downsample: %0.6f match: %0.6f solve: %0.6f  ICP:
  // %0.6f  map incre: %0.6f total: %0.6f \n"
  //         "[ mapping time ]: average: icp: %0.6f construct H: %0.6f, total:
  //         %0.6f \n", t_prop - t0, t1 - t_prop, match_time, solve_time_, t3 -
  //         t1, t5 - t3, t5 - t0, aver_time_icp, aver_time_const_H_time,
  //         aver_time_consu);

  // printf("\033[1;36m[ LIO mapping time ]: current scan: icp: %0.6f secs, map
  // incre: %0.6f secs, total: %0.6f secs.\033[0m\n"
  //         "\033[1;36m[ LIO mapping time ]: average: icp: %0.6f secs, map
  //         incre: %0.6f secs, total: %0.6f secs.\033[0m\n", t2 - t1, t4 - t3,
  //         t4 - t0, aver_time_icp, aver_time_map_inre, aver_time_consu);
  printf(
      "\033[1;34m+-------------------------------------------------------------"
      "+\033[0m\n");
  printf(
      "\033[1;34m|                         LIO Mapping Time                    "
      "|\033[0m\n");
  printf(
      "\033[1;34m+-------------------------------------------------------------"
      "+\033[0m\n");
  printf("\033[1;34m| %-29s | %-27s |\033[0m\n", "Algorithm Stage",
         "Time (secs)");
  printf(
      "\033[1;34m+-------------------------------------------------------------"
      "+\033[0m\n");
  printf("\033[1;36m| %-29s | %-27f |\033[0m\n", "DownSample", t_down - t0);
  printf("\033[1;36m| %-29s | %-27f |\033[0m\n", "ICP", t2 - t1);
  printf("\033[1;36m| %-29s | %-27f |\033[0m\n", "updateVoxelMap", t4 - t3);
  printf(
      "\033[1;34m+-------------------------------------------------------------"
      "+\033[0m\n");
  printf("\033[1;36m| %-29s | %-27f |\033[0m\n", "Current Total Time", t4 - t0);
  printf("\033[1;36m| %-29s | %-27f |\033[0m\n", "Average Total Time",
         aver_time_consu_);
  printf(
      "\033[1;34m+-------------------------------------------------------------"
      "+\033[0m\n");

  euler_cur_ = RotMtoEuler(state_.rot_end);
  fout_out_ << std::setw(20)
            << Lidar_measures_.last_lio_update_time - first_lidar_time_ << " "
            << euler_cur_.transpose() * 57.3 << " "
            << state_.pos_end.transpose() << " " << state_.vel_end.transpose()
            << " " << state_.bias_g.transpose() << " "
            << state_.bias_a.transpose() << " "
            << V3D(state_.inv_expo_time, 0, 0).transpose() << " "
            << feats_undistort_->points.size() << std::endl;
}

void LIVMapper::savePCD() {
  if (pcd_save_en_ &&
      (pcl_wait_save_->points.size() > 0 ||
       pcl_wait_save_intensity_->points.size() > 0) &&
      pcd_save_interval_ < 0) {
    std::string raw_points_dir =
        std::string(ROOT_DIR) + "Log/PCD/all_raw_points.pcd";
    std::string downsampled_points_dir =
        std::string(ROOT_DIR) + "Log/PCD/all_downsampled_points.pcd";
    pcl::PCDWriter pcd_writer;

    if (img_en_) {
      pcl::PointCloud<pcl::PointXYZRGB>::Ptr downsampled_cloud(
          new pcl::PointCloud<pcl::PointXYZRGB>);
      pcl::VoxelGrid<pcl::PointXYZRGB> voxel_filter;
      voxel_filter.setInputCloud(pcl_wait_save_);
      voxel_filter.setLeafSize(filter_size_pcd_, filter_size_pcd_,
                               filter_size_pcd_);
      voxel_filter.filter(*downsampled_cloud);

      pcd_writer.writeBinary(raw_points_dir,
                             *pcl_wait_save_);  // Save the raw point cloud data
      std::cout << GREEN << "Raw point cloud data saved to: " << raw_points_dir
                << " with point count: " << pcl_wait_save_->points.size()
                << RESET << std::endl;

      pcd_writer.writeBinary(
          downsampled_points_dir,
          *downsampled_cloud);  // Save the downsampled point cloud data
      std::cout << GREEN << "Downsampled point cloud data saved to: "
                << downsampled_points_dir
                << " with point count after filtering: "
                << downsampled_cloud->points.size() << RESET << std::endl;

      if (colmap_output_en_) {
        fout_points_ << "# 3D point list with one line of data per point\n";
        fout_points_ << "#  POINT_ID, X, Y, Z, R, G, B, ERROR\n";
        for (size_t i = 0; i < downsampled_cloud->size(); ++i) {
          const auto &point = downsampled_cloud->points[i];
          fout_points_ << i << " " << std::fixed << std::setprecision(6)
                       << point.x << " " << point.y << " " << point.z << " "
                       << static_cast<int>(point.r) << " "
                       << static_cast<int>(point.g) << " "
                       << static_cast<int>(point.b) << " " << 0 << std::endl;
        }
      }
    } else {
      pcd_writer.writeBinary(raw_points_dir, *pcl_wait_save_intensity_);
      std::cout << GREEN << "Raw point cloud data saved to: " << raw_points_dir
                << " with point count: "
                << pcl_wait_save_intensity_->points.size() << RESET
                << std::endl;
    }
  }
}

// 主函数
void LIVMapper::run() {
  ros::Rate rate(5000);
  while (ros::ok()) {
    ros::spinOnce();
    if (!sync_packages(Lidar_measures_)) {
      rate.sleep();
      continue;
    }
    handleFirstFrame();

    processImu();

    stateEstimationAndMapping();
  }
  savePCD();
}

void LIVMapper::prop_imu_once(StatesGroup &imu_prop_state, const double dt,
                              V3D acc_avr, V3D angvel_avr) {
  double mean_acc_norm = p_imu_->imu_mean_acc_norm_;
  acc_avr = acc_avr * G_m_s2 / mean_acc_norm - imu_prop_state.bias_a;
  angvel_avr -= imu_prop_state.bias_g;

  M3D Exp_f = Exp(angvel_avr, dt);
  /* propogation of IMU attitude */
  imu_prop_state.rot_end = imu_prop_state.rot_end * Exp_f;

  /* Specific acceleration (global frame) of IMU */
  V3D acc_imu = imu_prop_state.rot_end * acc_avr +
                V3D(imu_prop_state.gravity[0], imu_prop_state.gravity[1],
                    imu_prop_state.gravity[2]);

  /* propogation of IMU */
  imu_prop_state.pos_end = imu_prop_state.pos_end +
                           imu_prop_state.vel_end * dt +
                           0.5 * acc_imu * dt * dt;

  /* velocity of IMU */
  imu_prop_state.vel_end = imu_prop_state.vel_end + acc_imu * dt;
}

void LIVMapper::imu_prop_callback(const ros::TimerEvent &e) {
  if (p_imu_->imu_need_init_ || !new_imu_ || !ekf_finish_once_) {
    return;
  }
  mtx_buffer_imu_prop_.lock();
  new_imu_ = false;  // 控制propagate频率和IMU频率一致
  if (imu_prop_enable_ && !prop_imu_buffer_.empty()) {
    static double last_t_from_lidar_end_time = 0;
    if (state_update_flg_) {
      imu_propagate_ = latest_ekf_state_;
      // drop all useless imu pkg
      while (
          (!prop_imu_buffer_.empty() &&
           prop_imu_buffer_.front().header.stamp.toSec() < latest_ekf_time_)) {
        prop_imu_buffer_.pop_front();
      }
      last_t_from_lidar_end_time = 0;
      for (int i = 0; i < prop_imu_buffer_.size(); i++) {
        double t_from_lidar_end_time =
            prop_imu_buffer_[i].header.stamp.toSec() - latest_ekf_time_;
        double dt = t_from_lidar_end_time - last_t_from_lidar_end_time;
        // cout << "prop dt" << dt << ", " << t_from_lidar_end_time << ", " <<
        // last_t_from_lidar_end_time << endl;
        V3D acc_imu(prop_imu_buffer_[i].linear_acceleration.x,
                    prop_imu_buffer_[i].linear_acceleration.y,
                    prop_imu_buffer_[i].linear_acceleration.z);
        V3D omg_imu(prop_imu_buffer_[i].angular_velocity.x,
                    prop_imu_buffer_[i].angular_velocity.y,
                    prop_imu_buffer_[i].angular_velocity.z);
        prop_imu_once(imu_propagate_, dt, acc_imu, omg_imu);
        last_t_from_lidar_end_time = t_from_lidar_end_time;
      }
      state_update_flg_ = false;
    } else {
      V3D acc_imu(newest_imu_.linear_acceleration.x,
                  newest_imu_.linear_acceleration.y,
                  newest_imu_.linear_acceleration.z);
      V3D omg_imu(newest_imu_.angular_velocity.x,
                  newest_imu_.angular_velocity.y,
                  newest_imu_.angular_velocity.z);
      double t_from_lidar_end_time =
          newest_imu_.header.stamp.toSec() - latest_ekf_time_;
      double dt = t_from_lidar_end_time - last_t_from_lidar_end_time;
      prop_imu_once(imu_propagate_, dt, acc_imu, omg_imu);
      last_t_from_lidar_end_time = t_from_lidar_end_time;
    }

    V3D posi, vel_i;
    Eigen::Quaterniond q;
    posi = imu_propagate_.pos_end;
    vel_i = imu_propagate_.vel_end;
    q = Eigen::Quaterniond(imu_propagate_.rot_end);
    imu_prop_odom_.header.frame_id = "world";
    imu_prop_odom_.header.stamp = newest_imu_.header.stamp;
    imu_prop_odom_.pose.pose.position.x = posi.x();
    imu_prop_odom_.pose.pose.position.y = posi.y();
    imu_prop_odom_.pose.pose.position.z = posi.z();
    imu_prop_odom_.pose.pose.orientation.w = q.w();
    imu_prop_odom_.pose.pose.orientation.x = q.x();
    imu_prop_odom_.pose.pose.orientation.y = q.y();
    imu_prop_odom_.pose.pose.orientation.z = q.z();
    imu_prop_odom_.twist.twist.linear.x = vel_i.x();
    imu_prop_odom_.twist.twist.linear.y = vel_i.y();
    imu_prop_odom_.twist.twist.linear.z = vel_i.z();
    pub_imu_prop_odom_.publish(imu_prop_odom_);
  }
  mtx_buffer_imu_prop_.unlock();
}

void LIVMapper::transformLidar(const Eigen::Matrix3d rot,
                               const Eigen::Vector3d t,
                               const PointCloudXYZIN::Ptr &input_cloud,
                               PointCloudXYZIN::Ptr &trans_cloud) {
  PointCloudXYZIN().swap(*trans_cloud);
  trans_cloud->reserve(input_cloud->size());
  for (size_t i = 0; i < input_cloud->size(); i++) {
    pcl::PointXYZINormal p_c = input_cloud->points[i];
    Eigen::Vector3d p(p_c.x, p_c.y, p_c.z);
    p = (rot * (ext_r_ * p + ext_t_) + t);
    PointXYZIN pi;
    pi.x = p(0);
    pi.y = p(1);
    pi.z = p(2);
    pi.intensity = p_c.intensity;
    trans_cloud->points.push_back(pi);
  }
}

void LIVMapper::pointBodyToWorld(const PointXYZIN &pi, PointXYZIN &po) {
  V3D p_body(pi.x, pi.y, pi.z);
  V3D p_global(state_.rot_end * (ext_r_ * p_body + ext_t_) + state_.pos_end);
  po.x = p_global(0);
  po.y = p_global(1);
  po.z = p_global(2);
  po.intensity = pi.intensity;
}

template <typename T>
void LIVMapper::pointBodyToWorld(const Matrix<T, 3, 1> &pi,
                                 Matrix<T, 3, 1> &po) {
  V3D p_body(pi[0], pi[1], pi[2]);
  V3D p_global(state_.rot_end * (ext_r_ * p_body + ext_t_) + state_.pos_end);
  po[0] = p_global(0);
  po[1] = p_global(1);
  po[2] = p_global(2);
}

template <typename T>
Matrix<T, 3, 1> LIVMapper::pointBodyToWorld(const Matrix<T, 3, 1> &pi) {
  V3D p(pi[0], pi[1], pi[2]);
  p = (state_.rot_end * (ext_r_ * p + ext_t_) + state_.pos_end);
  Matrix<T, 3, 1> po(p[0], p[1], p[2]);
  return po;
}

void LIVMapper::RGBpointBodyToWorld(PointXYZIN const *const pi,
                                    PointXYZIN *const po) {
  V3D p_body(pi->x, pi->y, pi->z);
  V3D p_global(state_.rot_end * (ext_r_ * p_body + ext_t_) + state_.pos_end);
  po->x = p_global(0);
  po->y = p_global(1);
  po->z = p_global(2);
  po->intensity = pi->intensity;
}

void LIVMapper::standard_pcl_cbk(
    const sensor_msgs::PointCloud2::ConstPtr &msg) {
  if (!lidar_en_) return;
  mtx_buffer_.lock();

  double cur_head_time = msg->header.stamp.toSec() + lidar_time_offset_;
  // cout<<"got feature"<<endl;
  if (cur_head_time < last_timestamp_lidar_) {
    ROS_ERROR("lidar loop back, clear buffer");
    lid_raw_data_buffer_.clear();
  }
  // ROS_INFO("get point cloud at time: %.6f", msg->header.stamp.toSec());
  PointCloudXYZIN::Ptr ptr(new PointCloudXYZIN());
  p_pre_->process(msg, ptr);
  lid_raw_data_buffer_.push_back(ptr);
  lid_header_time_buffer_.push_back(cur_head_time);
  last_timestamp_lidar_ = cur_head_time;

  mtx_buffer_.unlock();
  sig_buffer_.notify_all();
}

void LIVMapper::livox_pcl_cbk(
    const livox_ros_driver::CustomMsg::ConstPtr &msg_in) {
  if (!lidar_en_) return;
  mtx_buffer_.lock();
  livox_ros_driver::CustomMsg::Ptr msg(
      new livox_ros_driver::CustomMsg(*msg_in));
  // if ((abs(msg->header.stamp.toSec() - last_timestamp_lidar) > 0.2 &&
  // last_timestamp_lidar > 0) || sync_jump_flag_)
  // {
  //   ROS_WARN("lidar jumps %.3f\n", msg->header.stamp.toSec() -
  //   last_timestamp_lidar); sync_jump_flag_ = true; msg->header.stamp =
  //   ros::Time().fromSec(last_timestamp_lidar + 0.1);
  // }
  if (abs(last_timestamp_imu_ - msg->header.stamp.toSec()) > 1.0 &&
      !imu_buffer_.empty()) {
    double timediff_imu_wrt_lidar =
        last_timestamp_imu_ - msg->header.stamp.toSec();
    printf("\033[95mSelf sync IMU and LiDAR, HARD time lag is %.10lf \n\033[0m",
           timediff_imu_wrt_lidar - 0.100);
    // imu_time_offset_ = timediff_imu_wrt_lidar;
  }

  double cur_head_time = msg->header.stamp.toSec();
  ROS_INFO("Get LiDAR, its header time: %.6f", cur_head_time);
  if (cur_head_time < last_timestamp_lidar_) {
    ROS_ERROR("lidar loop back, clear buffer");
    lid_raw_data_buffer_.clear();
  }
  // ROS_INFO("get point cloud at time: %.6f", msg->header.stamp.toSec());
  PointCloudXYZIN::Ptr ptr(new PointCloudXYZIN());
  p_pre_->process(msg, ptr);

  if (!ptr || ptr->empty()) {
    ROS_ERROR("Received an empty point cloud");
    mtx_buffer_.unlock();
    return;
  }

  lid_raw_data_buffer_.push_back(ptr);
  lid_header_time_buffer_.push_back(cur_head_time);
  last_timestamp_lidar_ = cur_head_time;

  mtx_buffer_.unlock();
  sig_buffer_.notify_all();
}

void LIVMapper::imu_cbk(const sensor_msgs::Imu::ConstPtr &msg_in) {
  if (!imu_en_) return;

  if (last_timestamp_lidar_ < 0.0) return;
  // ROS_INFO("get imu at time: %.6f", msg_in->header.stamp.toSec());
  sensor_msgs::Imu::Ptr msg(new sensor_msgs::Imu(*msg_in));
  msg->header.stamp =
      ros::Time().fromSec(msg->header.stamp.toSec() - imu_time_offset_);
  double timestamp = msg->header.stamp.toSec();

  if (fabs(last_timestamp_lidar_ - timestamp) > 0.5 && (!ros_driver_fix_en_)) {
    ROS_WARN("IMU and LiDAR not synced! delta time: %lf .\n",
             last_timestamp_lidar_ - timestamp);
  }

  if (ros_driver_fix_en_)
    timestamp += std::round(last_timestamp_lidar_ - timestamp);
  msg->header.stamp = ros::Time().fromSec(timestamp);

  mtx_buffer_.lock();

  if (last_timestamp_imu_ > 0.0 && timestamp < last_timestamp_imu_) {
    mtx_buffer_.unlock();
    sig_buffer_.notify_all();
    ROS_ERROR("imu loop back, offset: %lf \n", last_timestamp_imu_ - timestamp);
    return;
  }

  // if (last_timestamp_imu > 0.0 && timestamp > last_timestamp_imu + 0.2)
  // {

  //   ROS_WARN("imu time stamp Jumps %0.4lf seconds \n", timestamp -
  //   last_timestamp_imu); mtx_buffer.unlock(); sig_buffer.notify_all();
  //   return;
  // }

  last_timestamp_imu_ = timestamp;

  imu_buffer_.push_back(msg);
  // cout<<"got imu: "<<timestamp<<" imu size "<<imu_buffer.size()<<endl;
  mtx_buffer_.unlock();
  if (imu_prop_enable_) {
    mtx_buffer_imu_prop_.lock();
    if (imu_prop_enable_ && !p_imu_->imu_need_init_) {
      prop_imu_buffer_.push_back(*msg);
    }
    newest_imu_ = *msg;
    new_imu_ = true;
    mtx_buffer_imu_prop_.unlock();
  }
  sig_buffer_.notify_all();
}

cv::Mat LIVMapper::getImageFromMsg(const sensor_msgs::ImageConstPtr &img_msg) {
  cv::Mat img;
  img = cv_bridge::toCvCopy(img_msg, "bgr8")->image;
  return img;
}

// static int i = 0;
void LIVMapper::img_cbk(const sensor_msgs::ImageConstPtr &msg_in) {
  if (!img_en_) return;
  sensor_msgs::Image::Ptr msg(new sensor_msgs::Image(*msg_in));
  // if ((abs(msg->header.stamp.toSec() - last_timestamp_img) > 0.2 &&
  // last_timestamp_img > 0) || sync_jump_flag_)
  // {
  //   ROS_WARN("img jumps %.3f\n", msg->header.stamp.toSec() -
  //   last_timestamp_img); sync_jump_flag_ = true; msg->header.stamp =
  //   ros::Time().fromSec(last_timestamp_img + 0.1);
  // }

  // Hiliti2022 40Hz
  // if (hilti_en)
  // {
  //   i++;
  //   if (i % 4 != 0) return;
  // }
  // double msg_header_time =  msg->header.stamp.toSec();
  double msg_header_time = msg->header.stamp.toSec() + img_time_offset_;
  if (abs(msg_header_time - last_timestamp_img_) < 0.001) return;
  ROS_INFO("Get image, its header time: %.6f", msg_header_time);
  if (last_timestamp_lidar_ < 0) return;

  if (msg_header_time < last_timestamp_img_) {
    ROS_ERROR("image loop back. \n");
    return;
  }

  mtx_buffer_.lock();

  double img_time_correct = msg_header_time;  // last_timestamp_lidar + 0.105;

  if (img_time_correct - last_timestamp_img_ < 0.02) {
    ROS_WARN("Image need Jumps: %.6f", img_time_correct);
    mtx_buffer_.unlock();
    sig_buffer_.notify_all();
    return;
  }

  cv::Mat img_cur = getImageFromMsg(msg);
  img_buffer_.push_back(img_cur);
  img_time_buffer_.push_back(img_time_correct);

  // ROS_INFO("Correct Image time: %.6f", img_time_correct);

  last_timestamp_img_ = img_time_correct;
  // cv::imshow("img", img);
  // cv::waitKey(1);
  // cout<<"last_timestamp_img:::"<<last_timestamp_img<<endl;
  mtx_buffer_.unlock();
  sig_buffer_.notify_all();
}

bool LIVMapper::sync_packages(LidarMeasureGroup &meas) {
  if (lid_raw_data_buffer_.empty() && lidar_en_) return false;
  if (img_buffer_.empty() && img_en_) return false;
  if (imu_buffer_.empty() && imu_en_) return false;

  // LIVO模式下，LIO和VIO的更新时间相同，LIO在前，VIO紧随其后。
  EKF_STATE last_lio_vio_flg = meas.lio_vio_flg;
  switch (last_lio_vio_flg) {
    case WAIT:
    case VIO: {
      // 注意：此处整理的是LIO使用的数据
      // 图像开始时间+曝光时间
      double img_capture_time = img_time_buffer_.front() + exposure_time_init_;
      // 存在图像话题，但图像话题时间戳大于激光雷达结束时间此时处理激光雷达话题。LIO更新后，meas.lidar_frame_end_time将被刷新。
      if (meas.last_lio_update_time < 0.0) {
        meas.last_lio_update_time = lid_header_time_buffer_.front();
      }

      // 取雷达和IMU最新的时间
      double lid_newest_time =
          lid_header_time_buffer_.back() +
          lid_raw_data_buffer_.back()->points.back().curvature / double(1000);
      double imu_newest_time = imu_buffer_.back()->header.stamp.toSec();

      // 图像数据时间戳过小，丢弃过时数据
      if (img_capture_time < meas.last_lio_update_time + 0.00001) {
        img_buffer_.pop_front();
        img_time_buffer_.pop_front();
        ROS_ERROR("[ Data Cut ] Throw one image frame! \n");
        return false;
      }

      // 图像的获取时间大于雷达和IMU最新的时间，等待雷达和IMU数据
      if (img_capture_time > lid_newest_time ||
          img_capture_time > imu_newest_time) {
        return false;
      }

      struct MeasureGroup m;
      // 处理IMU数据
      m.lio_time = img_capture_time;
      mtx_buffer_.lock();
      while (!imu_buffer_.empty()) {
        if (imu_buffer_.front()->header.stamp.toSec() > m.lio_time) break;

        if (imu_buffer_.front()->header.stamp.toSec() >
            meas.last_lio_update_time)
          m.imu.push_back(imu_buffer_.front());

        imu_buffer_.pop_front();
      }
      mtx_buffer_.unlock();
      sig_buffer_.notify_all();

      // 处理激光雷达数据
      // 上一帧的next移动至当前帧的cur，同时清理next
      *(meas.pcl_proc_cur) = *(meas.pcl_proc_next);
      PointCloudXYZIN().swap(*meas.pcl_proc_next);

      int lid_frame_num = lid_raw_data_buffer_.size();
      int max_size = meas.pcl_proc_cur->size() + 24000 * lid_frame_num;
      meas.pcl_proc_cur->reserve(max_size);
      meas.pcl_proc_next->reserve(max_size);

      while (!lid_raw_data_buffer_.empty()) {
        if (lid_header_time_buffer_.front() > img_capture_time) break;
        auto pcl(lid_raw_data_buffer_.front()->points);
        double frame_header_time(lid_header_time_buffer_.front());
        float max_offs_time_ms = (m.lio_time - frame_header_time) * 1000.0f;
        // 时间小于图像获取时间的点，放入当前帧
        for (int i = 0; i < pcl.size(); i++) {
          auto pt = pcl[i];
          if (pcl[i].curvature < max_offs_time_ms) {
            pt.curvature +=
                (frame_header_time - meas.last_lio_update_time) * 1000.0f;
            meas.pcl_proc_cur->points.push_back(pt);
          } else {
            pt.curvature += (frame_header_time - m.lio_time) * 1000.0f;
            meas.pcl_proc_next->points.push_back(pt);
          }
        }
        lid_raw_data_buffer_.pop_front();
        lid_header_time_buffer_.pop_front();
      }

      meas.measures.push_back(m);
      // 模式调整为LIO
      meas.lio_vio_flg = LIO;
      return true;
    }

    case LIO: {
      // 注意：此处整理的是VIO使用的数据
      double img_capture_time = img_time_buffer_.front() + exposure_time_init_;
      // 模式调整为VIO
      meas.lio_vio_flg = VIO;
      meas.measures.clear();
      double imu_time = imu_buffer_.front()->header.stamp.toSec();
      // 只添加图片
      struct MeasureGroup m;
      m.vio_time = img_capture_time;
      m.lio_time = meas.last_lio_update_time;
      m.img = img_buffer_.front();
      mtx_buffer_.lock();
      img_buffer_.pop_front();
      img_time_buffer_.pop_front();
      mtx_buffer_.unlock();
      sig_buffer_.notify_all();
      meas.measures.push_back(m);
      lidar_pushed_ = false;
      return true;
    }

    default: {
      return false;
    }
  }

  ROS_ERROR("out sync");
}

void LIVMapper::publish_img_rgb(const image_transport::Publisher &pubImage,
                                VIOManagerPtr vio_manager) {
  cv::Mat img_rgb = vio_manager->img_cp_;
  cv_bridge::CvImage out_msg;
  out_msg.header.stamp = ros::Time::now();
  // out_msg.header.frame_id = "camera_init";
  out_msg.encoding = sensor_msgs::image_encodings::BGR8;
  out_msg.image = img_rgb;
  pubImage.publish(out_msg.toImageMsg());
}

void LIVMapper::publish_frame_world(const ros::Publisher &pubLaserCloudFullRes,
                                    VIOManagerPtr vio_manager) {
  if (pcl_w_wait_pub_->empty()) return;
  PointCloudXYZRGB::Ptr laserCloudWorldRGB(new PointCloudXYZRGB());
  if (img_en_) {
    static int pub_num = 1;
    *pcl_wait_pub_ += *pcl_w_wait_pub_;
    if (pub_num == pub_scan_num_) {
      pub_num = 1;
      size_t size = pcl_wait_pub_->points.size();
      laserCloudWorldRGB->reserve(size);
      // double inv_expo = _state.inv_expo_time;
      cv::Mat img_rgb = vio_manager->img_rgb_;
      for (size_t i = 0; i < size; i++) {
        PointTypeRGB pointRGB;
        pointRGB.x = pcl_wait_pub_->points[i].x;
        pointRGB.y = pcl_wait_pub_->points[i].y;
        pointRGB.z = pcl_wait_pub_->points[i].z;

        V3D p_w(pcl_wait_pub_->points[i].x, pcl_wait_pub_->points[i].y,
                pcl_wait_pub_->points[i].z);
        V3D pf(vio_manager->new_frame_->w2f(p_w));
        if (pf[2] < 0) continue;
        V2D pc(vio_manager->new_frame_->w2c(p_w));

        if (vio_manager->new_frame_->cam_->isInFrame(pc.cast<int>(), 3))  // 100
        {
          V3F pixel = vio_manager->GetInterpolatedPixel(img_rgb, pc);
          pointRGB.r = pixel[2];
          pointRGB.g = pixel[1];
          pointRGB.b = pixel[0];
          // pointRGB.r = pixel[2] * inv_expo; pointRGB.g = pixel[1] * inv_expo;
          // pointRGB.b = pixel[0] * inv_expo; if (pointRGB.r > 255) pointRGB.r
          // = 255; else if (pointRGB.r < 0) pointRGB.r = 0; if (pointRGB.g >
          // 255) pointRGB.g = 255; else if (pointRGB.g < 0) pointRGB.g = 0; if
          // (pointRGB.b > 255) pointRGB.b = 255; else if (pointRGB.b < 0)
          // pointRGB.b = 0;
          if (pf.norm() > blind_rgb_points_)
            laserCloudWorldRGB->push_back(pointRGB);
        }
      }
    } else {
      pub_num++;
    }
  }

  /*** Publish Frame ***/
  sensor_msgs::PointCloud2 laserCloudmsg;
  if (img_en_) {
    // cout << "RGB pointcloud size: " << laserCloudWorldRGB->size() << endl;
    pcl::toROSMsg(*laserCloudWorldRGB, laserCloudmsg);
  } else {
    pcl::toROSMsg(*pcl_w_wait_pub_, laserCloudmsg);
  }
  laserCloudmsg.header.stamp =
      ros::Time::now();  //.fromSec(last_timestamp_lidar);
  laserCloudmsg.header.frame_id = "camera_init";
  pubLaserCloudFullRes.publish(laserCloudmsg);

  /**************** save map ****************/
  /* 1. make sure you have enough memories
  /* 2. noted that pcd save will influence the real-time performences **/
  if (pcd_save_en_) {
    int size = feats_undistort_->points.size();
    PointCloudXYZIN::Ptr laserCloudWorld(new PointCloudXYZIN(size, 1));
    static int scan_wait_num = 0;

    if (img_en_) {
      *pcl_wait_save_ += *laserCloudWorldRGB;
    } else {
      *pcl_wait_save_intensity_ += *pcl_w_wait_pub_;
    }
    scan_wait_num++;

    if ((pcl_wait_save_->size() > 0 || pcl_wait_save_intensity_->size() > 0) &&
        pcd_save_interval_ > 0 && scan_wait_num >= pcd_save_interval_) {
      pcd_index_++;
      string all_points_dir(string(string(ROOT_DIR) + "Log/PCD/") +
                            to_string(pcd_index_) + string(".pcd"));
      pcl::PCDWriter pcd_writer;
      if (pcd_save_en_) {
        cout << "current scan saved to /PCD/" << all_points_dir << endl;
        if (img_en_) {
          pcd_writer.writeBinary(
              all_points_dir,
              *pcl_wait_save_);  // pcl::io::savePCDFileASCII(all_points_dir,
                                 // *pcl_wait_save);
          PointCloudXYZRGB().swap(*pcl_wait_save_);
        } else {
          pcd_writer.writeBinary(all_points_dir, *pcl_wait_save_intensity_);
          PointCloudXYZIN().swap(*pcl_wait_save_intensity_);
        }
        Eigen::Quaterniond q(state_.rot_end);
        fout_pcd_pos_ << state_.pos_end[0] << " " << state_.pos_end[1] << " "
                      << state_.pos_end[2] << " " << q.w() << " " << q.x()
                      << " " << q.y() << " " << q.z() << " " << endl;
        scan_wait_num = 0;
      }
    }
  }
  if (laserCloudWorldRGB->size() > 0) PointCloudXYZIN().swap(*pcl_wait_pub_);
  PointCloudXYZIN().swap(*pcl_w_wait_pub_);
}

void LIVMapper::publish_visual_sub_map(const ros::Publisher &pubSubVisualMap) {
  PointCloudXYZIN::Ptr laserCloudFullRes(visual_sub_map_);
  int size = laserCloudFullRes->points.size();
  if (size == 0) return;
  PointCloudXYZIN::Ptr sub_pcl_visual_map_pub(new PointCloudXYZIN());
  *sub_pcl_visual_map_pub = *laserCloudFullRes;
  if (1) {
    sensor_msgs::PointCloud2 laserCloudmsg;
    pcl::toROSMsg(*sub_pcl_visual_map_pub, laserCloudmsg);
    laserCloudmsg.header.stamp = ros::Time::now();
    laserCloudmsg.header.frame_id = "camera_init";
    pubSubVisualMap.publish(laserCloudmsg);
  }
}

void LIVMapper::publish_effect_world(
    const ros::Publisher &pubLaserCloudEffect,
    const std::vector<PointToPlane> &ptpl_list) {
  int effect_feat_num = ptpl_list.size();
  PointCloudXYZIN::Ptr laserCloudWorld(new PointCloudXYZIN(effect_feat_num, 1));
  for (int i = 0; i < effect_feat_num; i++) {
    laserCloudWorld->points[i].x = ptpl_list[i].point_w_[0];
    laserCloudWorld->points[i].y = ptpl_list[i].point_w_[1];
    laserCloudWorld->points[i].z = ptpl_list[i].point_w_[2];
  }
  sensor_msgs::PointCloud2 laserCloudFullRes3;
  pcl::toROSMsg(*laserCloudWorld, laserCloudFullRes3);
  laserCloudFullRes3.header.stamp = ros::Time::now();
  laserCloudFullRes3.header.frame_id = "camera_init";
  pubLaserCloudEffect.publish(laserCloudFullRes3);
}

template <typename T>
void LIVMapper::set_posestamp(T &out) {
  out.position.x = state_.pos_end(0);
  out.position.y = state_.pos_end(1);
  out.position.z = state_.pos_end(2);
  out.orientation.x = geo_quat_.x;
  out.orientation.y = geo_quat_.y;
  out.orientation.z = geo_quat_.z;
  out.orientation.w = geo_quat_.w;
}

void LIVMapper::publish_odometry(const ros::Publisher &pubOdomAftMapped) {
  odom_aft_mapped_.header.frame_id = "camera_init";
  odom_aft_mapped_.child_frame_id = "aft_mapped";
  odom_aft_mapped_.header.stamp =
      ros::Time::now();  //.ros::Time()fromSec(last_timestamp_lidar);
  set_posestamp(odom_aft_mapped_.pose.pose);

  static tf::TransformBroadcaster br;
  tf::Transform transform;
  tf::Quaternion q;
  transform.setOrigin(
      tf::Vector3(state_.pos_end(0), state_.pos_end(1), state_.pos_end(2)));
  q.setW(geo_quat_.w);
  q.setX(geo_quat_.x);
  q.setY(geo_quat_.y);
  q.setZ(geo_quat_.z);
  transform.setRotation(q);
  br.sendTransform(tf::StampedTransform(
      transform, odom_aft_mapped_.header.stamp, "camera_init", "aft_mapped"));
  pubOdomAftMapped.publish(odom_aft_mapped_);
}

void LIVMapper::publish_mavros(const ros::Publisher &mavros_pose_publisher) {
  msg_body_pose_.header.stamp = ros::Time::now();
  msg_body_pose_.header.frame_id = "camera_init";
  set_posestamp(msg_body_pose_.pose);
  mavros_pose_publisher.publish(msg_body_pose_);
}

void LIVMapper::publish_path(const ros::Publisher pubPath) {
  set_posestamp(msg_body_pose_.pose);
  msg_body_pose_.header.stamp = ros::Time::now();
  msg_body_pose_.header.frame_id = "camera_init";
  path_.poses.push_back(msg_body_pose_);
  pubPath.publish(path_);
}

// bool LIVMapper::sync_packages(LidarMeasureGroup &meas) {
//   if (lid_raw_data_buffer_.empty() && lidar_en_) return false;
//   if (img_buffer_.empty() && img_en_) return false;
//   if (imu_buffer_.empty() && imu_en_) return false;

//   switch (slam_mode_) {
//     case ONLY_LIO: {
//       if (meas.last_lio_update_time < 0.0)
//         meas.last_lio_update_time = lid_header_time_buffer_.front();
//       if (!lidar_pushed_) {
//         // If not push the lidar into measurement data buffer
//         meas.lidar =
//             lid_raw_data_buffer_.front();  // push the first lidar topic
//         if (meas.lidar->points.size() <= 1) return false;

//         meas.lidar_frame_beg_time =
//             lid_header_time_buffer_.front();  // generate
//             lidar_frame_beg_time
//         meas.lidar_frame_end_time =
//             meas.lidar_frame_beg_time +
//             meas.lidar->points.back().curvature /
//                 double(1000);  // calc lidar scan end time
//         meas.pcl_proc_cur = meas.lidar;
//         lidar_pushed_ = true;  // flag
//       }

//       if (imu_en_ &&
//           last_timestamp_imu_ <
//               meas.lidar_frame_end_time) {  // waiting imu message needs to
//               be
//         // larger than _lidar_frame_end_time,
//         // make sure complete propagate.
//         // ROS_ERROR("out sync");
//         return false;
//       }

//       struct MeasureGroup m;  // standard method to keep imu message.

//       m.imu.clear();
//       m.lio_time = meas.lidar_frame_end_time;
//       mtx_buffer_.lock();
//       while (!imu_buffer_.empty()) {
//         if (imu_buffer_.front()->header.stamp.toSec() >
//             meas.lidar_frame_end_time)
//           break;
//         m.imu.push_back(imu_buffer_.front());
//         imu_buffer_.pop_front();
//       }
//       lid_raw_data_buffer_.pop_front();
//       lid_header_time_buffer_.pop_front();
//       mtx_buffer_.unlock();
//       sig_buffer_.notify_all();

//       meas.lio_vio_flg =
//           LIO;  // process lidar topic, so timestamp should be lidar scan
//           end.
//       meas.measures.push_back(m);
//       // ROS_INFO("ONlY HAS LiDAR and IMU, NO IMAGE!");
//       lidar_pushed_ = false;  // sync one whole lidar scan.
//       return true;

//       break;
//     }

//     case LIVO: {
//       // LIVO模式下，LIO和VIO的更新时间相同，LIO在前，VIO紧随其后。
//       EKF_STATE last_lio_vio_flg = meas.lio_vio_flg;
//       switch (last_lio_vio_flg) {
//         case WAIT:
//         case VIO: {
//           // 注意：此处整理的是LIO使用的数据
//           // 图像开始时间+曝光时间
//           double img_capture_time =
//               img_time_buffer_.front() + exposure_time_init_;
//           //
//           存在图像话题，但图像话题时间戳大于激光雷达结束时间此时处理激光雷达话题。LIO更新后，meas.lidar_frame_end_time将被刷新。
//           if (meas.last_lio_update_time < 0.0) {
//             meas.last_lio_update_time = lid_header_time_buffer_.front();
//           }

//           // 取雷达和IMU最新的时间
//           double lid_newest_time =
//               lid_header_time_buffer_.back() +
//               lid_raw_data_buffer_.back()->points.back().curvature /
//                   double(1000);
//           double imu_newest_time = imu_buffer_.back()->header.stamp.toSec();

//           // 图像数据时间戳过小，丢弃过时数据
//           if (img_capture_time < meas.last_lio_update_time + 0.00001) {
//             img_buffer_.pop_front();
//             img_time_buffer_.pop_front();
//             ROS_ERROR("[ Data Cut ] Throw one image frame! \n");
//             return false;
//           }

//           // 图像的获取时间大于雷达和IMU最新的时间，等待雷达和IMU数据
//           if (img_capture_time > lid_newest_time ||
//               img_capture_time > imu_newest_time) {
//             return false;
//           }

//           struct MeasureGroup m;
//           // 处理IMU数据
//           m.lio_time = img_capture_time;
//           mtx_buffer_.lock();
//           while (!imu_buffer_.empty()) {
//             if (imu_buffer_.front()->header.stamp.toSec() > m.lio_time)
//             break;

//             if (imu_buffer_.front()->header.stamp.toSec() >
//                 meas.last_lio_update_time)
//               m.imu.push_back(imu_buffer_.front());

//             imu_buffer_.pop_front();
//           }
//           mtx_buffer_.unlock();
//           sig_buffer_.notify_all();

//           // 处理激光雷达数据
//           // 上一帧的next移动当前帧的cur，同时清理next
//           *(meas.pcl_proc_cur) = *(meas.pcl_proc_next);
//           PointCloudXYZIN().swap(*meas.pcl_proc_next);

//           int lid_frame_num = lid_raw_data_buffer_.size();
//           int max_size = meas.pcl_proc_cur->size() + 24000 * lid_frame_num;
//           meas.pcl_proc_cur->reserve(max_size);
//           meas.pcl_proc_next->reserve(max_size);

//           while (!lid_raw_data_buffer_.empty()) {
//             if (lid_header_time_buffer_.front() > img_capture_time) break;
//             auto pcl(lid_raw_data_buffer_.front()->points);
//             double frame_header_time(lid_header_time_buffer_.front());
//             float max_offs_time_ms = (m.lio_time - frame_header_time) *
//             1000.0f;
//             // 时间小于图像获取时间的点，放入当前帧
//             for (int i = 0; i < pcl.size(); i++) {
//               auto pt = pcl[i];
//               if (pcl[i].curvature < max_offs_time_ms) {
//                 pt.curvature +=
//                     (frame_header_time - meas.last_lio_update_time) *
//                     1000.0f;
//                 meas.pcl_proc_cur->points.push_back(pt);
//               } else {
//                 pt.curvature += (frame_header_time - m.lio_time) * 1000.0f;
//                 meas.pcl_proc_next->points.push_back(pt);
//               }
//             }
//             lid_raw_data_buffer_.pop_front();
//             lid_header_time_buffer_.pop_front();
//           }

//           meas.measures.push_back(m);
//           // 模式调整为LIO
//           meas.lio_vio_flg = LIO;
//           return true;
//         }

//         case LIO: {
//           // 注意：此处整理的是VIO使用的数据
//           double img_capture_time =
//               img_time_buffer_.front() + exposure_time_init_;
//           // 模式调整为VIO
//           meas.lio_vio_flg = VIO;
//           meas.measures.clear();
//           double imu_time = imu_buffer_.front()->header.stamp.toSec();
//           // 只添加图片
//           struct MeasureGroup m;
//           m.vio_time = img_capture_time;
//           m.lio_time = meas.last_lio_update_time;
//           m.img = img_buffer_.front();
//           mtx_buffer_.lock();
//           img_buffer_.pop_front();
//           img_time_buffer_.pop_front();
//           mtx_buffer_.unlock();
//           sig_buffer_.notify_all();
//           meas.measures.push_back(m);
//           lidar_pushed_ = false;
//           return true;
//         }

//         default: {
//           return false;
//         }
//       }
//       break;
//     }

//     case ONLY_LO: {
//       if (!lidar_pushed_) {
//         if (lid_raw_data_buffer_.empty())
//           return false;  // 如果激光雷达缓冲区为空，返回false
//         meas.lidar = lid_raw_data_buffer_
//                          .front();  // 推送第一个激光雷达数据到测量数据缓冲区
//         meas.lidar_frame_beg_time =
//             lid_header_time_buffer_.front();  // 生成激光雷达扫描开始时间
//         meas.lidar_frame_end_time =
//             meas.lidar_frame_beg_time +
//             meas.lidar->points.back().curvature /
//                 double(1000);  // 计算激光雷达扫描结束时间
//         lidar_pushed_ = true;
//       }
//       struct MeasureGroup m;  // 用于保持IMU数据的标准方法
//       m.lio_time =
//           meas.lidar_frame_end_time;  //
//           设置激光雷达惯性里程计的时间戳为激光雷达扫描结束时间
//       mtx_buffer_.lock();             // 锁定缓冲区以避免并发访问
//       lid_raw_data_buffer_.pop_front();  // 从激光雷达缓冲区移除已使用的数据
//       lid_header_time_buffer_
//           .pop_front();  // 从激光雷达时间戳缓冲区移除已使用的数据
//       mtx_buffer_.unlock();        // 解锁缓冲区
//       sig_buffer_.notify_all();    // 通知所有等待的线程
//       lidar_pushed_ = false;       // 标记激光雷达数据已同步
//       meas.lio_vio_flg = LO;       // 设置当前处理的标志为LO
//       meas.measures.push_back(m);  // 将测量组添加到测量数据中
//       return true;
//       break;
//     }

//     default: {
//       printf("!! WRONG SLAM TYPE !!");
//       return false;
//     }
//   }
//   ROS_ERROR("out sync");
// }
