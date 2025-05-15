/*
This file is part of FAST-LIVO2: Fast, Direct LiDAR-Inertial-Visual Odometry.

Developer: Chunran Zheng <zhengcr@connect.hku.hk>

For commercial use, please contact me at <zhengcr@connect.hku.hk> or
Prof. Fu Zhang at <fuzhang@hku.hk>.

This file is subject to the terms and conditions outlined in the 'LICENSE' file,
which is included as part of this source code package.
*/

#include "IMU_Processing.h"

#include <omp.h>

const bool time_list(PointXYZIN &x, PointXYZIN &y) {
  return (x.curvature < y.curvature);
}

ImuProcess::ImuProcess()
    : eye3d_(M3D::Identity()),
      zero3d_(0, 0, 0),
      b_first_frame_(true),
      imu_need_init_(true) {
  init_iter_num_ = 1;
  cov_acc_ = V3D(0.1, 0.1, 0.1);
  cov_gyr_ = V3D(0.1, 0.1, 0.1);
  cov_bias_gyr_ = V3D(0.1, 0.1, 0.1);
  cov_bias_acc_ = V3D(0.1, 0.1, 0.1);
  cov_inv_expo_ = 0.2;
  mean_acc_ = V3D(0, 0, -1.0);
  mean_gyr_ = V3D(0, 0, 0);
  angvel_last_ = zero3d_;
  acc_s_last_ = zero3d_;
  lidar_offset_to_imu_ = zero3d_;
  lidar_rot_to_imu_ = eye3d_;
  last_imu_.reset(new sensor_msgs::Imu());
  cur_pcl_un_.reset(new PointCloudXYZIN());
}

ImuProcess::~ImuProcess() {}

void ImuProcess::Reset() {
  ROS_WARN("Reset ImuProcess");
  mean_acc_ = V3D(0, 0, -1.0);
  mean_gyr_ = V3D(0, 0, 0);
  angvel_last_ = zero3d_;
  imu_need_init_ = true;
  init_iter_num_ = 1;
  imu_pose_.clear();
  last_imu_.reset(new sensor_msgs::Imu());
  cur_pcl_un_.reset(new PointCloudXYZIN());
}

void ImuProcess::disable_imu() {
  cout << "IMU Disabled !!!!!" << endl;
  imu_en_ = false;
  imu_need_init_ = false;
}

void ImuProcess::disable_gravity_est() {
  cout << "Online Gravity Estimation Disabled !!!!!" << endl;
  gravity_est_en_ = false;
}

void ImuProcess::disable_bias_est() {
  cout << "Bias Estimation Disabled !!!!!" << endl;
  ba_bg_est_en_ = false;
}

void ImuProcess::disable_exposure_est() {
  cout << "Online Time Offset Estimation Disabled !!!!!" << endl;
  exposure_estimate_en_ = false;
}

void ImuProcess::set_extrinsic(const MD(4, 4) & T) {
  lidar_offset_to_imu_ = T.block<3, 1>(0, 3);
  lidar_rot_to_imu_ = T.block<3, 3>(0, 0);
}

void ImuProcess::set_extrinsic(const V3D &transl) {
  lidar_offset_to_imu_ = transl;
  lidar_rot_to_imu_.setIdentity();
}

void ImuProcess::set_extrinsic(const V3D &transl, const M3D &rot) {
  lidar_offset_to_imu_ = transl;
  lidar_rot_to_imu_ = rot;
}

void ImuProcess::set_gyr_cov_scale(const V3D &scaler) { cov_gyr_ = scaler; }

void ImuProcess::set_acc_cov_scale(const V3D &scaler) { cov_acc_ = scaler; }

void ImuProcess::set_gyr_bias_cov(const V3D &b_g) { cov_bias_gyr_ = b_g; }

void ImuProcess::set_inv_expo_cov(const double &inv_expo) {
  cov_inv_expo_ = inv_expo;
}

void ImuProcess::set_acc_bias_cov(const V3D &b_a) { cov_bias_acc_ = b_a; }

void ImuProcess::set_imu_init_frame_num(const int &num) {
  max_ini_count_ = num;
}

void ImuProcess::IMU_init(const MeasureGroup &meas, StatesGroup &state_inout,
                          int &N) {
  /** 1. initializing the gravity, gyro bias, acc and gyro covariance
   ** 2. normalize the acceleration measurenments to unit gravity **/
  ROS_INFO("IMU Initializing: %.1f %%", double(N) / max_ini_count_ * 100);
  V3D cur_acc, cur_gyr;

  if (b_first_frame_) {
    Reset();
    N = 1;
    b_first_frame_ = false;
    const auto &imu_acc = meas.imu.front()->linear_acceleration;
    const auto &gyr_acc = meas.imu.front()->angular_velocity;
    mean_acc_ << imu_acc.x, imu_acc.y, imu_acc.z;
    mean_gyr_ << gyr_acc.x, gyr_acc.y, gyr_acc.z;
  }

  for (const auto &imu : meas.imu) {
    const auto &imu_acc = imu->linear_acceleration;
    const auto &gyr_acc = imu->angular_velocity;
    cur_acc << imu_acc.x, imu_acc.y, imu_acc.z;
    cur_gyr << gyr_acc.x, gyr_acc.y, gyr_acc.z;

    mean_acc_ += (cur_acc - mean_acc_) / N;
    mean_gyr_ += (cur_gyr - mean_gyr_) / N;

    // cov_acc = cov_acc * (N - 1.0) / N + (cur_acc -
    // mean_acc).cwiseProduct(cur_acc - mean_acc) * (N - 1.0) / (N * N); cov_gyr
    // = cov_gyr * (N - 1.0) / N + (cur_gyr - mean_gyr).cwiseProduct(cur_gyr -
    // mean_gyr) * (N - 1.0) / (N * N);

    N++;
  }
  imu_mean_acc_norm_ = mean_acc_.norm();
  state_inout.gravity = -mean_acc_ / mean_acc_.norm() * G_m_s2;
  state_inout.rot_end =
      eye3d_;  // Exp(mean_acc.cross(V3D(0, 0, -1 / scale_gravity)));
  state_inout.bias_g = zero3d_;  // mean_gyr;
}

void ImuProcess::Forward_without_imu(LidarMeasureGroup &meas,
                                     StatesGroup &state_inout,
                                     PointCloudXYZIN &pcl_out) {
  pcl_out = *(meas.lidar);
  // 按时间戳排序
  const double &pcl_beg_time = meas.lidar_frame_beg_time;
  sort(pcl_out.points.begin(), pcl_out.points.end(), time_list);
  const double &pcl_end_time =
      pcl_beg_time + pcl_out.points.back().curvature / double(1000);
  meas.last_lio_update_time = pcl_end_time;
  const double &pcl_end_offset_time =
      pcl_out.points.back().curvature / double(1000);

  Mat19d F_x, cov_w;
  double dt = 0;

  if (b_first_frame_) {
    dt = 0.1;
    b_first_frame_ = false;
  } else {
    dt = pcl_beg_time - time_last_scan_;
  }

  time_last_scan_ = pcl_beg_time;
  // 误差状态传播
  // M3D acc_avr_skew;
  M3D Exp_f = Exp(state_inout.bias_g, dt);

  F_x.setIdentity();
  cov_w.setZero();

  F_x.block<3, 3>(0, 0) = Exp(state_inout.bias_g, -dt);
  F_x.block<3, 3>(0, 10) = eye3d_ * dt;
  F_x.block<3, 3>(3, 7) = eye3d_ * dt;
  // F_x.block<3, 3>(6, 0)  = - R_imu * acc_avr_skew * dt;
  // F_x.block<3, 3>(6, 12) = - R_imu * dt;
  // F_x.block<3, 3>(6, 15) = Eye3d * dt;

  cov_w.block<3, 3>(10, 10).diagonal() =
      cov_gyr_ * dt * dt;  // for omega in constant model
  cov_w.block<3, 3>(7, 7).diagonal() =
      cov_acc_ * dt * dt;  // for velocity in constant model
  // cov_w.block<3, 3>(6, 6) =
  //     R_imu * cov_acc.asDiagonal() * R_imu.transpose() * dt * dt;
  // cov_w.block<3, 3>(9, 9).diagonal() =
  //     cov_bias_gyr * dt * dt; // bias gyro covariance
  // cov_w.block<3, 3>(12, 12).diagonal() =
  //     cov_bias_acc * dt * dt; // bias acc covariance

  // std::cout << "before propagete:" << state_inout.cov.diagonal().transpose()
  //           << std::endl;
  state_inout.cov = F_x * state_inout.cov * F_x.transpose() + cov_w;
  // std::cout << "cov_w:" << cov_w.diagonal().transpose() << std::endl;
  // std::cout << "after propagete:" << state_inout.cov.diagonal().transpose()
  //           << std::endl;
  state_inout.rot_end = state_inout.rot_end * Exp_f;
  state_inout.pos_end = state_inout.pos_end + state_inout.vel_end * dt;

  if (lidar_type_ != L515) {
    auto it_pcl = pcl_out.points.end() - 1;
    double dt_j = 0.0;
    for (; it_pcl != pcl_out.points.begin(); it_pcl--) {
      dt_j = pcl_end_offset_time - it_pcl->curvature / double(1000);
      M3D R_jk(Exp(state_inout.bias_g, -dt_j));
      V3D P_j(it_pcl->x, it_pcl->y, it_pcl->z);
      // Using rotation and translation to un-distort points
      V3D p_jk;
      p_jk = -state_inout.rot_end.transpose() * state_inout.vel_end * dt_j;

      V3D P_compensate = R_jk * P_j + p_jk;

      /// save Undistorted points and their rotation
      it_pcl->x = P_compensate(0);
      it_pcl->y = P_compensate(1);
      it_pcl->z = P_compensate(2);
    }
  }
}

void ImuProcess::UndistortPcl(LidarMeasureGroup &lidar_meas,
                              StatesGroup &state_inout,
                              PointCloudXYZIN &pcl_out) {
  pcl_out.clear();
  /*** add the imu of the last frame-tail to the of current frame-head ***/
  MeasureGroup &meas = lidar_meas.measures.back();
  // cout<<"meas.imu.size: "<<meas.imu.size()<<endl;
  auto v_imu = meas.imu;
  v_imu.push_front(last_imu_);
  const double &imu_beg_time = v_imu.front()->header.stamp.toSec();
  const double &imu_end_time = v_imu.back()->header.stamp.toSec();
  const double prop_beg_time = last_prop_end_time_;

  const double prop_end_time =
      lidar_meas.lio_vio_flg == LIO ? meas.lio_time : meas.vio_time;

  /*** cut lidar point based on the propagation-start time and required
   * propagation-end time ***/
  if (lidar_meas.lio_vio_flg == LIO) {
    pcl_wait_proc_.resize(lidar_meas.pcl_proc_cur->points.size());
    pcl_wait_proc_ = *(lidar_meas.pcl_proc_cur);
    lidar_meas.lidar_scan_index_now = 0;
    imu_pose_.push_back(set_pose6d(0.0, acc_s_last_, angvel_last_,
                                   state_inout.vel_end, state_inout.pos_end,
                                   state_inout.rot_end));
  }
  /*** Initialize IMU pose ***/
  /*** forward propagation at each imu point ***/
  // 上一帧最后时刻的状态
  V3D acc_imu(acc_s_last_), angvel_avr(angvel_last_), acc_avr,
      vel_imu(state_inout.vel_end), pos_imu(state_inout.pos_end);

  M3D R_imu(state_inout.rot_end);
  Mat19d F_x, cov_w;
  double dt, dt_all = 0.0;
  double offs_t;

  double tau;
  if (!imu_time_init_) {
    tau = 1.0;
    imu_time_init_ = true;
  } else {
    tau = state_inout.inv_expo_time;
  }

  // IMU位姿预测
  switch (lidar_meas.lio_vio_flg) {
    case LIO:
    case VIO:
      dt = 0;
      for (int i = 0; i < v_imu.size() - 1; i++) {
        auto head = v_imu[i];
        auto tail = v_imu[i + 1];

        if (tail->header.stamp.toSec() < prop_beg_time) continue;
        // 中值滤波
        angvel_avr << 0.5 *
                          (head->angular_velocity.x + tail->angular_velocity.x),
            0.5 * (head->angular_velocity.y + tail->angular_velocity.y),
            0.5 * (head->angular_velocity.z + tail->angular_velocity.z);

        acc_avr << 0.5 * (head->linear_acceleration.x +
                          tail->linear_acceleration.x),
            0.5 * (head->linear_acceleration.y + tail->linear_acceleration.y),
            0.5 * (head->linear_acceleration.z + tail->linear_acceleration.z);

        fout_imu_ << setw(10) << head->header.stamp.toSec() - first_lidar_time_
                  << " " << angvel_avr.transpose() << " " << acc_avr.transpose()
                  << endl;

        angvel_avr -= state_inout.bias_g;
        acc_avr = acc_avr * G_m_s2 / mean_acc_.norm() - state_inout.bias_a;

        if (head->header.stamp.toSec() < prop_beg_time) {
          dt = tail->header.stamp.toSec() - last_prop_end_time_;
          offs_t = tail->header.stamp.toSec() - prop_beg_time;
        } else if (i != v_imu.size() - 2) {
          dt = tail->header.stamp.toSec() - head->header.stamp.toSec();
          offs_t = tail->header.stamp.toSec() - prop_beg_time;
        } else {
          dt = prop_end_time - head->header.stamp.toSec();
          offs_t = prop_end_time - prop_beg_time;
        }

        dt_all += dt;

        // 状态预测
        M3D acc_avr_skew;
        M3D Exp_f = Exp(angvel_avr, dt);
        acc_avr_skew << SKEW_SYM_MATRX(acc_avr);

        F_x.setIdentity(); //雅可比矩阵
        cov_w.setZero(); //噪声

        //雅可比矩阵规则 R:0~2, p:3~5, τ:6, v:7~9, bg:10~12, ba:13~15, g:16~18
        F_x.block<3, 3>(0, 0) = Exp(angvel_avr, -dt);              // R对R
        if (ba_bg_est_en_) F_x.block<3, 3>(0, 10) = -eye3d_ * dt;  // R对bg
        // F_x.block<3,3>(3,0)  = R_imu * off_vel_skew * dt; //p对R
        F_x.block<3, 3>(3, 7) = eye3d_ * dt;                        // p对v
        F_x.block<3, 3>(7, 0) = -R_imu * acc_avr_skew * dt;         // v对R
        if (ba_bg_est_en_) F_x.block<3, 3>(7, 13) = -R_imu * dt;    // v对ba
        if (gravity_est_en_) F_x.block<3, 3>(7, 16) = eye3d_ * dt;  // v对g

        // tau = 1.0 / (0.25 * sin(2 * CV_PI * 0.5 * imu_time) + 0.75);
        // F_x(6,6) = 0.25 * 2 * CV_PI * 0.5 * cos(2 * CV_PI * 0.5 * imu_time) *
        // (-tau*tau); F_x(18,18) = 0.00001;
        if (exposure_estimate_en_) cov_w(6, 6) = cov_inv_expo_ * dt * dt;
        cov_w.block<3, 3>(0, 0).diagonal() = cov_gyr_ * dt * dt;
        cov_w.block<3, 3>(7, 7) =
            R_imu * cov_acc_.asDiagonal() * R_imu.transpose() * dt * dt;
        cov_w.block<3, 3>(10, 10).diagonal() =
            cov_bias_gyr_ * dt * dt;  // bias gyro covariance
        cov_w.block<3, 3>(13, 13).diagonal() =
            cov_bias_acc_ * dt * dt;  // bias acc covariance

        state_inout.cov = F_x * state_inout.cov * F_x.transpose() + cov_w;
        // state_inout.cov.block<18,18>(0,0) = F_x.block<18,18>(0,0) *
        // state_inout.cov.block<18,18>(0,0) * F_x.block<18,18>(0,0).transpose()
        // + cov_w.block<18,18>(0,0);

        // tau = tau + 0.25 * 2 * CV_PI * 0.5 * cos(2 * CV_PI * 0.5 * imu_time)
        // *
        // (-tau*tau) * dt;

        // tau = 1.0 / (0.25 * sin(2 * CV_PI * 0.5 * imu_time) + 0.75);

        /* propogation of IMU attitude */
        R_imu = R_imu * Exp_f;

        /* Specific acceleration (global frame) of IMU */
        acc_imu = R_imu * acc_avr + state_inout.gravity;

        /* propogation of IMU */
        pos_imu = pos_imu + vel_imu * dt + 0.5 * acc_imu * dt * dt;

        /* velocity of IMU */
        vel_imu = vel_imu + acc_imu * dt;

        /* save the poses at each IMU measurements */
        angvel_last_ = angvel_avr;
        acc_s_last_ = acc_imu;

        // cout<<setw(20)<<"offset_t: "<<offs_t<<"tail->header.stamp.toSec():
        // "<<tail->header.stamp.toSec()<<endl; printf("[ LIO Propagation ]
        // offs_t: %lf \n", offs_t);
        imu_pose_.push_back(
            set_pose6d(offs_t, acc_imu, angvel_avr, vel_imu, pos_imu, R_imu));
      }

      // unbiased_gyr = V3D(IMUpose.back().gyr[0], IMUpose.back().gyr[1],
      // IMUpose.back().gyr[2]); cout<<"prop end - start: "<<prop_end_time -
      // prop_beg_time<<" dt_all: "<<dt_all<<endl;
      lidar_meas.last_lio_update_time = prop_end_time;
      break;
  }

  state_inout.vel_end = vel_imu;
  state_inout.rot_end = R_imu;
  state_inout.pos_end = pos_imu;
  state_inout.inv_expo_time = tau;

  /*** calculated the pos and attitude prediction at the frame-end ***/
  // if (imu_end_time>prop_beg_time)
  // {
  //   double note = prop_end_time > imu_end_time ? 1.0 : -1.0;
  //   dt = note * (prop_end_time - imu_end_time);
  //   state_inout.vel_end = vel_imu + note * acc_imu * dt;
  //   state_inout.rot_end = R_imu * Exp(V3D(note * angvel_avr), dt);
  //   state_inout.pos_end = pos_imu + note * vel_imu * dt + note * 0.5 *
  //   acc_imu * dt * dt;
  // }
  // else
  // {
  //   double note = prop_end_time > prop_beg_time ? 1.0 : -1.0;
  //   dt = note * (prop_end_time - prop_beg_time);
  //   state_inout.vel_end = vel_imu + note * acc_imu * dt;
  //   state_inout.rot_end = R_imu * Exp(V3D(note * angvel_avr), dt);
  //   state_inout.pos_end = pos_imu + note * vel_imu * dt + note * 0.5 *
  //   acc_imu * dt * dt;
  // }

  // cout<<"[ Propagation ] output state: "<<state_inout.vel_end.transpose() <<
  // state_inout.pos_end.transpose()<<endl;

  last_imu_ = v_imu.back();
  last_prop_end_time_ = prop_end_time;

  // auto pos_liD_e = state_inout.pos_end + state_inout.rot_end *
  // Lid_offset_to_IMU; auto R_liD_e   = state_inout.rot_end * Lidar_R_to_IMU;

  if (pcl_wait_proc_.points.size() < 1) return;

  // 点云去畸变
  if (lidar_meas.lio_vio_flg == LIO) {
    auto it_pcl = pcl_wait_proc_.points.end() - 1;
    M3D extR_Ri(lidar_rot_to_imu_.transpose() *
                state_inout.rot_end.transpose());
    V3D exrR_extT(lidar_rot_to_imu_.transpose() * lidar_offset_to_imu_);
    for (auto it_kp = imu_pose_.end() - 1; it_kp != imu_pose_.begin();
         it_kp--) {
      auto head = it_kp - 1;
      auto tail = it_kp;
      R_imu << MAT_FROM_ARRAY(head->rot);
      acc_imu << VEC_FROM_ARRAY(head->acc);
      vel_imu << VEC_FROM_ARRAY(head->vel);
      pos_imu << VEC_FROM_ARRAY(head->pos);
      angvel_avr << VEC_FROM_ARRAY(head->gyr);


      for (; it_pcl->curvature / double(1000) > head->offset_time; it_pcl--) {
        dt = it_pcl->curvature / double(1000) - head->offset_time;

        /* Transform to the 'end' frame */
        M3D R_i(R_imu * Exp(angvel_avr, dt));
        V3D T_ei(pos_imu + vel_imu * dt + 0.5 * acc_imu * dt * dt -
                 state_inout.pos_end);

        V3D P_i(it_pcl->x, it_pcl->y, it_pcl->z);
        // V3D P_compensate = Lid_rot_to_IMU.transpose() *
        // (state_inout.rot_end.transpose() * (R_i * (Lid_rot_to_IMU * P_i +
        // Lid_offset_to_IMU) + T_ei) - Lid_offset_to_IMU);
        V3D P_compensate =
            (extR_Ri * (R_i * (lidar_rot_to_imu_ * P_i + lidar_offset_to_imu_) +
                        T_ei) -
             exrR_extT);

        /// save Undistorted points and their rotation
        it_pcl->x = P_compensate(0);
        it_pcl->y = P_compensate(1);
        it_pcl->z = P_compensate(2);

        if (it_pcl == pcl_wait_proc_.points.begin()) break;
      }
    }
    pcl_out = pcl_wait_proc_;
    pcl_wait_proc_.clear();
    imu_pose_.clear();
  }
}

void ImuProcess::Process2(LidarMeasureGroup &lidar_meas, StatesGroup &state,
                          PointCloudXYZIN::Ptr cur_pcl_un_) {
  double t1, t2, t3;
  t1 = omp_get_wtime();
  ROS_ASSERT(lidar_meas.lidar != nullptr);
  // 无IMU模式
  if (!imu_en_) {
    Forward_without_imu(lidar_meas, state, *cur_pcl_un_);
    return;
  }

  MeasureGroup meas = lidar_meas.measures.back();

  if (imu_need_init_) {
    double pcl_end_time =
        lidar_meas.lio_vio_flg == LIO ? meas.lio_time : meas.vio_time;

    if (meas.imu.empty()) {
      return;
    };
    /// The very first lidar frame
    IMU_init(meas, state, init_iter_num_);

    last_imu_ = meas.imu.back();

    if (init_iter_num_ > max_ini_count_) {
      // cov_acc *= pow(G_m_s2 / mean_acc.norm(), 2);
      imu_need_init_ = false;
      ROS_INFO(
          "IMU Initials: Gravity: %.4f %.4f %.4f %.4f; acc covarience: "
          "%.8f %.8f %.8f; gry covarience: %.8f %.8f %.8f \n",
          state.gravity[0], state.gravity[1], state.gravity[2],
          mean_acc_.norm(), cov_acc_[0], cov_acc_[1], cov_acc_[2], cov_gyr_[0],
          cov_gyr_[1], cov_gyr_[2]);
      ROS_INFO(
          "IMU Initials: ba covarience: %.8f %.8f %.8f; bg covarience: "
          "%.8f %.8f %.8f",
          cov_bias_acc_[0], cov_bias_acc_[1], cov_bias_acc_[2],
          cov_bias_gyr_[0], cov_bias_gyr_[1], cov_bias_gyr_[2]);
      fout_imu_.open(DEBUG_FILE_DIR("imu.txt"), ios::out);
    }

    return;
  }

  UndistortPcl(lidar_meas, state, *cur_pcl_un_);
}