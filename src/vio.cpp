/*
This file is part of FAST-LIVO2: Fast, Direct LiDAR-Inertial-Visual Odometry.

Developer: Chunran Zheng <zhengcr@connect.hku.hk>

For commercial use, please contact me at <zhengcr@connect.hku.hk> or
Prof. Fu Zhang at <fuzhang@hku.hk>.

This file is subject to the terms and conditions outlined in the 'LICENSE' file,
which is included as part of this source code package.
*/

#include "vio.h"

#include <glog/logging.h>

VIOManager::VIOManager() {
  // downSizeFilter.setLeafSize(0.2, 0.2, 0.2);
}

VIOManager::~VIOManager() {
  delete visual_submap_;
  for (auto &pair : warp_map_) delete pair.second;
  warp_map_.clear();
  for (auto &pair : vp_map_) delete pair.second->second;
  vp_map_.clear();
  // for (auto &pair : feat_map_) delete pair.second;
  // feat_map_.clear();
  vp_data_.clear();
}

void VIOManager::SetImuToLidarExtrinsic(const V3D &transl, const M3D &rot) {
  Pli_ = -rot.transpose() * transl;
  Rli_ = rot.transpose();
  se3_imu_lidar_ = SE3(Mat3ToSO3(Rli_), Pli_);
}

void VIOManager::SetLidarToCameraExtrinsic(std::vector<double> &R,
                                           std::vector<double> &P) {
  Rcl_ << MAT_FROM_ARRAY(R);
  Pcl_ << VEC_FROM_ARRAY(P);
  se3_lidar_cam_ = SE3(Mat3ToSO3(Rcl_), Pcl_);
}

void VIOManager::InitializeVIO() {
  visual_submap_ = new SubSparseMap;

  fx_ = cam_->fx();
  fy_ = cam_->fy();
  cx_ = cam_->cx();
  cy_ = cam_->cy();
  image_resize_factor_ = cam_->scale();

  printf("intrinsic: %.6lf, %.6lf, %.6lf, %.6lf\n", fx_, fy_, cx_, cy_);

  width_ = cam_->width();
  height_ = cam_->height();

  printf("width: %d, height: %d, scale: %f\n", width_, height_,
         image_resize_factor_);
  Rci_ = Rcl_ * Rli_;
  Pci_ = Rcl_ * Pli_ + Pcl_;

  V3D Pic;
  M3D tmp;
  Jdphi_dR_ = Rci_;
  Pic = -Rci_.transpose() * Pci_;
  tmp << SKEW_SYM_MATRX(Pic);
  Jdp_dR_ = -Rci_ * tmp;

  if (grid_size_ > 10) {
    grid_n_width_ = ceil(static_cast<double>(width_ / grid_size_));
    grid_n_height_ = ceil(static_cast<double>(height_ / grid_size_));
  } else {
    grid_size_ = static_cast<int>(height_ / grid_n_height_);
    grid_n_height_ = ceil(static_cast<double>(height_ / grid_size_));
    grid_n_width_ = ceil(static_cast<double>(width_ / grid_size_));
  }
  length_ = grid_n_width_ * grid_n_height_;

  if (raycast_en_) {
    // cv::Mat img_test = cv::Mat::zeros(height, width, CV_8UC1);
    // uchar* it = (uchar*)img_test.data;

    border_flag_.resize(length_, 0);

    std::vector<std::vector<V3D>>().swap(rays_with_sample_points_);
    rays_with_sample_points_.reserve(length_);
    printf("grid_size: %d, grid_n_height: %d, grid_n_width: %d, length: %d\n",
           grid_size_, grid_n_height_, grid_n_width_, length_);

    float d_min = 0.1;
    float d_max = 3.0;
    float step = 0.2;
    for (int grid_row = 1; grid_row <= grid_n_height_; grid_row++) {
      for (int grid_col = 1; grid_col <= grid_n_width_; grid_col++) {
        std::vector<V3D> SamplePointsEachGrid;
        int index = (grid_row - 1) * grid_n_width_ + grid_col - 1;

        if (grid_row == 1 || grid_col == 1 || grid_row == grid_n_height_ ||
            grid_col == grid_n_width_)
          border_flag_[index] = 1;

        int u = grid_size_ / 2 + (grid_col - 1) * grid_size_;
        int v = grid_size_ / 2 + (grid_row - 1) * grid_size_;
        // it[ u + v * width ] = 255;
        for (float d_temp = d_min; d_temp <= d_max; d_temp += step) {
          V3D xyz;
          xyz = cam_->cam2world(u, v);
          xyz *= d_temp / xyz[2];
          // xyz[0] = (u - cx) / fx * d_temp;
          // xyz[1] = (v - cy) / fy * d_temp;
          // xyz[2] = d_temp;
          SamplePointsEachGrid.push_back(xyz);
        }
        rays_with_sample_points_.push_back(SamplePointsEachGrid);
      }
    }
    // printf("rays_with_sample_points: %d, RaysWithSamplePointsCapacity: %d,
    // rays_with_sample_points[0].capacity(): %d, rays_with_sample_points[0]:
    // %d\n", rays_with_sample_points.size(),
    // rays_with_sample_points.capacity(),
    // rays_with_sample_points[0].capacity(),
    // rays_with_sample_points[0].size()); for (const auto & it :
    // rays_with_sample_points[0]) cout << it.transpose() << endl;
    // cv::imshow("img_test", img_test);
    // cv::waitKey(1);
  }

  if (colmap_output_en_) {
    pinhole_cam_ = dynamic_cast<vk::PinholeCamera *>(cam_);
    fout_colmap_.open(DEBUG_FILE_DIR("Colmap/sparse/0/images.txt"),
                      std::ios::out);
    fout_colmap_ << "# Image list with two lines of data per image:\n";
    fout_colmap_
        << "#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n";
    fout_colmap_ << "#   POINTS2D[] as (X, Y, POINT3D_ID)\n";
    fout_camera_.open(DEBUG_FILE_DIR("Colmap/sparse/0/cameras.txt"),
                      std::ios::out);
    fout_camera_ << "# Camera list with one line of data per camera:\n";
    fout_camera_ << "#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n";
    fout_camera_ << "1 PINHOLE " << width_ << " " << height_ << " "
                 << std::fixed << std::setprecision(6)  // 控制浮点数精度为10位
                 << fx_ << " " << fy_ << " " << cx_ << " " << cy_ << std::endl;
    fout_camera_.close();
  }
  grid_num_.resize(length_);
  map_index_.resize(length_);
  map_dist_.resize(length_);
  update_flag_.resize(length_);
  scan_value_.resize(length_);

  patch_size_total_ = patch_size_ * patch_size_;
  patch_size_half_ = static_cast<int>(patch_size_ / 2);
  patch_buffer_.resize(patch_size_total_);
  warp_len_ = patch_size_total_ * patch_pyrimid_level_;
  border_ = (patch_size_half_ + 1) * (1 << patch_pyrimid_level_);

  retrieve_voxel_points_.reserve(length_);
  append_voxel_points_.reserve(length_);

  sub_feat_map_.clear();
}

void VIOManager::ResetGrid() {
  fill(grid_num_.begin(), grid_num_.end(), TYPE_UNKNOWN);
  fill(map_index_.begin(), map_index_.end(), 0);
  fill(map_dist_.begin(), map_dist_.end(), 10000.0f);
  fill(update_flag_.begin(), update_flag_.end(), 0);
  fill(scan_value_.begin(), scan_value_.end(), 0.0f);

  retrieve_voxel_points_.clear();
  retrieve_voxel_points_.resize(length_);

  append_voxel_points_.clear();
  append_voxel_points_.resize(length_);

  total_points_ = 0;
}

// void VIOManager::resetRvizDisplay()
// {
// sub_map_ray.clear();
// sub_map_ray_fov.clear();
// visual_sub_map_cur.clear();
// visual_converged_point.clear();
// map_cur_frame.clear();
// sample_points.clear();
// }

void VIOManager::ComputeProjectionJacobian(V3D p, MD(2, 3) & J) {
  // 视觉SLAM十四讲公式(8.18)
  const double x = p[0];
  const double y = p[1];
  const double z_inv = 1. / p[2];
  const double z_inv_2 = z_inv * z_inv;
  J(0, 0) = fx_ * z_inv;
  J(0, 1) = 0.0;
  J(0, 2) = -fx_ * x * z_inv_2;
  J(1, 0) = 0.0;
  J(1, 1) = fy_ * z_inv;
  J(1, 2) = -fy_ * y * z_inv_2;
}

void VIOManager::GetImagePatch(cv::Mat img, V2D pc, float *patch_tmp,
                               int level) {
  const float u_ref = pc[0];
  const float v_ref = pc[1];
  const int scale = (1 << level);
  const int u_ref_i = floorf(pc[0] / scale) * scale;
  const int v_ref_i = floorf(pc[1] / scale) * scale;
  const float subpix_u_ref = (u_ref - u_ref_i) / scale;
  const float subpix_v_ref = (v_ref - v_ref_i) / scale;
  const float w_ref_tl = (1.0 - subpix_u_ref) * (1.0 - subpix_v_ref);
  const float w_ref_tr = subpix_u_ref * (1.0 - subpix_v_ref);
  const float w_ref_bl = (1.0 - subpix_u_ref) * subpix_v_ref;
  const float w_ref_br = subpix_u_ref * subpix_v_ref;
  for (int x = 0; x < patch_size_; x++) {
    uint8_t *img_ptr =
        (uint8_t *)img.data +
        (v_ref_i - patch_size_half_ * scale + x * scale) * width_ +
        (u_ref_i - patch_size_half_ * scale);
    for (int y = 0; y < patch_size_; y++, img_ptr += scale) {
      patch_tmp[patch_size_total_ * level + x * patch_size_ + y] =
          w_ref_tl * img_ptr[0] + w_ref_tr * img_ptr[scale] +
          w_ref_bl * img_ptr[scale * width_] +
          w_ref_br * img_ptr[scale * width_ + scale];
    }
  }
}

#if 0
void VIOManager::InsertPointIntoFeatureMap(VisualPoint *pt_new) {
  V3D pt_w(pt_new->pos_[0], pt_new->pos_[1], pt_new->pos_[2]);
  double voxel_size = 0.5;
  float loc_xyz[3];
  for (int j = 0; j < 3; j++) {
    loc_xyz[j] = pt_w[j] / voxel_size;
    if (loc_xyz[j] < 0) {
      loc_xyz[j] -= 1.0;
    }
  }
  VOXEL_LOCATION position((int64_t)loc_xyz[0], (int64_t)loc_xyz[1],
                          (int64_t)loc_xyz[2]);
  auto iter = feat_map_.find(position);
  if (iter != feat_map_.end()) {
    iter->second->voxel_points.push_back(pt_new);
    iter->second->count++;
  } else {
    VOXEL_POINTS *ot = new VOXEL_POINTS(0);
    ot->voxel_points.push_back(pt_new);
    feat_map_[position] = ot;
  }
}
#endif

void VIOManager::InsertPointIntoFeatureMapLRU(VisualPoint *pt_new) {
  V3D pt_w(pt_new->pos_[0], pt_new->pos_[1], pt_new->pos_[2]);
  double voxel_size = 0.5;
  float loc_xyz[3];
  for (int j = 0; j < 3; j++) {
    loc_xyz[j] = pt_w[j] / voxel_size;
    if (loc_xyz[j] < 0) {
      loc_xyz[j] -= 1.0;
    }
  }
  VOXEL_LOCATION position((int64_t)loc_xyz[0], (int64_t)loc_xyz[1],
                          (int64_t)loc_xyz[2]);
  auto iter = vp_map_.find(position);
  if (iter != vp_map_.end()) {
    iter->second->second->voxel_points.push_back(pt_new);
    iter->second->second->count++;
    // 更新的放至最前
    vp_data_.splice(vp_data_.begin(), vp_data_, iter->second);
    iter->second = vp_data_.begin();
  } else {
    VOXEL_POINTS *ot = new VOXEL_POINTS(0);
    ot->voxel_points.push_back(pt_new);
    vp_data_.push_front({position, {ot}});
    vp_map_.insert({position, vp_data_.begin()});

    // LRU
    if (vp_data_.size() >= lru_size_) {
      // 删除一个尾部的数据
      vp_map_.erase(vp_data_.back().first);
      delete vp_data_.back().second;
      vp_data_.pop_back();
    }
  }
}

void VIOManager::GetWarpMatrixAffineHomography(
    const vk::AbstractCamera &cam, const V2D &px_ref, const V3D &xyz_ref,
    const V3D &normal_ref, const SE3 &T_cur_ref, const int level_ref,
    Eigen::Matrix2d &A_cur_ref) {
  // 构建单应矩阵
  const V3D t = T_cur_ref.inverse().translation();
  const Eigen::Matrix3d H_cur_ref =
      T_cur_ref.rotationMatrix() *
      (normal_ref.dot(xyz_ref) * Eigen::Matrix3d::Identity() -
       t * normal_ref.transpose());
  // 单应投影计算仿射变换矩阵A_ref_cur
  const int kHalfPatchSize = 4;
  V3D f_du_ref(cam.cam2world(px_ref + Eigen::Vector2d(kHalfPatchSize, 0) *
                                          (1 << level_ref)));
  V3D f_dv_ref(cam.cam2world(px_ref + Eigen::Vector2d(0, kHalfPatchSize) *
                                          (1 << level_ref)));
  //   f_du_ref = f_du_ref/f_du_ref[2];
  //   f_dv_ref = f_dv_ref/f_dv_ref[2];
  const V3D f_cur(H_cur_ref * xyz_ref);
  const V3D f_du_cur = H_cur_ref * f_du_ref;
  const V3D f_dv_cur = H_cur_ref * f_dv_ref;
  V2D px_cur(cam.world2cam(f_cur));
  V2D px_du_cur(cam.world2cam(f_du_cur));
  V2D px_dv_cur(cam.world2cam(f_dv_cur));
  A_cur_ref.col(0) = (px_du_cur - px_cur) / kHalfPatchSize;
  A_cur_ref.col(1) = (px_dv_cur - px_cur) / kHalfPatchSize;
}

void VIOManager::GetWarpMatrixAffine(
    const vk::AbstractCamera &cam, const Eigen::Vector2d &px_ref,
    const Eigen::Vector3d &f_ref, const double depth_ref, const SE3 &T_cur_ref,
    const int level_ref, const int pyramid_level, const int halfpatch_size,
    Eigen::Matrix2d &A_cur_ref) {
  // Compute affine warp matrix A_ref_cur
  const Eigen::Vector3d xyz_ref(f_ref * depth_ref);
  Eigen::Vector3d xyz_du_ref(
      cam.cam2world(px_ref + Eigen::Vector2d(halfpatch_size, 0) *
                                 (1 << level_ref) * (1 << pyramid_level)));
  Eigen::Vector3d xyz_dv_ref(
      cam.cam2world(px_ref + Eigen::Vector2d(0, halfpatch_size) *
                                 (1 << level_ref) * (1 << pyramid_level)));
  xyz_du_ref *= xyz_ref[2] / xyz_du_ref[2];
  xyz_dv_ref *= xyz_ref[2] / xyz_dv_ref[2];
  const Eigen::Vector2d px_cur(cam.world2cam(T_cur_ref * (xyz_ref)));
  const Eigen::Vector2d px_du(cam.world2cam(T_cur_ref * (xyz_du_ref)));
  const Eigen::Vector2d px_dv(cam.world2cam(T_cur_ref * (xyz_dv_ref)));
  A_cur_ref.col(0) = (px_du - px_cur) / halfpatch_size;
  A_cur_ref.col(1) = (px_dv - px_cur) / halfpatch_size;
}

void VIOManager::WarpAffine(const Eigen::Matrix2d &A_cur_ref,
                            const cv::Mat &img_ref,
                            const Eigen::Vector2d &px_ref, const int level_ref,
                            const int search_level, const int pyramid_level,
                            const int halfpatch_size, float *patch) {
  const int patch_size = halfpatch_size * 2;
  // 参考帧到当前帧的变换
  const Eigen::Matrix2f A_ref_cur = A_cur_ref.inverse().cast<float>();
  if (isnan(A_ref_cur(0, 0))) {
    printf("Affine warp is NaN, probably camera has no translation\n");  // TODO
    return;
  }

  float *patch_ptr = patch;
  for (int y = 0; y < patch_size; ++y) {
    for (int x = 0; x < patch_size; ++x)  //, ++patch_ptr)
    {
      Eigen::Vector2f px_patch(x - halfpatch_size, y - halfpatch_size);
      px_patch *= (1 << search_level);
      px_patch *= (1 << pyramid_level);
      const Eigen::Vector2f px(A_ref_cur * px_patch + px_ref.cast<float>());
      if (px[0] < 0 || px[1] < 0 || px[0] >= img_ref.cols - 1 ||
          px[1] >= img_ref.rows - 1)
        patch_ptr[patch_size_total_ * pyramid_level + y * patch_size + x] = 0;
      else
        patch_ptr[patch_size_total_ * pyramid_level + y * patch_size + x] =
            (float)vk::interpolateMat_8u(img_ref, px[0], px[1]);
    }
  }
}

int VIOManager::GetBestSearchLevel(const Eigen::Matrix2d &A_cur_ref,
                                   const int max_level) {
  // Compute patch level in other image
  int search_level = 0;
  double D = A_cur_ref.determinant();
  while (D > 3.0 && search_level < max_level) {
    search_level += 1;
    D *= 0.25;
  }
  return search_level;
}

double VIOManager::CalculateNCC(float *ref_patch, float *cur_patch,
                                int patch_size) {
  double sum_ref = std::accumulate(ref_patch, ref_patch + patch_size, 0.0);
  double mean_ref = sum_ref / patch_size;

  double sum_cur = std::accumulate(cur_patch, cur_patch + patch_size, 0.0);
  double mean_curr = sum_cur / patch_size;

  double numerator = 0, demoniator1 = 0, demoniator2 = 0;
  for (int i = 0; i < patch_size; i++) {
    double n = (ref_patch[i] - mean_ref) * (cur_patch[i] - mean_curr);
    numerator += n;
    demoniator1 += (ref_patch[i] - mean_ref) * (ref_patch[i] - mean_ref);
    demoniator2 += (cur_patch[i] - mean_curr) * (cur_patch[i] - mean_curr);
  }
  return numerator / sqrt(demoniator1 * demoniator2 + 1e-10);
}

#if 0
void VIOManager::RetrieveFromVisualSparseMap(
    cv::Mat img, std::vector<pointWithVar> &pv_list,
    const std::unordered_map<VOXEL_LOCATION, VoxelOctoTree *> &voxel_map) {
  if (feat_map_.size() <= 0) return;
  double ts0 = omp_get_wtime();

  // resetRvizDisplay();
  visual_submap_->reset();

  // 控制是否添加来自前一帧的视觉子地图
  sub_feat_map_.clear();

  float voxel_size = 0.5;

  // 法向量默认开启
  if (!normal_en_) warp_map_.clear();

  cv::Mat depth_img = cv::Mat::zeros(height_, width_, CV_32FC1);
  // data连续内存
  float *it = (float *)depth_img.data;

  int loc_xyz[3];

  // 遍历点云数据，生成深度图
  for (int i = 0; i < pv_list.size(); i++) {
    V3D pt_w = pv_list[i].point_w;

    for (int j = 0; j < 3; j++) {
      loc_xyz[j] = floor(pt_w[j] / voxel_size);
      if (loc_xyz[j] < 0) {
        loc_xyz[j] -= 1.0;
      }
    }
    VOXEL_LOCATION position(loc_xyz[0], loc_xyz[1], loc_xyz[2]);

    // 添加当前帧点云对应的体素
    auto iter = sub_feat_map_.find(position);
    if (iter == sub_feat_map_.end()) {
      sub_feat_map_[position] = 0;
    } else {
      iter->second = 0;
    }

    V3D pt_c(new_frame_->w2f(pt_w));

    if (pt_c[2] > 0) {
      V2D px;
      px = new_frame_->cam_->world2cam(pt_c);
      // 记录深度
      if (new_frame_->cam_->isInFrame(px.cast<int>(), border_)) {
        float depth = pt_c[2];
        int col = int(px[0]);
        int row = int(px[1]);
        it[width_ * row + col] = depth;
      }
    }
  }

  // 筛选和更新视觉子地图
  std::vector<VOXEL_LOCATION> DeleteKeyList;  // sub_feat_map_中无用的数据

  for (auto &iter : sub_feat_map_) {
    VOXEL_LOCATION position = iter.first;

    auto corre_voxel = feat_map_.find(position);

    if (corre_voxel != feat_map_.end()) {
      bool voxel_in_fov = false;
      std::vector<VisualPoint *> &voxel_points =
          corre_voxel->second->voxel_points;
      int voxel_num = voxel_points.size();

      for (int i = 0; i < voxel_num; i++) {
        VisualPoint *pt = voxel_points[i];
        if (pt == nullptr) continue;
        // 没有相关图像块，跳过
        if (pt->obs_.size() == 0) continue;

        V3D norm_vec(new_frame_->T_f_w_.rotationMatrix() * pt->normal_);
        V3D dir(new_frame_->T_f_w_ * pt->pos_);
        if (dir[2] < 0) continue;

        V2D pc(new_frame_->w2c(pt->pos_));
        if (new_frame_->cam_->isInFrame(pc.cast<int>(), border_)) {
          voxel_in_fov = true;
          int index = static_cast<int>(pc[1] / grid_size_) * grid_n_width_ +
                      static_cast<int>(pc[0] / grid_size_);
          grid_num_[index] = TYPE_MAP;
          Eigen::Vector3d obs_vec(new_frame_->pos() - pt->pos_);
          float cur_dist = obs_vec.norm();
          if (cur_dist <= map_dist_[index]) {
            map_dist_[index] = cur_dist;
            retrieve_voxel_points_[index] = pt;
          }
        }
      }
      if (!voxel_in_fov) {
        DeleteKeyList.push_back(position);
      }
    }
  }

  // 光线追踪模型，用来补充特征，默认关闭
  if (raycast_en_) {
    for (int i = 0; i < length_; i++) {
      if (grid_num_[i] == TYPE_MAP || border_flag_[i] == 1) continue;

      // int row = static_cast<int>(i / grid_n_width) * grid_size + grid_size
      /
      // 2; int col = (i - static_cast<int>(i / grid_n_width) * grid_n_width)
      *
      // grid_size + grid_size / 2;

      // cv::circle(img_cp, cv::Point2f(col, row), 3, cv::Scalar(255, 255,
      0),
      // -1, 8);

      // vector<V3D> sample_points_temp;
      // bool add_sample = false;

      for (const auto &it : rays_with_sample_points_[i]) {
        V3D sample_point_w = new_frame_->f2w(it);
        // sample_points_temp.push_back(sample_point_w);

        for (int j = 0; j < 3; j++) {
          loc_xyz[j] = floor(sample_point_w[j] / voxel_size);
          if (loc_xyz[j] < 0) {
            loc_xyz[j] -= 1.0;
          }
        }

        VOXEL_LOCATION sample_pos(loc_xyz[0], loc_xyz[1], loc_xyz[2]);

        auto corre_sub_feat_map = sub_feat_map_.find(sample_pos);
        if (corre_sub_feat_map != sub_feat_map_.end()) break;

        auto corre_feat_map = feat_map_.find(sample_pos);
        if (corre_feat_map != feat_map_.end()) {
          bool voxel_in_fov = false;

          std::vector<VisualPoint *> &voxel_points =
              corre_feat_map->second->voxel_points;
          int voxel_num = voxel_points.size();
          if (voxel_num == 0) continue;

          for (int j = 0; j < voxel_num; j++) {
            VisualPoint *pt = voxel_points[j];

            if (pt == nullptr) continue;
            if (pt->obs_.size() == 0) continue;

            // sub_map_ray.push_back(pt); // cloud_visual_sub_map
            // add_sample = true;

            V3D norm_vec(new_frame_->T_f_w_.rotationMatrix() * pt->normal_);
            V3D dir(new_frame_->T_f_w_ * pt->pos_);
            if (dir[2] < 0) continue;
            dir.normalize();
            // if (dir.dot(norm_vec) <= 0.17) continue; // 0.34 70 degree
            0.17
            // 80 degree 0.08 85 degree

            V2D pc(new_frame_->w2c(pt->pos_));

            if (new_frame_->cam_->isInFrame(pc.cast<int>(), border_)) {
              // cv::circle(img_cp, cv::Point2f(pc[0], pc[1]), 3,
              // cv::Scalar(255, 255, 0), -1, 8);
              sub_map_ray_fov.push_back(pt);

              voxel_in_fov = true;
              int index = static_cast<int>(pc[1] / grid_size_) *
              grid_n_width_ +
                          static_cast<int>(pc[0] / grid_size_);
              grid_num_[index] = TYPE_MAP;
              Eigen::Vector3d obs_vec(new_frame_->pos() - pt->pos_);

              float cur_dist = obs_vec.norm();

              if (cur_dist <= map_dist_[index]) {
                map_dist_[index] = cur_dist;
                retrieve_voxel_points_[index] = pt;
              }
            }
          }

          if (voxel_in_fov) sub_feat_map_[sample_pos] = 0;
          break;
        } else {
          VOXEL_LOCATION sample_pos(loc_xyz[0], loc_xyz[1], loc_xyz[2]);
          auto iter = voxel_map.find(sample_pos);
          if (iter != voxel_map.end()) {
            VoxelOctoTree *current_octo;
            current_octo = iter->second->FindCorrespond(sample_point_w);
            if (current_octo->plane_ptr_->is_plane_) {
              pointWithVar plane_center;
              VoxelPlane &plane = *current_octo->plane_ptr_;
              plane_center.point_w = plane.center_;
              plane_center.normal = plane.normal_;
              visual_submap_->add_from_voxel_map.push_back(plane_center);
              break;
            }
          }
        }
      }
      // if(add_sample) sample_points.push_back(sample_points_temp);
    }
  }

  for (auto &key : DeleteKeyList) {
    sub_feat_map_.erase(key);
  }

  for (int i = 0; i < length_; i++) {
    if (grid_num_[i] == TYPE_MAP) {
      VisualPoint *pt = retrieve_voxel_points_[i];
      // visual_sub_map_cur.push_back(pt); // before

      V2D pc(new_frame_->w2c(pt->pos_));

      // cv::circle(img_cp, cv::Point2f(pc[0], pc[1]), 3, cv::Scalar(0, 0,
      255),
      // -1, 8); // Green Sparse Align tracked

      V3D pt_cam(new_frame_->w2f(pt->pos_));
      // 深度连续性检查
      bool depth_discontinous = false;
      for (int u = -patch_size_half_; u <= patch_size_half_; u++) {
        for (int v = -patch_size_half_; v <= patch_size_half_; v++) {
          if (u == 0 && v == 0) continue;

          float depth = it[width_ * (v + int(pc[1])) + u + int(pc[0])];

          if (depth == 0.) continue;

          double delta_dist = abs(pt_cam[2] - depth);

          if (delta_dist > 0.5) {
            depth_discontinous = true;
            break;
          }
        }
        if (depth_discontinous) break;
      }
      if (depth_discontinous) continue;

      Feature *ref_ftr;
      std::vector<float> patch_wrap(warp_len_);

      int search_level;
      Eigen::Matrix2d A_cur_ref_zero;

      if (!pt->is_normal_initialized_) continue;

      if (normal_en_) {
        float phtometric_errors_min = std::numeric_limits<float>::max();

        if (pt->obs_.size() == 1) {
          // 只有一个关联图像块，将其选为参考图像块
          ref_ftr = *pt->obs_.begin();
          pt->ref_patch = ref_ftr;
          pt->has_ref_patch_ = true;
        } else if (!pt->has_ref_patch_) {
          for (auto it = pt->obs_.begin(), ite = pt->obs_.end(); it != ite;
               ++it) {
            // 不止一个关联图像块，选择光度误差最小的为参考图像块
            Feature *ref_patch_temp = *it;
            float *patch_temp = ref_patch_temp->patch_;
            float photometric_errors = 0.0;
            int count = 0;
            for (auto itm = pt->obs_.begin(), itme = pt->obs_.end();
                 itm != itme; ++itm) {
              if ((*itm)->id_ == ref_patch_temp->id_) continue;
              float *patch_cache = (*itm)->patch_;

              for (int ind = 0; ind < patch_size_total_; ind++) {
                photometric_errors += (patch_temp[ind] - patch_cache[ind]) *
                                      (patch_temp[ind] - patch_cache[ind]);
              }
              count++;
            }
            photometric_errors = photometric_errors / count;
            if (photometric_errors < phtometric_errors_min) {
              phtometric_errors_min = photometric_errors;
              ref_ftr = ref_patch_temp;
            }
          }
          pt->ref_patch = ref_ftr;
          pt->has_ref_patch_ = true;
        } else {
          ref_ftr = pt->ref_patch;
        }
      } else {
        if (!pt->getCloseViewObs(new_frame_->pos(), ref_ftr, pc)) continue;
      }

      // 计算仿射变换矩阵和搜索层级
      if (normal_en_) {
        V3D norm_vec =
            (ref_ftr->T_f_w_.rotationMatrix() * pt->normal_).normalized();

        V3D pf(ref_ftr->T_f_w_ * pt->pos_);

        SE3 T_cur_ref = new_frame_->T_f_w_ * ref_ftr->T_f_w_.inverse();

        GetWarpMatrixAffineHomography(*cam_, ref_ftr->px_, pf, norm_vec,
                                      T_cur_ref, 0, A_cur_ref_zero);

        search_level = GetBestSearchLevel(A_cur_ref_zero, 2);
      } else {
        auto iter_warp = warp_map_.find(ref_ftr->id_);
        if (iter_warp != warp_map_.end()) {
          search_level = iter_warp->second->search_level;
          A_cur_ref_zero = iter_warp->second->A_cur_ref;
        } else {
          GetWarpMatrixAffine(*cam_, ref_ftr->px_, ref_ftr->f_,
                              (ref_ftr->pos() - pt->pos_).norm(),
                              new_frame_->T_f_w_ * ref_ftr->T_f_w_.inverse(),
                              ref_ftr->level_, 0, patch_size_half_,
                              A_cur_ref_zero);

          search_level = GetBestSearchLevel(A_cur_ref_zero, 2);

          Warp *ot = new Warp(search_level, A_cur_ref_zero);
          warp_map_[ref_ftr->id_] = ot;
        }
      }

      for (int pyramid_level = 0; pyramid_level <= patch_pyrimid_level_ - 1;
           pyramid_level++) {
        WarpAffine(A_cur_ref_zero, ref_ftr->img_, ref_ftr->px_,
        ref_ftr->level_,
                   search_level, pyramid_level, patch_size_half_,
                   patch_wrap.data());
      }
      // 获取当前帧特征点对应的patch
      GetImagePatch(img, pc, patch_buffer_.data(), 0);

      float error = 0.0;
      for (int ind = 0; ind < patch_size_total_; ind++) {
        error += (ref_ftr->inv_expo_time_ * patch_wrap[ind] -
                  state_->inv_expo_time * patch_buffer_[ind]) *
                 (ref_ftr->inv_expo_time_ * patch_wrap[ind] -
                  state_->inv_expo_time * patch_buffer_[ind]);
      }

      // 默认关闭
      if (ncc_en_) {
        double ncc = CalculateNCC(patch_wrap.data(), patch_buffer_.data(),
                                  patch_size_total_);
        if (ncc < ncc_thre_) {
          // grid_num[i] = TYPE_UNKNOWN;
          continue;
        }
      }

      if (error > outlier_threshold_ * patch_size_total_) continue;

      visual_submap_->voxel_points.push_back(pt);
      visual_submap_->propa_errors.push_back(error);
      visual_submap_->search_levels.push_back(search_level);
      visual_submap_->errors.push_back(error);
      visual_submap_->warp_patch.push_back(patch_wrap);
      visual_submap_->inv_expo_list.push_back(ref_ftr->inv_expo_time_);
    }
  }
  total_points_ = visual_submap_->voxel_points.size();

  printf("[ VIO ] Retrieve %d points from visual sparse map\n",
  total_points_);
}
#endif

void VIOManager::RetrieveFromVisualSparseMapLRU(
    cv::Mat img, std::vector<pointWithVar> &pv_list,
    const std::unordered_map<VOXEL_LOCATION,
                             typename std::list<VMData>::iterator> &vm_map) {
  if (vp_map_.size() <= 0) return;
  double ts0 = omp_get_wtime();

  // resetRvizDisplay();
  visual_submap_->reset();

  // 控制是否添加来自前一帧的视觉子地图
  sub_feat_map_.clear();

  float voxel_size = 0.5;

  // 法向量默认开启
  if (!normal_en_) warp_map_.clear();

  cv::Mat depth_img = cv::Mat::zeros(height_, width_, CV_32FC1);
  // data连续内存
  float *it = (float *)depth_img.data;

  int loc_xyz[3];

  // 遍历点云数据，生成深度图
  for (int i = 0; i < pv_list.size(); i++) {
    V3D pt_w = pv_list[i].point_w;

    for (int j = 0; j < 3; j++) {
      loc_xyz[j] = floor(pt_w[j] / voxel_size);
      if (loc_xyz[j] < 0) {
        loc_xyz[j] -= 1.0;
      }
    }
    VOXEL_LOCATION position(loc_xyz[0], loc_xyz[1], loc_xyz[2]);

    // 添加当前帧点云对应的体素
    auto iter = sub_feat_map_.find(position);
    if (iter == sub_feat_map_.end()) {
      sub_feat_map_[position] = 0;
    } else {
      iter->second = 0;
    }

    V3D pt_c(new_frame_->w2f(pt_w));

    if (pt_c[2] > 0) {
      V2D px;
      px = new_frame_->cam_->world2cam(pt_c);
      // 记录深度
      if (new_frame_->cam_->isInFrame(px.cast<int>(), border_)) {
        float depth = pt_c[2];
        int col = int(px[0]);
        int row = int(px[1]);
        it[width_ * row + col] = depth;
      }
    }
  }

  // 筛选和更新视觉子地图
  std::vector<VOXEL_LOCATION> DeleteKeyList;  // sub_feat_map_中无用的数据

  for (auto &iter : sub_feat_map_) {
    VOXEL_LOCATION position = iter.first;

    auto corre_voxel = vp_map_.find(position);

    if (corre_voxel != vp_map_.end()) {
      bool voxel_in_fov = false;
      std::vector<VisualPoint *> &voxel_points =
          corre_voxel->second->second->voxel_points;
      int voxel_num = voxel_points.size();

      for (int i = 0; i < voxel_num; i++) {
        VisualPoint *pt = voxel_points[i];
        if (pt == nullptr) continue;
        // 没有相关图像块，跳过
        if (pt->obs_.size() == 0) continue;

        V3D norm_vec(new_frame_->T_f_w_.rotationMatrix() * pt->normal_);
        V3D dir(new_frame_->T_f_w_ * pt->pos_);
        if (dir[2] < 0) continue;

        V2D pc(new_frame_->w2c(pt->pos_));
        if (new_frame_->cam_->isInFrame(pc.cast<int>(), border_)) {
          voxel_in_fov = true;
          int index = static_cast<int>(pc[1] / grid_size_) * grid_n_width_ +
                      static_cast<int>(pc[0] / grid_size_);
          grid_num_[index] = TYPE_MAP;
          Eigen::Vector3d obs_vec(new_frame_->pos() - pt->pos_);
          float cur_dist = obs_vec.norm();
          if (cur_dist <= map_dist_[index]) {
            map_dist_[index] = cur_dist;
            retrieve_voxel_points_[index] = pt;
          }
        }
      }
      if (!voxel_in_fov) {
        DeleteKeyList.push_back(position);
      }
    }
  }

  // 光线追踪模型，用来补充特征，默认关闭
  if (raycast_en_) {
    for (int i = 0; i < length_; i++) {
      if (grid_num_[i] == TYPE_MAP || border_flag_[i] == 1) continue;

      // int row = static_cast<int>(i / grid_n_width) * grid_size + grid_size /
      // 2; int col = (i - static_cast<int>(i / grid_n_width) * grid_n_width) *
      // grid_size + grid_size / 2;

      // cv::circle(img_cp, cv::Point2f(col, row), 3, cv::Scalar(255, 255, 0),
      // -1, 8);

      // vector<V3D> sample_points_temp;
      // bool add_sample = false;

      for (const auto &it : rays_with_sample_points_[i]) {
        V3D sample_point_w = new_frame_->f2w(it);
        // sample_points_temp.push_back(sample_point_w);

        for (int j = 0; j < 3; j++) {
          loc_xyz[j] = floor(sample_point_w[j] / voxel_size);
          if (loc_xyz[j] < 0) {
            loc_xyz[j] -= 1.0;
          }
        }

        VOXEL_LOCATION sample_pos(loc_xyz[0], loc_xyz[1], loc_xyz[2]);

        auto corre_sub_feat_map = sub_feat_map_.find(sample_pos);
        if (corre_sub_feat_map != sub_feat_map_.end()) break;

        auto corre_feat_map = vp_map_.find(sample_pos);
        if (corre_feat_map != vp_map_.end()) {
          bool voxel_in_fov = false;

          std::vector<VisualPoint *> &voxel_points =
              corre_feat_map->second->second->voxel_points;
          int voxel_num = voxel_points.size();
          if (voxel_num == 0) continue;

          for (int j = 0; j < voxel_num; j++) {
            VisualPoint *pt = voxel_points[j];

            if (pt == nullptr) continue;
            if (pt->obs_.size() == 0) continue;

            // sub_map_ray.push_back(pt); // cloud_visual_sub_map
            // add_sample = true;

            V3D norm_vec(new_frame_->T_f_w_.rotationMatrix() * pt->normal_);
            V3D dir(new_frame_->T_f_w_ * pt->pos_);
            if (dir[2] < 0) continue;
            dir.normalize();
            // if (dir.dot(norm_vec) <= 0.17) continue; // 0.34 70 degree 0.17
            // 80 degree 0.08 85 degree

            V2D pc(new_frame_->w2c(pt->pos_));

            if (new_frame_->cam_->isInFrame(pc.cast<int>(), border_)) {
              // cv::circle(img_cp, cv::Point2f(pc[0], pc[1]), 3,
              // cv::Scalar(255, 255, 0), -1, 8); sub_map_ray_fov.push_back(pt);

              voxel_in_fov = true;
              int index = static_cast<int>(pc[1] / grid_size_) * grid_n_width_ +
                          static_cast<int>(pc[0] / grid_size_);
              grid_num_[index] = TYPE_MAP;
              Eigen::Vector3d obs_vec(new_frame_->pos() - pt->pos_);

              float cur_dist = obs_vec.norm();

              if (cur_dist <= map_dist_[index]) {
                map_dist_[index] = cur_dist;
                retrieve_voxel_points_[index] = pt;
              }
            }
          }

          if (voxel_in_fov) sub_feat_map_[sample_pos] = 0;
          break;
        } else {
          VOXEL_LOCATION sample_pos(loc_xyz[0], loc_xyz[1], loc_xyz[2]);
          auto iter = vm_map.find(sample_pos);
          if (iter != vm_map.end()) {
            VoxelOctoTree *current_octo;
            current_octo = iter->second->second->FindCorrespond(sample_point_w);
            if (current_octo->plane_ptr_->is_plane_) {
              pointWithVar plane_center;
              VoxelPlane &plane = *current_octo->plane_ptr_;
              plane_center.point_w = plane.center_;
              plane_center.normal = plane.normal_;
              visual_submap_->add_from_voxel_map.push_back(plane_center);
              break;
            }
          }
        }
      }
      // if(add_sample) sample_points.push_back(sample_points_temp);
    }
  }

  for (auto &key : DeleteKeyList) {
    sub_feat_map_.erase(key);
  }

  for (int i = 0; i < length_; i++) {
    if (grid_num_[i] == TYPE_MAP) {
      VisualPoint *pt = retrieve_voxel_points_[i];
      // visual_sub_map_cur.push_back(pt); // before

      V2D pc(new_frame_->w2c(pt->pos_));

      // cv::circle(img_cp, cv::Point2f(pc[0], pc[1]), 3, cv::Scalar(0, 0, 255),
      // -1, 8); // Green Sparse Align tracked

      V3D pt_cam(new_frame_->w2f(pt->pos_));
      // 深度连续性检查
      bool depth_discontinous = false;
      for (int u = -patch_size_half_; u <= patch_size_half_; u++) {
        for (int v = -patch_size_half_; v <= patch_size_half_; v++) {
          if (u == 0 && v == 0) continue;

          float depth = it[width_ * (v + int(pc[1])) + u + int(pc[0])];

          if (depth == 0.) continue;

          double delta_dist = abs(pt_cam[2] - depth);

          if (delta_dist > 0.5) {
            depth_discontinous = true;
            break;
          }
        }
        if (depth_discontinous) break;
      }
      if (depth_discontinous) continue;

      Feature *ref_ftr;
      std::vector<float> patch_wrap(warp_len_);

      int search_level;
      Eigen::Matrix2d A_cur_ref_zero;

      if (!pt->is_normal_initialized_) continue;

      if (normal_en_) {
        float phtometric_errors_min = std::numeric_limits<float>::max();

        if (pt->obs_.size() == 1) {
          // 只有一个关联图像块，将其选为参考图像块
          ref_ftr = *pt->obs_.begin();
          pt->ref_patch = ref_ftr;
          pt->has_ref_patch_ = true;
        } else if (!pt->has_ref_patch_) {
          for (auto it = pt->obs_.begin(), ite = pt->obs_.end(); it != ite;
               ++it) {
            // 不止一个关联图像块，选择光度误差最小的为参考图像块
            Feature *ref_patch_temp = *it;
            float *patch_temp = ref_patch_temp->patch_;
            float photometric_errors = 0.0;
            int count = 0;
            for (auto itm = pt->obs_.begin(), itme = pt->obs_.end();
                 itm != itme; ++itm) {
              if ((*itm)->id_ == ref_patch_temp->id_) continue;
              float *patch_cache = (*itm)->patch_;

              for (int ind = 0; ind < patch_size_total_; ind++) {
                photometric_errors += (patch_temp[ind] - patch_cache[ind]) *
                                      (patch_temp[ind] - patch_cache[ind]);
              }
              count++;
            }
            photometric_errors = photometric_errors / count;
            if (photometric_errors < phtometric_errors_min) {
              phtometric_errors_min = photometric_errors;
              ref_ftr = ref_patch_temp;
            }
          }
          pt->ref_patch = ref_ftr;
          pt->has_ref_patch_ = true;
        } else {
          ref_ftr = pt->ref_patch;
        }
      } else {
        if (!pt->getCloseViewObs(new_frame_->pos(), ref_ftr, pc)) continue;
      }

      // 计算仿射变换矩阵和搜索层级
      if (normal_en_) {
        V3D norm_vec =
            (ref_ftr->T_f_w_.rotationMatrix() * pt->normal_).normalized();

        V3D pf(ref_ftr->T_f_w_ * pt->pos_);

        SE3 T_cur_ref = new_frame_->T_f_w_ * ref_ftr->T_f_w_.inverse();

        GetWarpMatrixAffineHomography(*cam_, ref_ftr->px_, pf, norm_vec,
                                      T_cur_ref, 0, A_cur_ref_zero);

        search_level = GetBestSearchLevel(A_cur_ref_zero, 2);
      } else {
        auto iter_warp = warp_map_.find(ref_ftr->id_);
        if (iter_warp != warp_map_.end()) {
          search_level = iter_warp->second->search_level;
          A_cur_ref_zero = iter_warp->second->A_cur_ref;
        } else {
          GetWarpMatrixAffine(*cam_, ref_ftr->px_, ref_ftr->f_,
                              (ref_ftr->pos() - pt->pos_).norm(),
                              new_frame_->T_f_w_ * ref_ftr->T_f_w_.inverse(),
                              ref_ftr->level_, 0, patch_size_half_,
                              A_cur_ref_zero);

          search_level = GetBestSearchLevel(A_cur_ref_zero, 2);

          Warp *ot = new Warp(search_level, A_cur_ref_zero);
          warp_map_[ref_ftr->id_] = ot;
        }
      }

      for (int pyramid_level = 0; pyramid_level <= patch_pyrimid_level_ - 1;
           pyramid_level++) {
        WarpAffine(A_cur_ref_zero, ref_ftr->img_, ref_ftr->px_, ref_ftr->level_,
                   search_level, pyramid_level, patch_size_half_,
                   patch_wrap.data());
      }
      // 获取当前帧特征点对应的patch
      GetImagePatch(img, pc, patch_buffer_.data(), 0);

      float error = 0.0;
      for (int ind = 0; ind < patch_size_total_; ind++) {
        error += (ref_ftr->inv_expo_time_ * patch_wrap[ind] -
                  state_->inv_expo_time * patch_buffer_[ind]) *
                 (ref_ftr->inv_expo_time_ * patch_wrap[ind] -
                  state_->inv_expo_time * patch_buffer_[ind]);
      }

      // 默认关闭
      if (ncc_en_) {
        double ncc = CalculateNCC(patch_wrap.data(), patch_buffer_.data(),
                                  patch_size_total_);
        if (ncc < ncc_thre_) {
          // grid_num[i] = TYPE_UNKNOWN;
          continue;
        }
      }

      if (error > outlier_threshold_ * patch_size_total_) continue;

      visual_submap_->voxel_points.push_back(pt);
      visual_submap_->propa_errors.push_back(error);
      visual_submap_->search_levels.push_back(search_level);
      visual_submap_->errors.push_back(error);
      visual_submap_->warp_patch.push_back(patch_wrap);
      visual_submap_->inv_expo_list.push_back(ref_ftr->inv_expo_time_);
    }
  }
  total_points_ = visual_submap_->voxel_points.size();

  printf("[ VIO ] Retrieve %d points from visual sparse map\n", total_points_);
}

void VIOManager::ComputeJacobianAndUpdateEKF(cv::Mat img) {
  if (total_points_ == 0) return;

  compute_jacobian_time_ = update_ekf_time_ = 0.0;

  for (int level = patch_pyrimid_level_ - 1; level >= 0; level--) {
    if (inverse_composition_en_) {
      has_ref_patch_cache_ = false;
      UpdateStateInverse(img, level);
    } else {
      // 实际执行
      UpdateState(img, level);
    }
  }
  state_->cov -= G_ * state_->cov;
  UpdateFrameState(*state_);
}

void VIOManager::GenerateVisualMapPoints(cv::Mat img,
                                         std::vector<pointWithVar> &pv_list) {
  if (pv_list.size() <= 10) return;

  // 遍历当前帧点云数据，添加特征点至待处理列表中
  for (int i = 0; i < pv_list.size(); i++) {
    if (pv_list[i].normal == V3D(0, 0, 0)) continue;

    V3D pt = pv_list[i].point_w;
    V2D pc(new_frame_->w2c(pt));
    // 20px is the patch size in the matcher
    if (new_frame_->cam_->isInFrame(pc.cast<int>(), border_)) {
      int index = static_cast<int>(pc[1] / grid_size_) * grid_n_width_ +
                  static_cast<int>(pc[0] / grid_size_);

      if (grid_num_[index] != TYPE_MAP) {
        // Shi-Tomasi 角点检测
        float cur_value = vk::shiTomasiScore(img, pc[0], pc[1]);
        // if (cur_value < 5) continue;
        if (cur_value > scan_value_[index]) {
          scan_value_[index] = cur_value;
          append_voxel_points_[index] = pv_list[i];
          grid_num_[index] = TYPE_POINTCLOUD;
        }
      }
    }
  }

  // 遍历视觉局部地图（可能包含历史点等），添加特征点至待处理列表中
  for (int j = 0; j < visual_submap_->add_from_voxel_map.size(); j++) {
    V3D pt = visual_submap_->add_from_voxel_map[j].point_w;
    V2D pc(new_frame_->w2c(pt));
    // 20px is the patch size in the matcher
    if (new_frame_->cam_->isInFrame(pc.cast<int>(), border_)) {
      int index = static_cast<int>(pc[1] / grid_size_) * grid_n_width_ +
                  static_cast<int>(pc[0] / grid_size_);

      if (grid_num_[index] != TYPE_MAP) {
        float cur_value = vk::shiTomasiScore(img, pc[0], pc[1]);
        if (cur_value > scan_value_[index]) {
          scan_value_[index] = cur_value;
          append_voxel_points_[index] = visual_submap_->add_from_voxel_map[j];
          grid_num_[index] = TYPE_POINTCLOUD;
        }
      }
    }
  }

  int add = 0;
  for (int i = 0; i < length_; i++) {
    if (grid_num_[i] == TYPE_POINTCLOUD)  // && (scan_value[i]>=50))
    {
      pointWithVar pt_var = append_voxel_points_[i];
      V3D pt = pt_var.point_w;

      V3D norm_vec(new_frame_->T_f_w_.rotationMatrix() * pt_var.normal);
      V3D dir(new_frame_->T_f_w_ * pt);
      dir.normalize();
      double cos_theta = dir.dot(norm_vec);
      // if(std::fabs(cos_theta)<0.34) continue; // 70 degree
      V2D pc(new_frame_->w2c(pt));

      float *patch = new float[patch_size_total_];
      GetImagePatch(img, pc, patch, 0);

      VisualPoint *pt_new = new VisualPoint(pt);

      Eigen::Vector3d f = cam_->cam2world(pc);
      Feature *feature_new =
          new Feature(pt_new, patch, pc, f, new_frame_->T_f_w_, 0);
      feature_new->img_ = img;
      feature_new->id_ = new_frame_->id_;
      feature_new->inv_expo_time_ = state_->inv_expo_time;

      pt_new->addFrameRef(feature_new);
      pt_new->covariance_ = pt_var.var;
      pt_new->is_normal_initialized_ = true;

      if (cos_theta < 0) {
        pt_new->normal_ = -pt_var.normal;
      } else {
        pt_new->normal_ = pt_var.normal;
      }

      pt_new->previous_normal_ = pt_new->normal_;

      InsertPointIntoFeatureMapLRU(pt_new);
      add += 1;
      // map_cur_frame.push_back(pt_new);
    }
  }

  printf("[ VIO ] Append %d new visual map points\n", add);
}

void VIOManager::UpdateVisualMapPoints(cv::Mat img) {
  if (total_points_ == 0) return;

  int update_num = 0;
  SE3 pose_cur = new_frame_->T_f_w_;
  for (int i = 0; i < total_points_; i++) {
    VisualPoint *pt = visual_submap_->voxel_points[i];
    if (pt == nullptr) continue;
    if (pt->is_converged_) {
      pt->deleteNonRefPatchFeatures();
      continue;
    }

    V2D pc(new_frame_->w2c(pt->pos_));
    bool add_flag = false;

    float *patch_temp = new float[patch_size_total_];
    GetImagePatch(img, pc, patch_temp, 0);
    // TODO: condition: distance and view_angle
    // Step 1: time
    Feature *last_feature = pt->obs_.back();
    // if(new_frame_->id_ >= last_feature->id_ + 10) add_flag = true; // 10

    // Step 2: delta_pose
    SE3 pose_ref = last_feature->T_f_w_;
    SE3 delta_pose = pose_ref * pose_cur.inverse();
    double delta_p = delta_pose.translation().norm();
    double delta_theta =
        (delta_pose.rotationMatrix().trace() > 3.0 - 1e-6)
            ? 0.0
            : std::acos(0.5 * (delta_pose.rotationMatrix().trace() - 1));
    if (delta_p > 0.5 || delta_theta > 0.3) add_flag = true;  // 0.5 || 0.3

    // Step 3: pixel distance
    Eigen::Vector2d last_px = last_feature->px_;
    double pixel_dist = (pc - last_px).norm();
    if (pixel_dist > 40) add_flag = true;

    // Maintain the size of 3D point observation features.
    if (pt->obs_.size() >= 30) {
      Feature *ref_ftr;
      pt->findMinScoreFeature(new_frame_->pos(), ref_ftr);
      pt->deleteFeatureRef(ref_ftr);
      // cout<<"pt->obs_.size() exceed 20 !!!!!!"<<endl;
    }
    if (add_flag) {
      update_num += 1;
      update_flag_[i] = 1;
      Eigen::Vector3d f = cam_->cam2world(pc);
      Feature *ftr_new = new Feature(pt, patch_temp, pc, f, new_frame_->T_f_w_,
                                     visual_submap_->search_levels[i]);
      ftr_new->img_ = img;
      ftr_new->id_ = new_frame_->id_;
      ftr_new->inv_expo_time_ = state_->inv_expo_time;
      pt->addFrameRef(ftr_new);
    }
  }
  printf("[ VIO ] Update %d points in visual submap\n", update_num);
}

void VIOManager::UpdateReferencePatch(
    const std::unordered_map<VOXEL_LOCATION, VoxelOctoTree *> &plane_map) {
  if (total_points_ == 0) return;

  for (int i = 0; i < visual_submap_->voxel_points.size(); i++) {
    VisualPoint *pt = visual_submap_->voxel_points[i];

    if (!pt->is_normal_initialized_) continue;
    if (pt->is_converged_) continue;
    if (pt->obs_.size() <= 5) continue;
    if (update_flag_[i] == 0) continue;

    const V3D &p_w = pt->pos_;
    float loc_xyz[3];
    for (int j = 0; j < 3; j++) {
      loc_xyz[j] = p_w[j] / 0.5;
      if (loc_xyz[j] < 0) {
        loc_xyz[j] -= 1.0;
      }
    }
    VOXEL_LOCATION position((int64_t)loc_xyz[0], (int64_t)loc_xyz[1],
                            (int64_t)loc_xyz[2]);
    auto iter = plane_map.find(position);
    if (iter != plane_map.end()) {
      VoxelOctoTree *current_octo;
      current_octo = iter->second->FindCorrespond(p_w);
      if (current_octo->plane_ptr_->is_plane_) {
        VoxelPlane &plane = *current_octo->plane_ptr_;
        float dis_to_plane = plane.normal_(0) * p_w(0) +
                             plane.normal_(1) * p_w(1) +
                             plane.normal_(2) * p_w(2) + plane.d_;
        float dis_to_plane_abs = fabs(dis_to_plane);
        float dis_to_center =
            (plane.center_(0) - p_w(0)) * (plane.center_(0) - p_w(0)) +
            (plane.center_(1) - p_w(1)) * (plane.center_(1) - p_w(1)) +
            (plane.center_(2) - p_w(2)) * (plane.center_(2) - p_w(2));
        float range_dis = sqrt(dis_to_center - dis_to_plane * dis_to_plane);
        if (range_dis <= 3 * plane.radius_) {
          Eigen::Matrix<double, 1, 6> J_nq;
          J_nq.block<1, 3>(0, 0) = p_w - plane.center_;
          J_nq.block<1, 3>(0, 3) = -plane.normal_;
          double sigma_l = J_nq * plane.plane_var_ * J_nq.transpose();
          sigma_l +=
              plane.normal_.transpose() * pt->covariance_ * plane.normal_;
          // 检验更新法线方向
          if (dis_to_plane_abs < 3 * sqrt(sigma_l)) {
            // V3D norm_vec(new_frame_->T_f_w_.rotationMatrix() *
            // plane.normal_); V3D pf(new_frame_->T_f_w_ * pt->pos_); V3D
            // pf_ref(pt->ref_patch->T_f_w_ * pt->pos_); V3D
            // norm_vec_ref(pt->ref_patch->T_f_w_.rotationMatrix() *
            // plane.normal); double cos_ref = pf_ref.dot(norm_vec_ref);

            if (pt->previous_normal_.dot(plane.normal_) < 0) {
              pt->normal_ = -plane.normal_;
            } else {
              pt->normal_ = plane.normal_;
            }

            double normal_update = (pt->normal_ - pt->previous_normal_).norm();

            pt->previous_normal_ = pt->normal_;

            if (normal_update < 0.0001 && pt->obs_.size() > 10) {
              pt->is_converged_ = true;
              // visual_converged_point.push_back(pt);
            }
          }
        }
      }
    }

    // 计算NCC得分(公式12)选择最优参考图像块
    float score_max = -1000.;
    for (auto it = pt->obs_.begin(), ite = pt->obs_.end(); it != ite; ++it) {
      Feature *ref_patch_temp = *it;
      float *patch_temp = ref_patch_temp->patch_;
      float NCC_up = 0.0;
      float NCC_down1 = 0.0;
      float NCC_down2 = 0.0;
      float NCC = 0.0;
      float score = 0.0;
      int count = 0;

      V3D pf = ref_patch_temp->T_f_w_ * pt->pos_;
      V3D norm_vec = ref_patch_temp->T_f_w_.rotationMatrix() * pt->normal_;
      pf.normalize();
      double cos_angle = pf.dot(norm_vec);
      // if(fabs(cos_angle) < 0.86) continue; // 20 degree

      float ref_mean;
      if (abs(ref_patch_temp->mean_) < 1e-6) {
        float ref_sum =
            std::accumulate(patch_temp, patch_temp + patch_size_total_, 0.0);
        ref_mean = ref_sum / patch_size_total_;
        ref_patch_temp->mean_ = ref_mean;
      }

      for (auto itm = pt->obs_.begin(), itme = pt->obs_.end(); itm != itme;
           ++itm) {
        if ((*itm)->id_ == ref_patch_temp->id_) continue;
        float *patch_cache = (*itm)->patch_;

        float other_mean;
        if (abs((*itm)->mean_) < 1e-6) {
          float other_sum = std::accumulate(
              patch_cache, patch_cache + patch_size_total_, 0.0);
          other_mean = other_sum / patch_size_total_;
          (*itm)->mean_ = other_mean;
        }

        for (int ind = 0; ind < patch_size_total_; ind++) {
          NCC_up +=
              (patch_temp[ind] - ref_mean) * (patch_cache[ind] - other_mean);
          NCC_down1 +=
              (patch_temp[ind] - ref_mean) * (patch_temp[ind] - ref_mean);
          NCC_down2 +=
              (patch_cache[ind] - other_mean) * (patch_cache[ind] - other_mean);
        }
        NCC += fabs(NCC_up / sqrt(NCC_down1 * NCC_down2));
        count++;
      }

      NCC = NCC / count;

      score = NCC + cos_angle;

      ref_patch_temp->score_ = score;

      if (score > score_max) {
        score_max = score;
        pt->ref_patch = ref_patch_temp;
        pt->has_ref_patch_ = true;
      }
    }
  }
}

void VIOManager::UpdateReferencePatch(
    const std::unordered_map<VOXEL_LOCATION,
                             typename std::list<VMData>::iterator> &vm_map) {
  if (total_points_ == 0) return;

  for (int i = 0; i < visual_submap_->voxel_points.size(); i++) {
    VisualPoint *pt = visual_submap_->voxel_points[i];

    if (!pt->is_normal_initialized_) continue;
    if (pt->is_converged_) continue;
    if (pt->obs_.size() <= 5) continue;
    if (update_flag_[i] == 0) continue;

    const V3D &p_w = pt->pos_;
    float loc_xyz[3];
    for (int j = 0; j < 3; j++) {
      loc_xyz[j] = p_w[j] / 0.5;
      if (loc_xyz[j] < 0) {
        loc_xyz[j] -= 1.0;
      }
    }
    VOXEL_LOCATION position((int64_t)loc_xyz[0], (int64_t)loc_xyz[1],
                            (int64_t)loc_xyz[2]);
    auto iter = vm_map.find(position);
    if (iter != vm_map.end()) {
      VoxelOctoTree *current_octo;
      current_octo = iter->second->second->FindCorrespond(p_w);
      if (current_octo->plane_ptr_->is_plane_) {
        VoxelPlane &plane = *current_octo->plane_ptr_;
        float dis_to_plane = plane.normal_(0) * p_w(0) +
                             plane.normal_(1) * p_w(1) +
                             plane.normal_(2) * p_w(2) + plane.d_;
        float dis_to_plane_abs = fabs(dis_to_plane);
        float dis_to_center =
            (plane.center_(0) - p_w(0)) * (plane.center_(0) - p_w(0)) +
            (plane.center_(1) - p_w(1)) * (plane.center_(1) - p_w(1)) +
            (plane.center_(2) - p_w(2)) * (plane.center_(2) - p_w(2));
        float range_dis = sqrt(dis_to_center - dis_to_plane * dis_to_plane);
        if (range_dis <= 3 * plane.radius_) {
          Eigen::Matrix<double, 1, 6> J_nq;
          J_nq.block<1, 3>(0, 0) = p_w - plane.center_;
          J_nq.block<1, 3>(0, 3) = -plane.normal_;
          double sigma_l = J_nq * plane.plane_var_ * J_nq.transpose();
          sigma_l +=
              plane.normal_.transpose() * pt->covariance_ * plane.normal_;
          // 检验更新法线方向
          if (dis_to_plane_abs < 3 * sqrt(sigma_l)) {
            // V3D norm_vec(new_frame_->T_f_w_.rotationMatrix() *
            // plane.normal_); V3D pf(new_frame_->T_f_w_ * pt->pos_); V3D
            // pf_ref(pt->ref_patch->T_f_w_ * pt->pos_); V3D
            // norm_vec_ref(pt->ref_patch->T_f_w_.rotationMatrix() *
            // plane.normal); double cos_ref = pf_ref.dot(norm_vec_ref);

            if (pt->previous_normal_.dot(plane.normal_) < 0) {
              pt->normal_ = -plane.normal_;
            } else {
              pt->normal_ = plane.normal_;
            }

            double normal_update = (pt->normal_ - pt->previous_normal_).norm();

            pt->previous_normal_ = pt->normal_;

            if (normal_update < 0.0001 && pt->obs_.size() > 10) {
              pt->is_converged_ = true;
              // visual_converged_point.push_back(pt);
            }
          }
        }
      }
    }

    // 计算NCC得分(公式12)选择最优参考图像块
    float score_max = -1000.;
    for (auto it = pt->obs_.begin(), ite = pt->obs_.end(); it != ite; ++it) {
      Feature *ref_patch_temp = *it;
      float *patch_temp = ref_patch_temp->patch_;
      float NCC_up = 0.0;
      float NCC_down1 = 0.0;
      float NCC_down2 = 0.0;
      float NCC = 0.0;
      float score = 0.0;
      int count = 0;

      V3D pf = ref_patch_temp->T_f_w_ * pt->pos_;
      V3D norm_vec = ref_patch_temp->T_f_w_.rotationMatrix() * pt->normal_;
      pf.normalize();
      double cos_angle = pf.dot(norm_vec);
      // if(fabs(cos_angle) < 0.86) continue; // 20 degree

      float ref_mean;
      if (abs(ref_patch_temp->mean_) < 1e-6) {
        float ref_sum =
            std::accumulate(patch_temp, patch_temp + patch_size_total_, 0.0);
        ref_mean = ref_sum / patch_size_total_;
        ref_patch_temp->mean_ = ref_mean;
      }

      for (auto itm = pt->obs_.begin(), itme = pt->obs_.end(); itm != itme;
           ++itm) {
        if ((*itm)->id_ == ref_patch_temp->id_) continue;
        float *patch_cache = (*itm)->patch_;

        float other_mean;
        if (abs((*itm)->mean_) < 1e-6) {
          float other_sum = std::accumulate(
              patch_cache, patch_cache + patch_size_total_, 0.0);
          other_mean = other_sum / patch_size_total_;
          (*itm)->mean_ = other_mean;
        }

        for (int ind = 0; ind < patch_size_total_; ind++) {
          NCC_up +=
              (patch_temp[ind] - ref_mean) * (patch_cache[ind] - other_mean);
          NCC_down1 +=
              (patch_temp[ind] - ref_mean) * (patch_temp[ind] - ref_mean);
          NCC_down2 +=
              (patch_cache[ind] - other_mean) * (patch_cache[ind] - other_mean);
        }
        NCC += fabs(NCC_up / sqrt(NCC_down1 * NCC_down2));
        count++;
      }

      NCC = NCC / count;

      score = NCC + cos_angle;

      ref_patch_temp->score_ = score;

      if (score > score_max) {
        score_max = score;
        pt->ref_patch = ref_patch_temp;
        pt->has_ref_patch_ = true;
      }
    }
  }
}

void VIOManager::ProjectPatchFromRefToCur() {
  if (total_points_ == 0) return;
  // if(new_frame_->id_ != 2) return; //124

  int patch_size = 25;
  std::string dir = std::string(ROOT_DIR) + "Log/ref_cur_combine/";

  cv::Mat result = cv::Mat::zeros(height_, width_, CV_8UC1);
  cv::Mat result_normal = cv::Mat::zeros(height_, width_, CV_8UC1);
  cv::Mat result_dense = cv::Mat::zeros(height_, width_, CV_8UC1);

  cv::Mat img_photometric_error = new_frame_->img_.clone();

  uchar *it = (uchar *)result.data;
  uchar *it_normal = (uchar *)result_normal.data;
  uchar *it_dense = (uchar *)result_dense.data;

  struct pixel_member {
    Eigen::Vector2f pixel_pos;
    uint8_t pixel_value;
  };

  int num = 0;
  for (int i = 0; i < visual_submap_->voxel_points.size(); i++) {
    VisualPoint *pt = visual_submap_->voxel_points[i];

    if (pt->is_normal_initialized_) {
      Feature *ref_ftr;
      ref_ftr = pt->ref_patch;
      // Feature* ref_ftr;
      V2D pc(new_frame_->w2c(pt->pos_));
      V2D pc_prior(new_frame_->w2c_prior(pt->pos_));

      V3D norm_vec(ref_ftr->T_f_w_.rotationMatrix() * pt->normal_);
      V3D pf(ref_ftr->T_f_w_ * pt->pos_);

      if (pf.dot(norm_vec) < 0) norm_vec = -norm_vec;

      // norm_vec << norm_vec(1), norm_vec(0), norm_vec(2);
      cv::Mat img_cur = new_frame_->img_;
      cv::Mat img_ref = ref_ftr->img_;

      SE3 T_cur_ref = new_frame_->T_f_w_ * ref_ftr->T_f_w_.inverse();
      Eigen::Matrix2d A_cur_ref;
      GetWarpMatrixAffineHomography(*cam_, ref_ftr->px_, pf, norm_vec,
                                    T_cur_ref, 0, A_cur_ref);

      // const Matrix2f A_ref_cur = A_cur_ref.inverse().cast<float>();
      int search_level = GetBestSearchLevel(A_cur_ref.inverse(), 2);

      double D = A_cur_ref.determinant();
      if (D > 3) continue;

      num++;

      cv::Mat ref_cur_combine_temp;
      int radius = 20;
      cv::hconcat(img_cur, img_ref, ref_cur_combine_temp);
      cv::cvtColor(ref_cur_combine_temp, ref_cur_combine_temp, CV_GRAY2BGR);

      GetImagePatch(img_cur, pc, patch_buffer_.data(), 0);

      float error_est = 0.0;
      float error_gt = 0.0;

      for (int ind = 0; ind < patch_size_total_; ind++) {
        error_est +=
            (ref_ftr->inv_expo_time_ * visual_submap_->warp_patch[i][ind] -
             state_->inv_expo_time * patch_buffer_[ind]) *
            (ref_ftr->inv_expo_time_ * visual_submap_->warp_patch[i][ind] -
             state_->inv_expo_time * patch_buffer_[ind]);
      }
      std::string ref_est =
          "ref_est " + std::to_string(1.0 / ref_ftr->inv_expo_time_);
      std::string cur_est =
          "cur_est " + std::to_string(1.0 / state_->inv_expo_time);
      std::string cur_propa = "cur_gt " + std::to_string(error_gt);
      std::string cur_optimize = "cur_est " + std::to_string(error_est);

      cv::putText(ref_cur_combine_temp, ref_est,
                  cv::Point2f(ref_ftr->px_[0] + img_cur.cols - 40,
                              ref_ftr->px_[1] + 40),
                  cv::FONT_HERSHEY_COMPLEX, 0.4, cv::Scalar(0, 255, 0), 1, 8,
                  0);

      cv::putText(ref_cur_combine_temp, cur_est,
                  cv::Point2f(pc[0] - 40, pc[1] + 40), cv::FONT_HERSHEY_COMPLEX,
                  0.4, cv::Scalar(0, 255, 0), 1, 8, 0);
      cv::putText(ref_cur_combine_temp, cur_propa,
                  cv::Point2f(pc[0] - 40, pc[1] + 60), cv::FONT_HERSHEY_COMPLEX,
                  0.4, cv::Scalar(0, 0, 255), 1, 8, 0);
      cv::putText(ref_cur_combine_temp, cur_optimize,
                  cv::Point2f(pc[0] - 40, pc[1] + 80), cv::FONT_HERSHEY_COMPLEX,
                  0.4, cv::Scalar(0, 255, 0), 1, 8, 0);

      cv::rectangle(ref_cur_combine_temp,
                    cv::Point2f(ref_ftr->px_[0] + img_cur.cols - radius,
                                ref_ftr->px_[1] - radius),
                    cv::Point2f(ref_ftr->px_[0] + img_cur.cols + radius,
                                ref_ftr->px_[1] + radius),
                    cv::Scalar(0, 0, 255), 1);
      cv::rectangle(ref_cur_combine_temp,
                    cv::Point2f(pc[0] - radius, pc[1] - radius),
                    cv::Point2f(pc[0] + radius, pc[1] + radius),
                    cv::Scalar(0, 255, 0), 1);
      cv::rectangle(ref_cur_combine_temp,
                    cv::Point2f(pc_prior[0] - radius, pc_prior[1] - radius),
                    cv::Point2f(pc_prior[0] + radius, pc_prior[1] + radius),
                    cv::Scalar(255, 255, 255), 1);
      cv::circle(ref_cur_combine_temp,
                 cv::Point2f(ref_ftr->px_[0] + img_cur.cols, ref_ftr->px_[1]),
                 1, cv::Scalar(0, 0, 255), -1, 8);
      cv::circle(ref_cur_combine_temp, cv::Point2f(pc[0], pc[1]), 1,
                 cv::Scalar(0, 255, 0), -1, 8);
      cv::circle(ref_cur_combine_temp, cv::Point2f(pc_prior[0], pc_prior[1]), 1,
                 cv::Scalar(255, 255, 255), -1, 8);
      cv::imwrite(dir + std::to_string(new_frame_->id_) + "_" +
                      std::to_string(ref_ftr->id_) + "_" + std::to_string(num) +
                      ".png",
                  ref_cur_combine_temp);

      std::vector<std::vector<pixel_member>> pixel_warp_matrix;

      for (int y = 0; y < patch_size; ++y) {
        std::vector<pixel_member> pixel_warp_vec;
        for (int x = 0; x < patch_size; ++x)  //, ++patch_ptr)
        {
          Eigen::Vector2f px_patch(x - patch_size / 2, y - patch_size / 2);
          px_patch *= (1 << search_level);
          const Eigen::Vector2f px_ref(px_patch + ref_ftr->px_.cast<float>());
          uint8_t pixel_value =
              (uint8_t)vk::interpolateMat_8u(img_ref, px_ref[0], px_ref[1]);

          const Eigen::Vector2f px(A_cur_ref.cast<float>() * px_patch +
                                   pc.cast<float>());
          if (px[0] < 0 || px[1] < 0 || px[0] >= img_cur.cols - 1 ||
              px[1] >= img_cur.rows - 1)
            continue;
          else {
            pixel_member pixel_warp;
            pixel_warp.pixel_pos << px[0], px[1];
            pixel_warp.pixel_value = pixel_value;
            pixel_warp_vec.push_back(pixel_warp);
          }
        }
        pixel_warp_matrix.push_back(pixel_warp_vec);
      }

      float x_min = 1000;
      float y_min = 1000;
      float x_max = 0;
      float y_max = 0;

      for (int i = 0; i < pixel_warp_matrix.size(); i++) {
        std::vector<pixel_member> pixel_warp_row = pixel_warp_matrix[i];
        for (int j = 0; j < pixel_warp_row.size(); j++) {
          float x_temp = pixel_warp_row[j].pixel_pos[0];
          float y_temp = pixel_warp_row[j].pixel_pos[1];
          if (x_temp < x_min) x_min = x_temp;
          if (y_temp < y_min) y_min = y_temp;
          if (x_temp > x_max) x_max = x_temp;
          if (y_temp > y_max) y_max = y_temp;
        }
      }
      int x_min_i = floor(x_min);
      int y_min_i = floor(y_min);
      int x_max_i = ceil(x_max);
      int y_max_i = ceil(y_max);
      Eigen::Matrix2f A_cur_ref_Inv = A_cur_ref.inverse().cast<float>();
      for (int i = x_min_i; i < x_max_i; i++) {
        for (int j = y_min_i; j < y_max_i; j++) {
          Eigen::Vector2f pc_temp(i, j);
          Eigen::Vector2f px_patch =
              A_cur_ref_Inv * (pc_temp - pc.cast<float>());
          if (px_patch[0] > (-patch_size / 2 * (1 << search_level)) &&
              px_patch[0] < (patch_size / 2 * (1 << search_level)) &&
              px_patch[1] > (-patch_size / 2 * (1 << search_level)) &&
              px_patch[1] < (patch_size / 2 * (1 << search_level))) {
            const Eigen::Vector2f px_ref(px_patch + ref_ftr->px_.cast<float>());
            uint8_t pixel_value =
                (uint8_t)vk::interpolateMat_8u(img_ref, px_ref[0], px_ref[1]);
            it_normal[width_ * j + i] = pixel_value;
          }
        }
      }
    }
  }
  for (int i = 0; i < visual_submap_->voxel_points.size(); i++) {
    VisualPoint *pt = visual_submap_->voxel_points[i];

    if (!pt->is_normal_initialized_) continue;

    Feature *ref_ftr;
    V2D pc(new_frame_->w2c(pt->pos_));
    ref_ftr = pt->ref_patch;

    Eigen::Matrix2d A_cur_ref;
    GetWarpMatrixAffine(*cam_, ref_ftr->px_, ref_ftr->f_,
                        (ref_ftr->pos() - pt->pos_).norm(),
                        new_frame_->T_f_w_ * ref_ftr->T_f_w_.inverse(), 0, 0,
                        patch_size_half_, A_cur_ref);
    int search_level = GetBestSearchLevel(A_cur_ref.inverse(), 2);
    double D = A_cur_ref.determinant();
    if (D > 3) continue;

    cv::Mat img_cur = new_frame_->img_;
    cv::Mat img_ref = ref_ftr->img_;
    for (int y = 0; y < patch_size; ++y) {
      for (int x = 0; x < patch_size; ++x)  //, ++patch_ptr)
      {
        Eigen::Vector2f px_patch(x - patch_size / 2, y - patch_size / 2);
        px_patch *= (1 << search_level);
        const Eigen::Vector2f px_ref(px_patch + ref_ftr->px_.cast<float>());
        uint8_t pixel_value =
            (uint8_t)vk::interpolateMat_8u(img_ref, px_ref[0], px_ref[1]);

        const Eigen::Vector2f px(A_cur_ref.cast<float>() * px_patch +
                                 pc.cast<float>());
        if (px[0] < 0 || px[1] < 0 || px[0] >= img_cur.cols - 1 ||
            px[1] >= img_cur.rows - 1)
          continue;
        else {
          int col = int(px[0]);
          int row = int(px[1]);
          it[width_ * row + col] = pixel_value;
        }
      }
    }
  }
  cv::Mat ref_cur_combine;
  cv::Mat ref_cur_combine_normal;
  cv::Mat ref_cur_combine_error;

  cv::hconcat(result, new_frame_->img_, ref_cur_combine);
  cv::hconcat(result_normal, new_frame_->img_, ref_cur_combine_normal);

  cv::cvtColor(ref_cur_combine, ref_cur_combine, CV_GRAY2BGR);
  cv::cvtColor(ref_cur_combine_normal, ref_cur_combine_normal, CV_GRAY2BGR);
  cv::absdiff(img_photometric_error, result_normal, img_photometric_error);
  cv::hconcat(img_photometric_error, new_frame_->img_, ref_cur_combine_error);

  cv::imwrite(dir + std::to_string(new_frame_->id_) + "_0_" + ".png",
              ref_cur_combine);
  cv::imwrite(dir + std::to_string(new_frame_->id_) + +"_0_" +
                  "photometric"
                  ".png",
              ref_cur_combine_error);
  cv::imwrite(dir + std::to_string(new_frame_->id_) + "_0_" + "normal" + ".png",
              ref_cur_combine_normal);
}

void VIOManager::PrecomputeReferencePatches(int level) {
  double t1 = omp_get_wtime();
  if (total_points_ == 0) return;
  MD(1, 2) Jimg;
  MD(2, 3) Jdpi;
  MD(1, 3) Jdphi, Jdp, JdR, Jdt;

  const int H_DIM = total_points_ * patch_size_total_;

  H_sub_inv_.resize(H_DIM, 6);
  H_sub_inv_.setZero();
  M3D p_w_hat;

  for (int i = 0; i < total_points_; i++) {
    const int scale = (1 << level);

    VisualPoint *pt = visual_submap_->voxel_points[i];
    cv::Mat img = pt->ref_patch->img_;

    if (pt == nullptr) continue;

    double depth((pt->pos_ - pt->ref_patch->pos()).norm());
    V3D pf = pt->ref_patch->f_ * depth;
    V2D pc = pt->ref_patch->px_;
    M3D R_ref_w = pt->ref_patch->T_f_w_.rotationMatrix();

    ComputeProjectionJacobian(pf, Jdpi);
    p_w_hat << SKEW_SYM_MATRX(pt->pos_);

    const float u_ref = pc[0];
    const float v_ref = pc[1];
    const int u_ref_i = floorf(pc[0] / scale) * scale;
    const int v_ref_i = floorf(pc[1] / scale) * scale;
    const float subpix_u_ref = (u_ref - u_ref_i) / scale;
    const float subpix_v_ref = (v_ref - v_ref_i) / scale;
    const float w_ref_tl = (1.0 - subpix_u_ref) * (1.0 - subpix_v_ref);
    const float w_ref_tr = subpix_u_ref * (1.0 - subpix_v_ref);
    const float w_ref_bl = (1.0 - subpix_u_ref) * subpix_v_ref;
    const float w_ref_br = subpix_u_ref * subpix_v_ref;

    for (int x = 0; x < patch_size_; x++) {
      uint8_t *img_ptr =
          (uint8_t *)img.data +
          (v_ref_i + x * scale - patch_size_half_ * scale) * width_ + u_ref_i -
          patch_size_half_ * scale;
      for (int y = 0; y < patch_size_; ++y, img_ptr += scale) {
        float du =
            0.5f * ((w_ref_tl * img_ptr[scale] + w_ref_tr * img_ptr[scale * 2] +
                     w_ref_bl * img_ptr[scale * width_ + scale] +
                     w_ref_br * img_ptr[scale * width_ + scale * 2]) -
                    (w_ref_tl * img_ptr[-scale] + w_ref_tr * img_ptr[0] +
                     w_ref_bl * img_ptr[scale * width_ - scale] +
                     w_ref_br * img_ptr[scale * width_]));
        float dv = 0.5f * ((w_ref_tl * img_ptr[scale * width_] +
                            w_ref_tr * img_ptr[scale + scale * width_] +
                            w_ref_bl * img_ptr[width_ * scale * 2] +
                            w_ref_br * img_ptr[width_ * scale * 2 + scale]) -
                           (w_ref_tl * img_ptr[-scale * width_] +
                            w_ref_tr * img_ptr[-scale * width_ + scale] +
                            w_ref_bl * img_ptr[0] + w_ref_br * img_ptr[scale]));

        Jimg << du, dv;
        Jimg = Jimg * (1.0 / scale);

        JdR = Jimg * Jdpi * R_ref_w * p_w_hat;
        Jdt = -Jimg * Jdpi * R_ref_w;

        H_sub_inv_.block<1, 6>(i * patch_size_total_ + x * patch_size_ + y, 0)
            << JdR,
            Jdt;
      }
    }
  }
  has_ref_patch_cache_ = true;
}

void VIOManager::UpdateStateInverse(cv::Mat img, int level) {
  if (total_points_ == 0) return;
  StatesGroup old_state = (*state_);
  V2D pc;
  MD(1, 2) Jimg;
  MD(2, 3) Jdpi;
  MD(1, 3) Jdphi, Jdp, JdR, Jdt;
  Eigen::VectorXd z;
  Eigen::MatrixXd H_sub;
  bool EKF_end = false;
  float last_error = std::numeric_limits<float>::max();
  compute_jacobian_time_ = update_ekf_time_ = 0.0;
  M3D P_wi_hat;
  bool z_init = true;
  const int H_DIM = total_points_ * patch_size_total_;

  z.resize(H_DIM);
  z.setZero();

  H_sub.resize(H_DIM, 6);
  H_sub.setZero();

  for (int iteration = 0; iteration < max_iterations_; iteration++) {
    double t1 = omp_get_wtime();
    double count_outlier = 0;
    if (has_ref_patch_cache_ == false) PrecomputeReferencePatches(level);
    int n_meas = 0;
    float error = 0.0;
    M3D Rwi(state_->rot_end);
    V3D Pwi(state_->pos_end);
    P_wi_hat << SKEW_SYM_MATRX(Pwi);
    Rcw_ = Rci_ * Rwi.transpose();
    Pcw_ = -Rci_ * Rwi.transpose() * Pwi + Pci_;

    M3D p_hat;

    for (int i = 0; i < total_points_; i++) {
      float patch_error = 0.0;

      const int scale = (1 << level);

      VisualPoint *pt = visual_submap_->voxel_points[i];

      if (pt == nullptr) continue;

      V3D pf = Rcw_ * pt->pos_ + Pcw_;
      pc = cam_->world2cam(pf);

      const float u_ref = pc[0];
      const float v_ref = pc[1];
      const int u_ref_i = floorf(pc[0] / scale) * scale;
      const int v_ref_i = floorf(pc[1] / scale) * scale;
      const float subpix_u_ref = (u_ref - u_ref_i) / scale;
      const float subpix_v_ref = (v_ref - v_ref_i) / scale;
      const float w_ref_tl = (1.0 - subpix_u_ref) * (1.0 - subpix_v_ref);
      const float w_ref_tr = subpix_u_ref * (1.0 - subpix_v_ref);
      const float w_ref_bl = (1.0 - subpix_u_ref) * subpix_v_ref;
      const float w_ref_br = subpix_u_ref * subpix_v_ref;

      std::vector<float> P = visual_submap_->warp_patch[i];
      for (int x = 0; x < patch_size_; x++) {
        uint8_t *img_ptr =
            (uint8_t *)img.data +
            (v_ref_i + x * scale - patch_size_half_ * scale) * width_ +
            u_ref_i - patch_size_half_ * scale;
        for (int y = 0; y < patch_size_; ++y, img_ptr += scale) {
          double res = w_ref_tl * img_ptr[0] + w_ref_tr * img_ptr[scale] +
                       w_ref_bl * img_ptr[scale * width_] +
                       w_ref_br * img_ptr[scale * width_ + scale] -
                       P[patch_size_total_ * level + x * patch_size_ + y];
          z(i * patch_size_total_ + x * patch_size_ + y) = res;
          patch_error += res * res;
          MD(1, 3)
          J_dR = H_sub_inv_.block<1, 3>(
              i * patch_size_total_ + x * patch_size_ + y, 0);
          MD(1, 3)
          J_dt = H_sub_inv_.block<1, 3>(
              i * patch_size_total_ + x * patch_size_ + y, 3);
          JdR = J_dR * Rwi + J_dt * P_wi_hat * Rwi;
          Jdt = J_dt * Rwi;
          H_sub.block<1, 6>(i * patch_size_total_ + x * patch_size_ + y, 0)
              << JdR,
              Jdt;
          n_meas++;
        }
      }
      visual_submap_->errors[i] = patch_error;
      error += patch_error;
    }

    error = error / n_meas;

    compute_jacobian_time_ += omp_get_wtime() - t1;

    double t3 = omp_get_wtime();

    if (error <= last_error) {
      old_state = (*state_);
      last_error = error;

      auto &&H_sub_T = H_sub.transpose();
      H_T_H_.setZero();
      G_.setZero();
      H_T_H_.block<6, 6>(0, 0) = H_sub_T * H_sub;
      Mat19d &&K_1 =
          (H_T_H_ + (state_->cov / img_point_cov_).inverse()).inverse();
      auto &&HTz = H_sub_T * z;
      auto vec = (*state_propagat_) - (*state_);
      G_.block<DIM_STATE, 6>(0, 0) =
          K_1.block<DIM_STATE, 6>(0, 0) * H_T_H_.block<6, 6>(0, 0);
      auto solution = -K_1.block<DIM_STATE, 6>(0, 0) * HTz + vec -
                      G_.block<DIM_STATE, 6>(0, 0) * vec.block<6, 1>(0, 0);
      (*state_) += solution;
      auto &&rot_add = solution.block<3, 1>(0, 0);
      auto &&t_add = solution.block<3, 1>(3, 0);

      if ((rot_add.norm() * 57.3f < 0.001f) &&
          (t_add.norm() * 100.0f < 0.001f)) {
        EKF_end = true;
      }
    } else {
      (*state_) = old_state;
      EKF_end = true;
    }

    update_ekf_time_ += omp_get_wtime() - t3;

    if (iteration == max_iterations_ || EKF_end) break;
  }
}

void VIOManager::UpdateState(cv::Mat img, int level) {
  if (total_points_ == 0) return;
  StatesGroup old_state = (*state_);

  Eigen::VectorXd z;
  Eigen::MatrixXd H_sub;
  bool EKF_end = false;
  float last_error = std::numeric_limits<float>::max();

  const int H_DIM = total_points_ * patch_size_total_;
  z.resize(H_DIM);
  z.setZero();
  H_sub.resize(H_DIM, 7);  // R, t, τ
  H_sub.setZero();

  for (int iteration = 0; iteration < max_iterations_; iteration++) {
    double t1 = omp_get_wtime();

    M3D Rwi(state_->rot_end);
    V3D Pwi(state_->pos_end);
    Rcw_ = Rci_ * Rwi.transpose();
    Pcw_ = -Rci_ * Rwi.transpose() * Pwi + Pci_;
    Jdp_dt_ = Rci_ * Rwi.transpose();

    float error = 0.0;
    int n_meas = 0;
    // int max_threads = omp_get_max_threads();
    // int desired_threads = std::min(max_threads, total_points);
    // omp_set_num_threads(desired_threads);

#ifdef MP_EN
    omp_set_num_threads(MP_PROC_NUM);
#pragma omp parallel for reduction(+ : error, n_meas)
#endif
    for (int i = 0; i < total_points_; i++) {
      // printf("thread is %d, i=%d, i address is %p\n", omp_get_thread_num(),
      // i, &i);
      MD(1, 2) Jimg;                  //图像雅可比
      MD(2, 3) Jdpi;                  //投影雅可比
      MD(1, 3) Jdphi, Jdp, JdR, Jdt;  //状态雅可比

      float patch_error = 0.0;
      int search_level = visual_submap_->search_levels[i];
      int pyramid_level = level + search_level;
      int scale = (1 << pyramid_level);  //等价于 scale = pow(2, pyramid_level)
      float inv_scale = 1.0f / scale;

      VisualPoint *pt = visual_submap_->voxel_points[i];

      if (pt == nullptr) continue;

      V3D pf = Rcw_ * pt->pos_ + Pcw_;
      V2D pc = cam_->world2cam(pf);
      // 检查投影点是否远离图像边界，确保有足够空间计算梯度
      int total_margin = patch_size_half_ * scale + scale;
      if (pc[0] < total_margin || pc[0] > (img.cols - total_margin) ||
          pc[1] < total_margin || pc[1] > (img.rows - total_margin)) {
        continue;
      }
      // 计算∂u/∂ẟξ
      ComputeProjectionJacobian(pf, Jdpi);
      M3D p_hat;
      p_hat << SKEW_SYM_MATRX(pf);
      // 图像中的浮点坐标
      float u_ref = pc[0];
      float v_ref = pc[1];
      // 坐标整数部分
      int u_ref_i = floorf(pc[0] / scale) * scale;
      int v_ref_i = floorf(pc[1] / scale) * scale;
      // 坐标小数部分
      float subpix_u_ref = (u_ref - u_ref_i) / scale;
      float subpix_v_ref = (v_ref - v_ref_i) / scale;
      // 左上、右上、左下、右下的权重，用于插值计算
      float w_ref_tl = (1.0 - subpix_u_ref) * (1.0 - subpix_v_ref);
      float w_ref_tr = subpix_u_ref * (1.0 - subpix_v_ref);
      float w_ref_bl = (1.0 - subpix_u_ref) * subpix_v_ref;
      float w_ref_br = subpix_u_ref * subpix_v_ref;

      // 参考帧的光度和曝光时间
      std::vector<float> P = visual_submap_->warp_patch[i];
      double inv_ref_expo = visual_submap_->inv_expo_list[i];

      for (int x = 0; x < patch_size_; x++) {
        uint8_t *img_ptr =
            (uint8_t *)img.data +
            (v_ref_i + x * scale - patch_size_half_ * scale) * width_ +
            u_ref_i - patch_size_half_ * scale;
        for (int y = 0; y < patch_size_; ++y, img_ptr += scale) {
          // 残差计算
          // 当前帧光度I_k
          double cur_value = w_ref_tl * img_ptr[0] + w_ref_tr * img_ptr[scale] +
                             w_ref_bl * img_ptr[scale * width_] +
                             w_ref_br * img_ptr[scale * width_ + scale];
          // 残差计算 根据公式（22）,z = τ_k * I_k - τ_r * I_r
          double res =
              state_->inv_expo_time * cur_value -
              inv_ref_expo * P[patch_size_total_ * level + x * patch_size_ + y];
          z(i * patch_size_total_ + x * patch_size_ + y) = res;
          patch_error += res * res;
          n_meas += 1;
          visual_submap_->errors[i] = patch_error;
          error += patch_error;

          // 雅可比计算
          // 像素梯度计算，∂I/∂u
          float du =
              0.5f *
              ((w_ref_tl * img_ptr[scale] + w_ref_tr * img_ptr[scale * 2] +
                w_ref_bl * img_ptr[scale * width_ + scale] +
                w_ref_br * img_ptr[scale * width_ + scale * 2]) -
               (w_ref_tl * img_ptr[-scale] + w_ref_tr * img_ptr[0] +
                w_ref_bl * img_ptr[scale * width_ - scale] +
                w_ref_br * img_ptr[scale * width_]));
          float dv =
              0.5f * ((w_ref_tl * img_ptr[scale * width_] +
                       w_ref_tr * img_ptr[scale + scale * width_] +
                       w_ref_bl * img_ptr[width_ * scale * 2] +
                       w_ref_br * img_ptr[width_ * scale * 2 + scale]) -
                      (w_ref_tl * img_ptr[-scale * width_] +
                       w_ref_tr * img_ptr[-scale * width_ + scale] +
                       w_ref_bl * img_ptr[0] + w_ref_br * img_ptr[scale]));

          Jimg << du, dv;
          Jimg = Jimg * state_->inv_expo_time;
          Jimg = Jimg * inv_scale;
          // J = - ∂I/∂u * ∂u/∂ẟξ (视觉SLAM14讲8.19)
          Jdphi = Jimg * Jdpi * p_hat;
          Jdp = -Jimg * Jdpi;
          JdR = Jdphi * Jdphi_dR_ + Jdp * Jdp_dR_;
          Jdt = Jdp * Jdp_dt_;

          if (exposure_estimate_en_) {
            H_sub.block<1, 7>(i * patch_size_total_ + x * patch_size_ + y, 0)
                << JdR,
                Jdt, cur_value;
          } else {
            H_sub.block<1, 6>(i * patch_size_total_ + x * patch_size_ + y, 0)
                << JdR,
                Jdt;
          }
        }
      }
    }

    error = error / n_meas;

    compute_jacobian_time_ += omp_get_wtime() - t1;

    double t3 = omp_get_wtime();

    if (error <= last_error) {
      old_state = (*state_);
      last_error = error;

      // 和LIO基本一致，根据公式11更新
      auto &&H_sub_T = H_sub.transpose();
      H_T_H_.setZero();
      G_.setZero();
      H_T_H_.block<7, 7>(0, 0) = H_sub_T * H_sub;
      Mat19d &&K_1 =
          (H_T_H_ + (state_->cov / img_point_cov_).inverse()).inverse();
      auto &&HTz = H_sub_T * z;
      auto vec = (*state_propagat_) - (*state_);
      G_.block<DIM_STATE, 7>(0, 0) =
          K_1.block<DIM_STATE, 7>(0, 0) * H_T_H_.block<7, 7>(0, 0);
      MD(DIM_STATE, 1)
      solution = -K_1.block<DIM_STATE, 7>(0, 0) * HTz + vec -
                 G_.block<DIM_STATE, 7>(0, 0) * vec.block<7, 1>(0, 0);

      (*state_) += solution;
      auto &&rot_add = solution.block<3, 1>(0, 0);
      auto &&t_add = solution.block<3, 1>(3, 0);

      auto &&expo_add = solution.block<1, 1>(6, 0);
      // if ((rot_add.norm() * 57.3f < 0.001f) && (t_add.norm() * 100.0f <
      // 0.001f) && (expo_add.norm() < 0.001f)) EKF_end = true;
      if ((rot_add.norm() * 57.3f < 0.001f) && (t_add.norm() * 100.0f < 0.001f))
        EKF_end = true;
    } else {
      (*state_) = old_state;
      EKF_end = true;
    }

    update_ekf_time_ += omp_get_wtime() - t3;

    if (iteration == max_iterations_ || EKF_end) break;
  }
}

void VIOManager::UpdateFrameState(StatesGroup state) {
  M3D Rwi(state.rot_end);
  V3D Pwi(state.pos_end);
  Rcw_ = Rci_ * Rwi.transpose();
  Pcw_ = -Rci_ * Rwi.transpose() * Pwi + Pci_;
  new_frame_->T_f_w_ = SE3(Mat3ToSO3(Rcw_), Pcw_);
}

void VIOManager::PlotTrackedPoints() {
  int total_points = visual_submap_->voxel_points.size();
  if (total_points == 0) return;

  for (int i = 0; i < total_points; i++) {
    VisualPoint *pt = visual_submap_->voxel_points[i];
    V2D pc(new_frame_->w2c(pt->pos_));

    if (visual_submap_->errors[i] <= visual_submap_->propa_errors[i]) {
      // inlier_count++;
      cv::circle(img_cp_, cv::Point2f(pc[0], pc[1]), 7, cv::Scalar(0, 255, 0),
                 -1, 8);  // Green Sparse Align tracked
    } else {
      cv::circle(img_cp_, cv::Point2f(pc[0], pc[1]), 7, cv::Scalar(255, 0, 0),
                 -1, 8);  // Blue Sparse Align tracked
    }
  }
}

V3F VIOManager::GetInterpolatedPixel(cv::Mat img, V2D pc) {
  const float u_ref = pc[0];
  const float v_ref = pc[1];
  const int u_ref_i = floorf(pc[0]);
  const int v_ref_i = floorf(pc[1]);
  const float subpix_u_ref = (u_ref - u_ref_i);
  const float subpix_v_ref = (v_ref - v_ref_i);
  const float w_ref_tl = (1.0 - subpix_u_ref) * (1.0 - subpix_v_ref);
  const float w_ref_tr = subpix_u_ref * (1.0 - subpix_v_ref);
  const float w_ref_bl = (1.0 - subpix_u_ref) * subpix_v_ref;
  const float w_ref_br = subpix_u_ref * subpix_v_ref;
  uint8_t *img_ptr = (uint8_t *)img.data + ((v_ref_i)*width_ + (u_ref_i)) * 3;
  float B = w_ref_tl * img_ptr[0] + w_ref_tr * img_ptr[0 + 3] +
            w_ref_bl * img_ptr[width_ * 3] +
            w_ref_br * img_ptr[width_ * 3 + 0 + 3];
  float G = w_ref_tl * img_ptr[1] + w_ref_tr * img_ptr[1 + 3] +
            w_ref_bl * img_ptr[1 + width_ * 3] +
            w_ref_br * img_ptr[width_ * 3 + 1 + 3];
  float R = w_ref_tl * img_ptr[2] + w_ref_tr * img_ptr[2 + 3] +
            w_ref_bl * img_ptr[2 + width_ * 3] +
            w_ref_br * img_ptr[width_ * 3 + 2 + 3];
  V3F pixel(B, G, R);
  return pixel;
}

void VIOManager::DumpDataForColmap() {
  static int cnt = 1;
  std::ostringstream ss;
  ss << std::setw(5) << std::setfill('0') << cnt;
  std::string cnt_str = ss.str();
  std::string image_path =
      std::string(ROOT_DIR) + "Log/Colmap/images/" + cnt_str + ".png";

  cv::Mat img_rgb_undistort;
  pinhole_cam_->undistortImage(img_rgb_, img_rgb_undistort);
  cv::imwrite(image_path, img_rgb_undistort);

  Eigen::Quaterniond q(new_frame_->T_f_w_.rotationMatrix());
  Eigen::Vector3d t = new_frame_->T_f_w_.translation();
  fout_colmap_ << cnt << " " << std::fixed
               << std::setprecision(6)  // 保证浮点数精度为6位
               << q.w() << " " << q.x() << " " << q.y() << " " << q.z() << " "
               << t.x() << " " << t.y() << " " << t.z() << " " << 1
               << " "  // CAMERA_ID (假设相机ID为1)
               << cnt_str << ".png" << std::endl;
  fout_colmap_ << "0.0 0.0 -1" << std::endl;
  cnt++;
}

#if 0
void VIOManager::ProcessFrame(
    cv::Mat &img, std::vector<pointWithVar> &pv_list,
    const std::unordered_map<VOXEL_LOCATION, VoxelOctoTree *> &voxel_map,
    double img_time) {
  if (width_ != img.cols || height_ != img.rows) {
    if (img.empty()) printf("[ VIO ] Empty Image!\n");
    cv::resize(img, img,
               cv::Size(img.cols * image_resize_factor_,
                        img.rows * image_resize_factor_),
               0, 0, CV_INTER_LINEAR);
  }
  img_rgb_ = img.clone();
  img_cp_ = img.clone();

  // 转灰度图
  if (img.channels() == 3) cv::cvtColor(img, img, CV_BGR2GRAY);
  // 组建相机-图片帧
  new_frame_.reset(new Frame(cam_, img));

  UpdateFrameState(*state_);

  ResetGrid();

  double t1 = omp_get_wtime();
  // 提取当前帧的特征
  RetrieveFromVisualSparseMap(img, pv_list, voxel_map);

  double t2 = omp_get_wtime();

  // 更新ESIEKF
  ComputeJacobianAndUpdateEKF(img);

  double t3 = omp_get_wtime();

  // 第一次执行时从此处开始，生成特征地图
  GenerateVisualMapPoints(img, pv_list);

  double t4 = omp_get_wtime();

  // 绘制被跟踪的特征点
  PlotTrackedPoints();

  // 实际不执行
  if (plot_flag_) ProjectPatchFromRefToCur();

  double t5 = omp_get_wtime();

  // 更新特征地图
  UpdateVisualMapPoints(img);

  double t6 = omp_get_wtime();
  // 更新相关图像块
  UpdateReferencePatch(voxel_map);

  double t7 = omp_get_wtime();

  // 保存高斯泼溅需要的ColMap
  if (colmap_output_en_) DumpDataForColmap();

  frame_count_++;
  ave_total_ = ave_total_ * (frame_count_ - 1) / frame_count_ +
               (t7 - t1 - (t5 - t4)) / frame_count_;

  printf(
      "\033[1;34m+-------------------------------------------------------------"
      "+\033[0m\n");
  printf(
      "\033[1;34m|                         VIO Time                            "
      "|\033[0m\n");
  printf(
      "\033[1;34m+-------------------------------------------------------------"
      "+\033[0m\n");
  printf("\033[1;34m| %-29s | %-27zu |\033[0m\n", "Sparse Map Size",
         voxel_map.size());
  printf(
      "\033[1;34m+-------------------------------------------------------------"
      "+\033[0m\n");
  printf("\033[1;34m| %-29s | %-27s |\033[0m\n", "Algorithm Stage",
         "Time (secs)");
  printf(
      "\033[1;34m+-------------------------------------------------------------"
      "+\033[0m\n");
  printf("\033[1;32m| %-29s | %-27lf |\033[0m\n", "retrieveFromVisualSparseMap",
         t2 - t1);
  printf("\033[1;32m| %-29s | %-27lf |\033[0m\n", "computeJacobianAndUpdateEKF",
         t3 - t2);
  printf("\033[1;32m| %-27s   | %-27lf |\033[0m\n", "-> computeJacobian",
         compute_jacobian_time_);
  printf("\033[1;32m| %-27s   | %-27lf |\033[0m\n", "-> updateEKF",
         update_ekf_time_);
  printf("\033[1;32m| %-29s | %-27lf |\033[0m\n", "generateVisualMapPoints",
         t4 - t3);
  printf("\033[1;32m| %-29s | %-27lf |\033[0m\n", "updateVisualMapPoints",
         t6 - t5);
  printf("\033[1;32m| %-29s | %-27lf |\033[0m\n", "updateReferencePatch",
         t7 - t6);
  printf(
      "\033[1;34m+-------------------------------------------------------------"
      "+\033[0m\n");
  printf("\033[1;32m| %-29s | %-27lf |\033[0m\n", "Current Total Time",
         t7 - t1 - (t5 - t4));
  printf("\033[1;32m| %-29s | %-27lf |\033[0m\n", "Average Total Time",
         ave_total_);
  printf(
      "\033[1;34m+-------------------------------------------------------------"
      "+\033[0m\n");
}
#endif

void VIOManager::ProcessFrame(
    cv::Mat &img, std::vector<pointWithVar> &pv_list,
    const std::unordered_map<VOXEL_LOCATION,
                             typename std::list<VMData>::iterator> &vm_map) {
  if (width_ != img.cols || height_ != img.rows) {
    if (img.empty()) printf("[ VIO ] Empty Image!\n");
    cv::resize(img, img,
               cv::Size(img.cols * image_resize_factor_,
                        img.rows * image_resize_factor_),
               0, 0, CV_INTER_LINEAR);
  }
  img_rgb_ = img.clone();
  img_cp_ = img.clone();

  // 转灰度图
  if (img.channels() == 3) cv::cvtColor(img, img, CV_BGR2GRAY);
  // 组建相机-图片帧
  new_frame_.reset(new ImageFrame(cam_, img));

  UpdateFrameState(*state_);

  ResetGrid();

  double t1 = omp_get_wtime();
  // 提取当前帧的特征
  RetrieveFromVisualSparseMapLRU(img, pv_list, vm_map);

  double t2 = omp_get_wtime();

  // 更新ESIEKF
  ComputeJacobianAndUpdateEKF(img);

  double t3 = omp_get_wtime();

  // 第一次执行时从此处开始，生成特征地图
  GenerateVisualMapPoints(img, pv_list);

  double t4 = omp_get_wtime();

  // 绘制被跟踪的特征点
  PlotTrackedPoints();

  // 实际不执行
  if (plot_flag_) ProjectPatchFromRefToCur();

  double t5 = omp_get_wtime();

  // 更新特征地图
  UpdateVisualMapPoints(img);

  double t6 = omp_get_wtime();
  // 更新相关图像块
  UpdateReferencePatch(vm_map);

  double t7 = omp_get_wtime();

  // 保存高斯泼溅需要的ColMap
  if (colmap_output_en_) DumpDataForColmap();

  frame_count_++;
  ave_total_ = ave_total_ * (frame_count_ - 1) / frame_count_ +
               (t7 - t1 - (t5 - t4)) / frame_count_;

  printf(
      "\033[1;34m+-------------------------------------------------------------"
      "+\033[0m\n");
  printf(
      "\033[1;34m|                         VIO Time                            "
      "|\033[0m\n");
  printf(
      "\033[1;34m+-------------------------------------------------------------"
      "+\033[0m\n");
  printf("\033[1;34m| %-29s | %-27zu |\033[0m\n", "Sparse Map Size",
         vm_map.size());
  printf(
      "\033[1;34m+-------------------------------------------------------------"
      "+\033[0m\n");
  printf("\033[1;34m| %-29s | %-27s |\033[0m\n", "Algorithm Stage",
         "Time (secs)");
  printf(
      "\033[1;34m+-------------------------------------------------------------"
      "+\033[0m\n");
  printf("\033[1;32m| %-29s | %-27lf |\033[0m\n", "retrieveFromVisualSparseMap",
         t2 - t1);
  printf("\033[1;32m| %-29s | %-27lf |\033[0m\n", "computeJacobianAndUpdateEKF",
         t3 - t2);
  printf("\033[1;32m| %-27s   | %-27lf |\033[0m\n", "-> computeJacobian",
         compute_jacobian_time_);
  printf("\033[1;32m| %-27s   | %-27lf |\033[0m\n", "-> updateEKF",
         update_ekf_time_);
  printf("\033[1;32m| %-29s | %-27lf |\033[0m\n", "generateVisualMapPoints",
         t4 - t3);
  printf("\033[1;32m| %-29s | %-27lf |\033[0m\n", "updateVisualMapPoints",
         t6 - t5);
  printf("\033[1;32m| %-29s | %-27lf |\033[0m\n", "updateReferencePatch",
         t7 - t6);
  printf(
      "\033[1;34m+-------------------------------------------------------------"
      "+\033[0m\n");
  printf("\033[1;32m| %-29s | %-27lf |\033[0m\n", "Current Total Time",
         t7 - t1 - (t5 - t4));
  printf("\033[1;32m| %-29s | %-27lf |\033[0m\n", "Average Total Time",
         ave_total_);
  printf(
      "\033[1;34m+-------------------------------------------------------------"
      "+\033[0m\n");
}