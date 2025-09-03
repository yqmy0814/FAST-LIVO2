/*
This file is part of FAST-LIVO2: Fast, Direct LiDAR-Inertial-Visual Odometry.

Developer: Chunran Zheng <zhengcr@connect.hku.hk>

For commercial use, please contact me at <zhengcr@connect.hku.hk> or
Prof. Fu Zhang at <fuzhang@hku.hk>.

This file is subject to the terms and conditions outlined in the 'LICENSE' file,
which is included as part of this source code package.
*/

#ifndef VIO_H_
#define VIO_H_

#include <opencv2/imgproc/imgproc_c.h>
#include <pcl/filters/voxel_grid.h>
#include <vikit/math_utils.h>
#include <vikit/pinhole_camera.h>
#include <vikit/robust_cost.h>
#include <vikit/vision.h>

#include <set>

#include "image_frame.h"
#include "voxel_map.h"

struct SubSparseMap {
  std::vector<float> propa_errors;
  std::vector<float> errors;
  std::vector<std::vector<float>> warp_patch;
  std::vector<int> search_levels;
  std::vector<VisualPoint *> voxel_points;
  std::vector<double> inv_expo_list;
  std::vector<pointWithVar> add_from_voxel_map;

  SubSparseMap() {
    propa_errors.reserve(SIZE_LARGE);
    errors.reserve(SIZE_LARGE);
    warp_patch.reserve(SIZE_LARGE);
    search_levels.reserve(SIZE_LARGE);
    voxel_points.reserve(SIZE_LARGE);
    inv_expo_list.reserve(SIZE_LARGE);
    add_from_voxel_map.reserve(SIZE_SMALL);
  };

  void reset() {
    propa_errors.clear();
    errors.clear();
    warp_patch.clear();
    search_levels.clear();
    voxel_points.clear();
    inv_expo_list.clear();
    add_from_voxel_map.clear();
  }
};

class Warp {
 public:
  Eigen::Matrix2d A_cur_ref;
  int search_level;
  Warp(int level, Eigen::Matrix2d warp_matrix)
      : search_level(level), A_cur_ref(warp_matrix) {}
  ~Warp() {}
};

class VOXEL_POINTS {
 public:
  std::vector<VisualPoint *> voxel_points;
  int count;
  VOXEL_POINTS(int num) : count(num) {}
  ~VOXEL_POINTS() {
    for (VisualPoint *vp : voxel_points) {
      if (vp != nullptr) {
        delete vp;
        vp = nullptr;
      }
    }
  }
};
using VPData = std::pair<VOXEL_LOCATION, VOXEL_POINTS *>;

class VIOManager {
 public:
  int grid_size_;
  vk::AbstractCamera *cam_;
  vk::PinholeCamera *pinhole_cam_;
  StatesGroup *state_;
  StatesGroup *state_propagat_;
  M3D Rli_, Rci_, Rcl_, Rcw_, Jdphi_dR_, Jdp_dt_, Jdp_dR_;
  V3D Pli_, Pci_, Pcl_, Pcw_;
  std::vector<int> grid_num_;
  std::vector<int> map_index_;
  std::vector<int> border_flag_;
  std::vector<int> update_flag_;
  std::vector<float> map_dist_;
  std::vector<float> scan_value_;
  std::vector<float> patch_buffer_;
  bool normal_en_, inverse_composition_en_, exposure_estimate_en_, raycast_en_,
      has_ref_patch_cache_;
  bool ncc_en_ = false, colmap_output_en_ = false;

  int width_, height_, grid_n_width_, grid_n_height_ = 17, length_;
  double image_resize_factor_;
  double fx_, fy_, cx_, cy_;
  int patch_pyrimid_level_, patch_size_, patch_size_total_, patch_size_half_,
      border_, warp_len_;
  int max_iterations_, total_points_;

  double img_point_cov_, outlier_threshold_, ncc_thre_;

  SubSparseMap *visual_submap_;
  std::vector<std::vector<V3D>> rays_with_sample_points_;

  double compute_jacobian_time_, update_ekf_time_;
  double ave_total_ = 0;
  // double ave_build_residual_time = 0;
  // double ave_ekf_time = 0;

  int frame_count_ = 0;
  bool plot_flag_;

  Eigen::Matrix<double, DIM_STATE, DIM_STATE> G_, H_T_H_;
  Eigen::MatrixXd K_, H_sub_inv_;

  std::ofstream fout_camera_, fout_colmap_;
  // std::unordered_map<VOXEL_LOCATION, VOXEL_POINTS *> feat_map_;
  std::unordered_map<VOXEL_LOCATION, int> sub_feat_map_;
  std::unordered_map<int, Warp *> warp_map_;

  std::unordered_map<VOXEL_LOCATION, typename std::list<VPData>::iterator>
      vp_map_;
  std::list<VPData> vp_data_;
  int lru_size_ = 1000000;
  std::vector<VisualPoint *> retrieve_voxel_points_;
  std::vector<pointWithVar> append_voxel_points_;
  FramePtr new_frame_;
  cv::Mat img_cp_, img_rgb_, img_test_;

  enum CellType { TYPE_MAP = 1, TYPE_POINTCLOUD, TYPE_UNKNOWN };

  VIOManager();

  ~VIOManager();

  void UpdateStateInverse(cv::Mat img, int level);

  void UpdateState(cv::Mat img, int level);

  // void ProcessFrame(
  //     cv::Mat &img, std::vector<pointWithVar> &pg,
  //     const std::unordered_map<VOXEL_LOCATION, VoxelOctoTree *> &feat_map,
  //     double img_time);

  void ProcessFrame(
      cv::Mat &img, std::vector<pointWithVar> &pv_list,
      const std::unordered_map<VOXEL_LOCATION,
                               typename std::list<VMData>::iterator> &vm_map);

  // void RetrieveFromVisualSparseMap(
  //     cv::Mat img, std::vector<pointWithVar> &pg,
  //     const std::unordered_map<VOXEL_LOCATION, VoxelOctoTree *> &plane_map);

  void RetrieveFromVisualSparseMapLRU(
      cv::Mat img, std::vector<pointWithVar> &pv_list,
      const std::unordered_map<VOXEL_LOCATION,
                               typename std::list<VMData>::iterator> &vm_map);

  void GenerateVisualMapPoints(cv::Mat img, std::vector<pointWithVar> &pg);

  void SetImuToLidarExtrinsic(const V3D &transl, const M3D &rot);

  void SetLidarToCameraExtrinsic(std::vector<double> &R,
                                 std::vector<double> &P);

  void InitializeVIO();

  void GetImagePatch(cv::Mat img, V2D pc, float *patch_tmp, int level);

  void ComputeProjectionJacobian(V3D p, MD(2, 3) & J);

  void ComputeJacobianAndUpdateEKF(cv::Mat img);

  void ResetGrid();

  void UpdateVisualMapPoints(cv::Mat img);

  void GetWarpMatrixAffine(const vk::AbstractCamera &cam,
                           const Eigen::Vector2d &px_ref,
                           const Eigen::Vector3d &f_ref, const double depth_ref,
                           const SE3 &T_cur_ref, const int level_ref,
                           const int pyramid_level, const int halfpatch_size,
                           Eigen::Matrix2d &A_cur_ref);

  void GetWarpMatrixAffineHomography(const vk::AbstractCamera &cam,
                                     const V2D &px_ref, const V3D &xyz_ref,
                                     const V3D &normal_ref,
                                     const SE3 &T_cur_ref, const int level_ref,
                                     Eigen::Matrix2d &A_cur_ref);

  void WarpAffine(const Eigen::Matrix2d &A_cur_ref, const cv::Mat &img_ref,
                  const Eigen::Vector2d &px_ref, const int level_ref,
                  const int search_level, const int pyramid_level,
                  const int halfpatch_size, float *patch);

  // void InsertPointIntoFeatureMap(VisualPoint *pt_new);

  void InsertPointIntoFeatureMapLRU(VisualPoint *pt_new);

  void PlotTrackedPoints();

  void UpdateFrameState(StatesGroup state);

  void ProjectPatchFromRefToCur();

  void UpdateReferencePatch(
      const std::unordered_map<VOXEL_LOCATION, VoxelOctoTree *> &plane_map);

  void UpdateReferencePatch(
      const std::unordered_map<VOXEL_LOCATION,
                               typename std::list<VMData>::iterator> &vm_map);

  void PrecomputeReferencePatches(int level);

  void DumpDataForColmap();

  double CalculateNCC(float *ref_patch, float *cur_patch, int patch_size);

  int GetBestSearchLevel(const Eigen::Matrix2d &A_cur_ref, const int max_level);

  V3F GetInterpolatedPixel(cv::Mat img, V2D pc);

  SE3 se3_imu_lidar_, se3_lidar_cam_, se3_imu_cam_;
};
typedef std::shared_ptr<VIOManager> VIOManagerPtr;

#endif  // VIO_H_