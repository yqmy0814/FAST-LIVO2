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

#include "voxel_map.h"
#include "feature.h"
#include <opencv2/imgproc/imgproc_c.h>
#include <pcl/filters/voxel_grid.h>
#include <set>
#include <vikit/math_utils.h>
#include <vikit/robust_cost.h>
#include <vikit/vision.h>
#include <vikit/pinhole_camera.h>

struct SubSparseMap
{
  vector<float> propa_errors;
  vector<float> errors;
  vector<vector<float>> warp_patch;
  vector<int> search_levels;
  vector<VisualPoint *> voxel_points;
  vector<double> inv_expo_list;
  vector<pointWithVar> add_from_voxel_map;

  SubSparseMap()
  {
    propa_errors.reserve(SIZE_LARGE);
    errors.reserve(SIZE_LARGE);
    warp_patch.reserve(SIZE_LARGE);
    search_levels.reserve(SIZE_LARGE);
    voxel_points.reserve(SIZE_LARGE);
    inv_expo_list.reserve(SIZE_LARGE);
    add_from_voxel_map.reserve(SIZE_SMALL);
  };

  void reset()
  {
    propa_errors.clear();
    errors.clear();
    warp_patch.clear();
    search_levels.clear();
    voxel_points.clear();
    inv_expo_list.clear();
    add_from_voxel_map.clear();
  }
};

class Warp
{
public:
  Matrix2d A_cur_ref;
  int search_level;
  Warp(int level, Matrix2d warp_matrix) : search_level(level), A_cur_ref(warp_matrix) {}
  ~Warp() {}
};

class VOXEL_POINTS
{
public:
  std::vector<VisualPoint *> voxel_points;
  int count;
  VOXEL_POINTS(int num) : count(num) {}
  ~VOXEL_POINTS() 
  { 
    for (VisualPoint* vp : voxel_points) 
    {
      if (vp != nullptr) { delete vp; vp = nullptr; }
    }
  }
};

class VIOManager
{
public:
  int grid_size_;
  vk::AbstractCamera *cam_;
  vk::PinholeCamera *pinhole_cam_;
  StatesGroup *state_;
  StatesGroup *state_propagat_;
  M3D Rli_, Rci_, Rcl_, Rcw_, Jdphi_dR_, Jdp_dt_, Jdp_dR_;
  V3D Pli_, Pci_, Pcl_, Pcw_;
  vector<int> grid_num_;
  vector<int> map_index_;
  vector<int> border_flag_;
  vector<int> update_flag_;
  vector<float> map_dist_;
  vector<float> scan_value_;
  vector<float> patch_buffer_;
  bool normal_en_, inverse_composition_en_, exposure_estimate_en_, raycast_en_, has_ref_patch_cache_;
  bool ncc_en_ = false, colmap_output_en_ = false;

  int width_, height_, grid_n_width_, grid_n_height_, length_;
  double image_resize_factor_;
  double fx_, fy_, cx_, cy_;
  int patch_pyrimid_level_, patch_size_, patch_size_total_, patch_size_half_, border_, warp_len_;
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

  Matrix<double, DIM_STATE, DIM_STATE> G_, H_T_H_;
  MatrixXd K_, H_sub_inv_;

  ofstream fout_camera_, fout_colmap_;
  unordered_map<VOXEL_LOCATION, VOXEL_POINTS *> feat_map_;
  unordered_map<VOXEL_LOCATION, int> sub_feat_map_; 
  unordered_map<int, Warp *> warp_map_;
  vector<VisualPoint *> retrieve_voxel_points_;
  vector<pointWithVar> append_voxel_points_;
  FramePtr new_frame_;
  cv::Mat img_cp_, img_rgb_, img_test_;

  enum CellType
  {
    TYPE_MAP = 1,
    TYPE_POINTCLOUD,
    TYPE_UNKNOWN
  };

  VIOManager();
  ~VIOManager();
  void UpdateStateInverse(cv::Mat img, int level);
  void UpdateState(cv::Mat img, int level);
  void ProcessFrame(cv::Mat &img, vector<pointWithVar> &pg, const unordered_map<VOXEL_LOCATION, VoxelOctoTree *> &feat_map, double img_time);
  void RetrieveFromVisualSparseMap(cv::Mat img, vector<pointWithVar> &pg, const unordered_map<VOXEL_LOCATION, VoxelOctoTree *> &plane_map);
  void GenerateVisualMapPoints(cv::Mat img, vector<pointWithVar> &pg);
  void SetImuToLidarExtrinsic(const V3D &transl, const M3D &rot);
  void SetLidarToCameraExtrinsic(vector<double> &R, vector<double> &P);
  void InitializeVIO();
  void GetImagePatch(cv::Mat img, V2D pc, float *patch_tmp, int level);
  void ComputeProjectionJacobian(V3D p, MD(2, 3) & J);
  void ComputeJacobianAndUpdateEKF(cv::Mat img);
  void ResetGrid();
  void UpdateVisualMapPoints(cv::Mat img);
  void GetWarpMatrixAffine(const vk::AbstractCamera &cam, const Vector2d &px_ref, const Vector3d &f_ref, const double depth_ref, const SE3 &T_cur_ref,
                           const int level_ref, 
                           const int pyramid_level, const int halfpatch_size, Matrix2d &A_cur_ref);
  void GetWarpMatrixAffineHomography(const vk::AbstractCamera &cam, const V2D &px_ref,
                                     const V3D &xyz_ref, const V3D &normal_ref, const SE3 &T_cur_ref, const int level_ref, Matrix2d &A_cur_ref);
  void WarpAffine(const Matrix2d &A_cur_ref, const cv::Mat &img_ref, const Vector2d &px_ref, const int level_ref, const int search_level,
                  const int pyramid_level, const int halfpatch_size, float *patch);
  void InsertPointIntoFeatureMap(VisualPoint *pt_new);
  void PlotTrackedPoints();
  void UpdateFrameState(StatesGroup state);
  void ProjectPatchFromRefToCur(const unordered_map<VOXEL_LOCATION, VoxelOctoTree *> &plane_map);
  void UpdateReferencePatch(const unordered_map<VOXEL_LOCATION, VoxelOctoTree *> &plane_map);
  void PrecomputeReferencePatches(int level);
  void DumpDataForColmap();
  double CalculateNCC(float *ref_patch, float *cur_patch, int patch_size);
  int GetBestSearchLevel(const Matrix2d &A_cur_ref, const int max_level);
  V3F GetInterpolatedPixel(cv::Mat img, V2D pc);
  
  // void resetRvizDisplay();
  // deque<VisualPoint *> map_cur_frame;
  // deque<VisualPoint *> sub_map_ray;
  // deque<VisualPoint *> sub_map_ray_fov;
  // deque<VisualPoint *> visual_sub_map_cur;
  // deque<VisualPoint *> visual_converged_point;
  // std::vector<std::vector<V3D>> sample_points;

  // PointCloudXYZIN::Ptr pg_down;
  // pcl::VoxelGrid<PointType> downSizeFilter;
};
typedef std::shared_ptr<VIOManager> VIOManagerPtr;

#endif // VIO_H_