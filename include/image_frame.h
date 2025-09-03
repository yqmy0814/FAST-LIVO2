/*
This file is part of FAST-LIVO2: Fast, Direct LiDAR-Inertial-Visual Odometry.

Developer: Chunran Zheng <zhengcr@connect.hku.hk>

For commercial use, please contact me at <zhengcr@connect.hku.hk> or
Prof. Fu Zhang at <fuzhang@hku.hk>.

This file is subject to the terms and conditions outlined in the 'LICENSE' file,
which is included as part of this source code package.
*/

#ifndef LIVO_FRAME_H_
#define LIVO_FRAME_H_

#include <vikit/abstract_camera.h>

#include <boost/noncopyable.hpp>

#include "common_lib.h"

class VisualPoint;
// A salient image region that is tracked across frames.
struct Feature {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  enum FeatureType { CORNER, EDGELET };
  int id_;
  FeatureType type_;    //!< Type can be corner or edgelet.
  cv::Mat img_;         //!< Image associated with the patch feature
  Eigen::Vector2d px_;  //!< Coordinates in pixels on pyramid level 0.
  Eigen::Vector3d f_;   //!< Unit-bearing vector of the patch feature.
  int level_;  //!< Image pyramid level where patch feature was extracted.
  VisualPoint
      *point_;  //!< Pointer to 3D point which corresponds to the patch feature.
  Eigen::Vector2d
      grad_;      //!< Dominant gradient direction for edglets, normalized.
  SE3 T_f_w_;     //!< Pose of the frame where the patch feature was extracted.
  float *patch_;  //!< Pointer to the image patch data.
  float score_;   //!< Score of the patch feature.
  float mean_;    //!< Mean intensity of the image patch feature, used for
                  //!< normalization.
  double inv_expo_time_;  //!< Inverse exposure time of the image where the
                          //!< patch feature was extracted.

  Feature(VisualPoint *_point, float *_patch, const Eigen::Vector2d &_px,
          const Eigen::Vector3d &_f, const SE3 &_T_f_w, int _level)
      : type_(CORNER),
        px_(_px),
        f_(_f),
        T_f_w_(_T_f_w),
        mean_(0),
        score_(0),
        level_(_level),
        patch_(_patch),
        point_(_point) {}

  inline Eigen::Vector3d pos() const { return T_f_w_.inverse().translation(); }

  ~Feature() {
    // ROS_WARN("The feature %d has been destructed.", id_);
    delete[] patch_;
  }
};

class VisualPoint : boost::noncopyable {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  Eigen::Vector3d pos_;                 //点的位置
  Eigen::Vector3d normal_;              //所在平面法向量
  Eigen::Matrix3d normal_information_;  //法向量协方差矩阵的逆
  Eigen::Vector3d previous_normal_;     //上次更新的法向量
  std::list<Feature *> obs_;            //所有的观察到该点的图像块
  Eigen::Matrix3d covariance_;          //点的协方差
  bool is_converged_;                   //是否收敛
  bool is_normal_initialized_;          //法向量是否初始化
  bool has_ref_patch_;                  //是否存在参考图像块
  Feature *ref_patch;                   //参考图像块

  VisualPoint(const Eigen::Vector3d &pos);
  ~VisualPoint();
  void findMinScoreFeature(const Eigen::Vector3d &framepos,
                           Feature *&ftr) const;
  void deleteNonRefPatchFeatures();
  void deleteFeatureRef(Feature *ftr);
  void addFrameRef(Feature *ftr);
  bool getCloseViewObs(const Eigen::Vector3d &pos, Feature *&obs,
                       const Eigen::Vector2d &cur_px) const;
};

typedef std::list<Feature *> Features;
typedef std::vector<cv::Mat> ImgPyr;
/// A frame saves the image, the associated features and the estimated pose.
class ImageFrame : boost::noncopyable {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  static int frame_counter_;  // 帧计数器，用于生成帧的唯一ID
  int id_;                    // 帧的唯一ID
  vk::AbstractCamera *cam_;   // 相机模型
  SE3 T_f_w_;                 // 相机在世界系下的位姿
  SE3 T_f_w_prior_;           // IMU先验位姿（没有用到）
  cv::Mat img_;               // 帧的图像
  Features fts_;              // 保存特征的链表

  ImageFrame(vk::AbstractCamera *cam, const cv::Mat &img);
  ~ImageFrame();

  /// 初始化新帧并创建图像金字塔。
  void initFrame(const cv::Mat &img);

  /// 返回点观测的数量。
  inline size_t nObs() const { return fts_.size(); }

  /// 将世界坐标系 (w) 中的点坐标转换为相机像素坐标系 (c) 中的坐标。
  inline Eigen::Vector2d w2c(const Eigen::Vector3d &xyz_w) const {
    return cam_->world2cam(T_f_w_ * xyz_w);
  }

  /// 使用 IMU 先验姿态将世界坐标系 (w) 中的点坐标转换为相机像素坐标系 (c)
  /// 中的坐标。
  inline Eigen::Vector2d w2c_prior(const Eigen::Vector3d &xyz_w) const {
    return cam_->world2cam(T_f_w_prior_ * xyz_w);
  }

  /// 将相机像素坐标系 (c) 中的坐标转换为帧单位球坐标系 (f) 中的坐标。
  inline Eigen::Vector3d c2f(const Eigen::Vector2d &px) const {
    return cam_->cam2world(px[0], px[1]);
  }

  /// 将相机像素坐标系 (c) 中的坐标转换为帧单位球坐标系 (f) 中的坐标。
  inline Eigen::Vector3d c2f(const double x, const double y) const {
    return cam_->cam2world(x, y);
  }

  /// 将世界坐标系 (w) 中的点坐标转换为相机坐标系 (f) 中的坐标。
  inline Eigen::Vector3d w2f(const Eigen::Vector3d &xyz_w) const {
    return T_f_w_ * xyz_w;
  }

  /// 将帧单位球坐标系 (f) 中的点坐标转换为世界坐标系 (w) 中的坐标。
  inline Eigen::Vector3d f2w(const Eigen::Vector3d &f) const {
    return T_f_w_.inverse() * f;
  }

  /// 将单位球坐标系 (f) 中的点投影到相机像素坐标系 (c) 中。
  inline Eigen::Vector2d f2c(const Eigen::Vector3d &f) const {
    return cam_->world2cam(f);
  }

  /// 返回帧在 (w) 世界坐标系中的姿态。
  inline Eigen::Vector3d pos() const { return T_f_w_.inverse().translation(); }
};

typedef std::unique_ptr<ImageFrame> FramePtr;

/// Some helper functions for the frame object.
namespace frame_utils {

/// 创建一个由半采样图像组成的图像金字塔。
void createImgPyramid(const cv::Mat &img_level_0, int n_levels, ImgPyr &pyr);

}  // namespace frame_utils

#endif  // LIVO_FRAME_H_
