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

#include <boost/noncopyable.hpp>
#include <vikit/abstract_camera.h>

class VisualPoint;
struct Feature;

typedef list<Feature *> Features;
typedef vector<cv::Mat> ImgPyr;

/// A frame saves the image, the associated features and the estimated pose.
class Frame : boost::noncopyable
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  static int frame_counter_; // 帧计数器，用于生成帧的唯一ID
  int id_;                   // 帧的唯一ID
  vk::AbstractCamera *cam_;  // 相机模型
  SE3 T_f_w_;                // 相机在世界系下的位姿
  SE3 T_f_w_prior_;          // IMU先验位姿（没有用到）
  cv::Mat img_;              // 帧的图像
  Features fts_;             // 保存特征的链表

  Frame(vk::AbstractCamera *cam, const cv::Mat &img);
  ~Frame();

  /// 初始化新帧并创建图像金字塔。
  void initFrame(const cv::Mat &img);

  /// 返回点观测的数量。
  inline size_t nObs() const { return fts_.size(); }

  /// 将世界坐标系 (w) 中的点坐标转换为相机像素坐标系 (c) 中的坐标。
  inline Vector2d w2c(const Vector3d &xyz_w) const { return cam_->world2cam(T_f_w_ * xyz_w); }

  /// 使用 IMU 先验姿态将世界坐标系 (w) 中的点坐标转换为相机像素坐标系 (c) 中的坐标。
  inline Vector2d w2c_prior(const Vector3d &xyz_w) const { return cam_->world2cam(T_f_w_prior_ * xyz_w); }
  
  /// 将相机像素坐标系 (c) 中的坐标转换为帧单位球坐标系 (f) 中的坐标。
  inline Vector3d c2f(const Vector2d &px) const { return cam_->cam2world(px[0], px[1]); }

  /// 将相机像素坐标系 (c) 中的坐标转换为帧单位球坐标系 (f) 中的坐标。
  inline Vector3d c2f(const double x, const double y) const { return cam_->cam2world(x, y); }

  /// 将世界坐标系 (w) 中的点坐标转换为相机坐标系 (f) 中的坐标。
  inline Vector3d w2f(const Vector3d &xyz_w) const { return T_f_w_ * xyz_w; }

  /// 将帧单位球坐标系 (f) 中的点坐标转换为世界坐标系 (w) 中的坐标。
  inline Vector3d f2w(const Vector3d &f) const { return T_f_w_.inverse() * f; }

  /// 将单位球坐标系 (f) 中的点投影到相机像素坐标系 (c) 中。
  inline Vector2d f2c(const Vector3d &f) const { return cam_->world2cam(f); }

  /// 返回帧在 (w) 世界坐标系中的姿态。
  inline Vector3d pos() const { return T_f_w_.inverse().translation(); }
};

typedef std::unique_ptr<Frame> FramePtr;

/// Some helper functions for the frame object.
namespace frame_utils
{

/// 创建一个由半采样图像组成的图像金字塔。
void createImgPyramid(const cv::Mat &img_level_0, int n_levels, ImgPyr &pyr);

} // namespace frame_utils

#endif // LIVO_FRAME_H_
