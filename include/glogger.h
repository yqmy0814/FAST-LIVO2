#ifndef GLOGGER_H_
#define GLOGGER_H_

#include <gflags/gflags.h>
#include <glog/logging.h>

#include <filesystem>
#include <thread>

class GLogger {
 public:
  // 构造函数
  GLogger(int argc, char** argv, const std::string& log_warning_path,
          const std::string& log_path) {
    google::InitGoogleLogging(argv[0]);
    // 解析输入的命令行参数
    google::ParseCommandLineFlags(&argc, &argv, true);

    // 设置日志路径
    if (log_path != "") {
      // 检查并创建信息日志目录
      CheckAndCreateDirectory(log_path, "Log directory");
      google::SetLogDestination(google::INFO, log_path.c_str());
      LOG(WARNING) << "Log path: " << log_path.c_str();
    }
    if (log_warning_path != "") {
      // 检查并创建警告日志目录
      CheckAndCreateDirectory(log_warning_path, "Log warning directory");
      google::SetLogDestination(google::WARNING, log_warning_path.c_str());
      LOG(WARNING) << "Log warning path: " << log_warning_path.c_str();
    }

    // 安装失败信号处理器，让glog输出失败
    google::InstallFailureSignalHandler();
    FLAGS_alsologtostderr = true;
    FLAGS_colorlogtostderr = true;
    FLAGS_max_log_size = 50;
  }

  // 析构函数
  ~GLogger() {
    // 关闭Google日志库
    google::ShutdownGoogleLogging();
  }

 private:
  // 检查并创建目录
  void CheckAndCreateDirectory(const std::string& path,
                               const std::string& log_type) {
    std::string dir_path = path;
    if (dir_path.back() == '/') {
      dir_path.erase(dir_path.end() - 1);
    }
    if (!std::filesystem::is_directory(dir_path)) {
      LOG(ERROR) << log_type << " can not be found: " << dir_path;
      if (std::filesystem::create_directories(dir_path)) {
        LOG(INFO) << "Create " << log_type << ": " << dir_path;
      } else {
        LOG(WARNING) << "Failed to create " << log_type << ": " << dir_path;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }
  }
};

#endif  // GLOGGER_H_