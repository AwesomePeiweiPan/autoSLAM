//
// Created by xiang on 22-12-29.
//

#include <gflags/gflags.h>   //允许程序定义命令行标志并且解析他们 对应下面的ParseCommandLineFlags(&argc,&argb,true);
#include <glog/logging.h>    //提供日志级别，并且可以将日志信息输出到控制台或文件:LOG(INFO);LOG(WARNING);LOG(ERROR);LOG(FATAL)
                             //初始化：InitGoogleLogging,FLAGS_stderrthreshold;FLAGS_colorlogtostderr
                             //记录日志：LOG(INFO)

#include "common/eigen_types.h"           //自己定义的头文件，里面包含了Eigen库的头文件
#include "common/math_utils.h"            //自己定义的头文件，包含库的头文件
#include "tools/ui/pangolin_window.h"     //自己定义的头文件，包含库的头文件

/// 本节程序演示一个正在作圆周运动的车辆
/// 车辆的角速度与线速度可以在flags中设置

//gflags库的自定义参数
DEFINE_double(angular_velocity, 10.0, "角速度（角度）制");    
DEFINE_double(linear_velocity, 5.0, "车辆前进线速度 m/s");
DEFINE_bool(use_quaternion, false, "是否使用四元数计算");

int main(int argc, char** argv) {
    google::InitGoogleLogging(argv[0]);     //程序的名称或者路径，方便初始化日志的时候作为日志的标识符
    FLAGS_stderrthreshold = google::INFO;   //日志的级别，INFO，WARNING等
    FLAGS_colorlogtostderr = true;          //彩色日志输出
    //argc argument count,命令行的参数数量;./program -a -b value, argc=4;
    //argv argument vector,命令行的参数;argv[0]:程序名称或者路径; argv[1]到argv[argc-1]:传递给程序的其他命令行参数
    //true:argv移除已处理的标志
    google::ParseCommandLineFlags(&argc, &argv, true);

    /// 可视化
    sad::ui::PangolinWindow ui;
    if (ui.Init() == false) {
        return -1;
    }

    double angular_velocity_rad = FLAGS_angular_velocity * sad::math::kDEG2RAD;  // 弧度制角速度
    SE3 pose;                                                                    // TWB表示的位姿  //Sophus库，四元数表示旋转
    Vec3d omega(0, 0, angular_velocity_rad);                                     // 角速度矢量     //Eigen库
    Vec3d v_body(FLAGS_linear_velocity, 0, 0);                                   // 本体系速度     //Eigen库
    const double dt = 0.05;                                                      // 每次更新的时间

    while (ui.ShouldQuit() == false) {
        // 更新自身位置
        Vec3d v_world = pose.so3() * v_body;
        pose.translation() += v_world * dt;

        // 更新自身旋转
        if (FLAGS_use_quaternion) {
            // unit_quaternion()转化为单位四元数
            Quatd q = pose.unit_quaternion() * Quatd(1, 0.5 * omega[0] * dt, 0.5 * omega[1] * dt, 0.5 * omega[2] * dt);
            // 归一化为单位四元数
            q.normalize();
            pose.so3() = SO3(q); //一个是四元数q，一个是SO3，还是有区别的
        } else {
            pose.so3() = pose.so3() * SO3::exp(omega * dt);  //SO3::exp(omega * dt)直接转换成四元数
        }

        LOG(INFO) << "pose: " << pose.translation().transpose();
        ui.UpdateNavState(sad::NavStated(0, pose, v_world));

        usleep(dt * 1e6);
    }

    ui.Quit();
    return 0;
}