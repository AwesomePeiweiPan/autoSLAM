//
// Created by xiang on 2021/7/19.
//

#ifndef SAD_NAV_STATE_H
#define SAD_NAV_STATE_H

#include <sophus/so3.hpp>             //Sophus库中的SO3
#include "common/eigen_types.h"       //Eigen库中的矩阵类型

namespace sad {

/**
 * 导航用的状态变量
 * @tparam T    标量类型
 *
 * 这是个封装好的类，部分程序使用本结构体，也有一部分程序使用散装的pvq，都是可以的
 */

//模版类结构体
template <typename T>
struct NavState {
    //定义三维变量 类型 的别名
    using Vec3 = Eigen::Matrix<T, 3, 1>;
    using SO3 = Sophus::SO3<T>;

    //默认构造函数，不会初始化类的成员变量
    NavState() = default;

    // from time, R, p, v, bg, ba
    //显式调用，调用的时候必须指明参数: sad::NavState<double> state1(123.456, R, t, v, bg, ba); 
    explicit NavState(double time, const SO3& R = SO3(), const Vec3& t = Vec3::Zero(), const Vec3& v = Vec3::Zero(),
                      const Vec3& bg = Vec3::Zero(), const Vec3& ba = Vec3::Zero())
        : timestamp_(time), R_(R), p_(t), v_(v), bg_(bg), ba_(ba) {}

    // from pose and vel
    //隐式调用，调用的时候不一定需要指明参数，但是会有风险: sad::NavStated state2 = {123.456, pose};
    NavState(double time, const SE3& pose, const Vec3& vel = Vec3::Zero())
        : timestamp_(time), R_(pose.so3()), p_(pose.translation()), v_(vel) {}

    /// 转换到Sophus
    Sophus::SE3<T> GetSE3() const { return SE3(R_, p_); }

    //友元函数，保证可以访问私有的变量，即使这里都是公有的，方便以后做扩展
    //返回值和函数名
        //std::ostream&     返回值为一个输出流类型的引用    
        //operator<<        重载了<<操作符
    //函数参数
        //std:ostream& os   输出流对象，例如 std::cout
        //const NavState    常量引用，要输出的NavState对象
    //返回值
        //返回输出流os，以便进行链式操作
    //使用
        //std::cout << state << std::endl;
    friend std::ostream& operator<<(std::ostream& os, const NavState<T>& s) {
        os << "p: " << s.p_.transpose() << ", v: " << s.v_.transpose()
           << ", q: " << s.R_.unit_quaternion().coeffs().transpose() << ", bg: " << s.bg_.transpose()
           << ", ba: " << s.ba_.transpose();
        return os;
    }

    //公有变量
    double timestamp_ = 0;    // 时间
    SO3 R_;                   // 旋转
    Vec3 p_ = Vec3::Zero();   // 平移
    Vec3 v_ = Vec3::Zero();   // 速度
    Vec3 bg_ = Vec3::Zero();  // gyro 零偏
    Vec3 ba_ = Vec3::Zero();  // acce 零偏
};

//定义在struct之外，在namespace之内，方便使用
//例如: ui.UpdateNavState(sad::NavStated(0, pose, v_world));
using NavStated = NavState<double>;
using NavStatef = NavState<float>;

}  // namespace sad

#endif
