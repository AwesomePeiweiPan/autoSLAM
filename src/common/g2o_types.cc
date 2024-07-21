//
// Created by xiang on 23-1-11.
//

#include "common/g2o_types.h"

namespace sad {

EdgePriorPoseNavState::EdgePriorPoseNavState(const NavStated& state, const Mat15d& info) {
    resize(4);                  //用于设置 边 所连接的顶点的数量，比如这里连接了4个顶点
    state_ = state;             //将传入的state对象赋值给edge累的成员变量 state_
    setInformation(info);       //用于设置 edge 的协方差信息矩阵，是协方差矩阵的逆，反映了观测的置信度
}

void EdgePriorPoseNavState::computeError() {
    auto* vp = dynamic_cast<const VertexPose*>(_vertices[0]);
    auto* vv = dynamic_cast<const VertexVelocity*>(_vertices[1]);
    auto* vg = dynamic_cast<const VertexGyroBias*>(_vertices[2]);
    auto* va = dynamic_cast<const VertexAccBias*>(_vertices[3]);

    //并没有在书里面进行直接的定义，可以自己推导出来，误差直接被定义为这个时刻的状态 减去 上一个时刻的状态
    const Vec3d er = SO3(state_.R_.matrix().transpose() * vp->estimate().so3().matrix()).log();
    const Vec3d ep = vp->estimate().translation() - state_.p_;
    const Vec3d ev = vv->estimate() - state_.v_;
    const Vec3d ebg = vg->estimate() - state_.bg_;
    const Vec3d eba = va->estimate() - state_.ba_;

    _error << er, ep, ev, ebg, eba;
}

void EdgePriorPoseNavState::linearizeOplus() {
    const auto* vp = dynamic_cast<const VertexPose*>(_vertices[0]);
    const Vec3d er = SO3(state_.R_.matrix().transpose() * vp->estimate().so3().matrix()).log();

    /// 注意有3个index, 顶点的，自己误差的，顶点内部变量的
    // 对顶点0 (位姿) 的雅可比矩阵
    _jacobianOplus[0].setZero();  //初始化
    _jacobianOplus[0].block<3, 3>(0, 0) = SO3::jr_inv(er);    // dr/dr
    _jacobianOplus[0].block<3, 3>(3, 3) = Mat3d::Identity();  // dp/dp
    // 对顶点1 (速度) 的雅可比矩阵
    _jacobianOplus[1].setZero();   //初始化
    _jacobianOplus[1].block<3, 3>(6, 0) = Mat3d::Identity();  // dv/dv
    // 对顶点2 (陀螺仪偏置) 的雅可比矩阵
    _jacobianOplus[2].setZero();   //初始化
    _jacobianOplus[2].block<3, 3>(9, 0) = Mat3d::Identity();  // dbg/dbg
    // 对顶点3 (加速度计偏置) 的雅可比矩阵
    _jacobianOplus[3].setZero();   //初始化
    _jacobianOplus[3].block<3, 3>(12, 0) = Mat3d::Identity();  // dba/dba
}

}  // namespace sad