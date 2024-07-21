//
// Created by xiang on 2021/7/19.
//

#include "ch4/gins_pre_integ.h"
#include "ch4/g2o_types.h"
#include "common/g2o_types.h"

#include <glog/logging.h>

#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/robust_kernel.h>
#include <g2o/solvers/eigen/linear_solver_eigen.h>

namespace sad {

void GinsPreInteg::AddImu(const IMU& imu) {
    if (first_gnss_received_ && first_imu_received_) {
        //imu预积分，里面计算了预积分delta三项，噪声更新矩阵，偏置更新矩阵
        pre_integ_->Integrate(imu, imu.timestamp_ - last_imu_.timestamp_);
    }

    //更新数据
    first_imu_received_ = true;
    last_imu_ = imu;
    current_time_ = imu.timestamp_;
}

//这个函数用来定义所有相关的信息矩阵
void GinsPreInteg::SetOptions(sad::GinsPreInteg::Options options) {
    //偏置的 信息矩阵:每个信息矩阵3x3，内容为测得的 偏置方差的倒数
    double bg_rw2 = 1.0 / (options_.bias_gyro_var_ * options_.bias_gyro_var_);
    options_.bg_rw_info_.diagonal() << bg_rw2, bg_rw2, bg_rw2;
    double ba_rw2 = 1.0 / (options_.bias_acce_var_ * options_.bias_acce_var_);
    options_.ba_rw_info_.diagonal() << ba_rw2, ba_rw2, ba_rw2;

    //GNSS的 信息矩阵:每个信息矩阵为 6x6，内容为三个角度的，两个水平方向的，一个竖直方向的
    double gp2 = options_.gnss_pos_noise_ * options_.gnss_pos_noise_;
    double gh2 = options_.gnss_height_noise_ * options_.gnss_height_noise_;
    double ga2 = options_.gnss_ang_noise_ * options_.gnss_ang_noise_;
    options_.gnss_info_.diagonal() << 1.0 / ga2, 1.0 / ga2, 1.0 / ga2, 1.0 / gp2, 1.0 / gp2, 1.0 / gh2;
   
    //创建了一个指向 IMUPreintegration 对象的共享指针，并且options初始化了
            //这个options是 IMUPreintegration::Options,
            //里面定义了：两个初始偏置零偏 + 陀螺噪声的标准差
    //make_shared:共享指针是一种智能指针，自动管理对象的生命周期，当最后一个指向该对象的共享指针被销毁时，对象也会被销毁
            //直接创建了一个IMUPreintegration的对象，并且返回一个指向该对象的指针，可以避免单独调用new
            //在内存分配和对象构造的方面有一定的优势
    pre_integ_ = std::make_shared<IMUPreintegration>(options.preinteg_options_);

    //Odemetry的 信息矩阵：3x3
    double o2 = 1.0 / (options_.odom_var_ * options_.odom_var_);
    options_.odom_info_.diagonal() << o2, o2, o2;

    //当前时刻的先验信息矩阵 15x15,初始化的时候被设定为100，这里的v,bg,ba被设定为1000000
    prior_info_.block<6, 6>(9, 9) = Mat6d ::Identity() * 1e6;

    //当前时刻状态，初始化的时候被设定为 nullptr
    if (this_frame_) {
        this_frame_->bg_ = options_.preinteg_options_.init_bg_;
        this_frame_->ba_ = options_.preinteg_options_.init_ba_;
    }
}

void GinsPreInteg::AddGnss(const GNSS& gnss) {
    //NavStated对象创建的时候被隐式调用
    this_frame_ = std::make_shared<NavStated>(current_time_);
    //this_gnss_本来就是一个GNSS对象
    this_gnss_ = gnss;

    if (!first_gnss_received_) {
        if (!gnss.heading_valid_) {
            // 要求首个GNSS必须有航向
            return;
        }

        // 首个gnss信号，将初始pose设置为该gnss信号
        this_frame_->timestamp_ = gnss.unix_time_;
        this_frame_->p_ = gnss.utm_pose_.translation();
        this_frame_->R_ = gnss.utm_pose_.so3();
        this_frame_->v_.setZero();
        this_frame_->bg_ = options_.preinteg_options_.init_bg_;
        this_frame_->ba_ = options_.preinteg_options_.init_ba_;

        pre_integ_ = std::make_shared<IMUPreintegration>(options_.preinteg_options_);

        last_frame_ = this_frame_;
        last_gnss_ = this_gnss_;
        first_gnss_received_ = true;
        current_time_ = gnss.unix_time_;    //unix系统时间
        return;
    }

    // 积分到GNSS时刻。使用的积分为预积分，里面计算了预积分delta三项，噪声更新矩阵，偏置更新矩阵
    pre_integ_->Integrate(last_imu_, gnss.unix_time_ - current_time_);

    //更新时间
    current_time_ = gnss.unix_time_;
    //这里面的预测为：使用预积分得到的delta三项，简单加上初值，进行递推，并且将递推结果赋值给state,也就是这里的*this_frame
    *this_frame_ = pre_integ_->Predict(*last_frame_, options_.gravity_);

    Optimize();

    //更新帧
    last_frame_ = this_frame_;
    //更新GNSS信号
    last_gnss_ = this_gnss_;
}

void GinsPreInteg::AddOdom(const sad::Odom& odom) {
    last_odom_ = odom;
    last_odom_set_ = true;
}

void GinsPreInteg::Optimize() {
    if (pre_integ_->dt_ < 1e-3) {
        // 未得到积分
        return;
    }


    //第一步：定义了一个块求解器类型，这里用的是BlockSolverX，这是一个通用的块求解器
    using BlockSolverType = g2o::BlockSolverX;  
    
    //第二步：创建一个线性求解器
    using LinearSolverType = g2o::LinearSolverEigen<BlockSolverType::PoseMatrixType>;  

    //第三步：创建了一个Levenberg-Marquardt算法的优化求解器对象 solver
        //OptimizationAlgorithmLevenberg 是一种优化算法
        //make_unique 是一种安全的动态内存分配方式，分配给了 BlockSolverType 对象 和 LinearSolverType 对象，并且传递给 solver
                    //不需要显式的使用new关键字，如果new构造函数有异常则内存不会正确释放，导致内存泄漏，但是make_unique可以自动释放在异常的时候
    auto* solver = new g2o::OptimizationAlgorithmLevenberg(
        g2o::make_unique<BlockSolverType>(g2o::make_unique<LinearSolverType>()));

    //第四步：创建稀疏化求解器
    g2o::SparseOptimizer optimizer; //创建优化器
    optimizer.setAlgorithm(solver); //用前面定义好的求解器作为求解方法

    //第五步：定义图的顶点并添加到优化器中
    // 上时刻顶点， pose, v, bg, ba
    auto v0_pose = new VertexPose();
    v0_pose->setId(0);
    v0_pose->setEstimate(last_frame_->GetSE3());
    optimizer.addVertex(v0_pose);

    auto v0_vel = new VertexVelocity();
    v0_vel->setId(1);
    v0_vel->setEstimate(last_frame_->v_);
    optimizer.addVertex(v0_vel);

    auto v0_bg = new VertexGyroBias();
    v0_bg->setId(2);
    v0_bg->setEstimate(last_frame_->bg_);
    optimizer.addVertex(v0_bg);

    auto v0_ba = new VertexAccBias();
    v0_ba->setId(3);
    v0_ba->setEstimate(last_frame_->ba_);
    optimizer.addVertex(v0_ba);

    // 本时刻顶点，pose, v, bg, ba
    auto v1_pose = new VertexPose();
    v1_pose->setId(4);
    v1_pose->setEstimate(this_frame_->GetSE3());
    optimizer.addVertex(v1_pose);

    auto v1_vel = new VertexVelocity();
    v1_vel->setId(5);
    v1_vel->setEstimate(this_frame_->v_);
    optimizer.addVertex(v1_vel);

    auto v1_bg = new VertexGyroBias();
    v1_bg->setId(6);
    v1_bg->setEstimate(this_frame_->bg_);
    optimizer.addVertex(v1_bg);

    auto v1_ba = new VertexAccBias();
    v1_ba->setId(7);
    v1_ba->setEstimate(this_frame_->ba_);
    optimizer.addVertex(v1_ba);

    //第六步：定义图的边并添加到优化器中
    // 预积分边
    auto edge_inertial = new EdgeInertial(pre_integ_, options_.gravity_);
    edge_inertial->setVertex(0, v0_pose);
    edge_inertial->setVertex(1, v0_vel);
    edge_inertial->setVertex(2, v0_bg);
    edge_inertial->setVertex(3, v0_ba);
    edge_inertial->setVertex(4, v1_pose);
    edge_inertial->setVertex(5, v1_vel);
    //Huber核函数是一种常用的鲁棒核函数，可以在处理异常的时候平滑的过度，从而减少异常值对优化结果的影响
    auto* rk = new g2o::RobustKernelHuber();//创建一个鲁棒核函数对象 rk
    rk->setDelta(200.0);                    //这个参数决定了残差的阈值。残差小于delta则采用二次损失函数，否则采用线性损失函数
                                                                  //平滑过度有效抑制大残差对优化的影响
    edge_inertial->setRobustKernel(rk);     //设置rk为鲁棒核函数
    optimizer.addEdge(edge_inertial);

    // 两个零偏随机游走
    auto* edge_gyro_rw = new EdgeGyroRW();
    edge_gyro_rw->setVertex(0, v0_bg);
    edge_gyro_rw->setVertex(1, v1_bg);
    edge_gyro_rw->setInformation(options_.bg_rw_info_);
    optimizer.addEdge(edge_gyro_rw);

    auto* edge_acc_rw = new EdgeAccRW();
    edge_acc_rw->setVertex(0, v0_ba);
    edge_acc_rw->setVertex(1, v1_ba);
    edge_acc_rw->setInformation(options_.ba_rw_info_);
    optimizer.addEdge(edge_acc_rw);

    // 上时刻先验
    auto* edge_prior = new EdgePriorPoseNavState(*last_frame_, prior_info_);
    edge_prior->setVertex(0, v0_pose);
    edge_prior->setVertex(1, v0_vel);
    edge_prior->setVertex(2, v0_bg);
    edge_prior->setVertex(3, v0_ba);
    optimizer.addEdge(edge_prior);

    // GNSS边
    auto edge_gnss0 = new EdgeGNSS(v0_pose, last_gnss_.utm_pose_);
    edge_gnss0->setInformation(options_.gnss_info_);
    optimizer.addEdge(edge_gnss0);

    auto edge_gnss1 = new EdgeGNSS(v1_pose, this_gnss_.utm_pose_);
    edge_gnss1->setInformation(options_.gnss_info_);
    optimizer.addEdge(edge_gnss1);

    // Odom边
    EdgeEncoder3D* edge_odom = nullptr;
    Vec3d vel_world = Vec3d::Zero();
    Vec3d vel_odom = Vec3d::Zero();
    if (last_odom_set_) {
        // velocity obs
        double velo_l =
            options_.wheel_radius_ * last_odom_.left_pulse_ / options_.circle_pulse_ * 2 * M_PI / options_.odom_span_;
        double velo_r =
            options_.wheel_radius_ * last_odom_.right_pulse_ / options_.circle_pulse_ * 2 * M_PI / options_.odom_span_;
        double average_vel = 0.5 * (velo_l + velo_r);
        vel_odom = Vec3d(average_vel, 0.0, 0.0);
        vel_world = this_frame_->R_ * vel_odom;

        edge_odom = new EdgeEncoder3D(v1_vel, vel_world);
        edge_odom->setInformation(options_.odom_info_);
        optimizer.addEdge(edge_odom);

        // 重置odom数据到达标志位，等待最新的odom数据
        last_odom_set_ = false;
    }

    //在优化过程中，输出调试信息
    optimizer.setVerbose(options_.verbose_);

    //第七步：设置优化参数并且开始执行优化
    optimizer.initializeOptimization(); //初始化优化问题
    optimizer.optimize(20); //设置迭代次数

    if (options_.verbose_) {
        // 获取结果，统计各类误差；chi2表示误差项的二次型误差，也就是残差的平方和，用来衡量优化中某个边或误差项的拟合程度的一个指标。
        //                     chi2越小，表示该边的误差项与实际越接近
        LOG(INFO) << "chi2/error: ";
        LOG(INFO) << "preintegration: " << edge_inertial->chi2() << "/" << edge_inertial->error().transpose();
        // LOG(INFO) << "gnss0: " << edge_gnss0->chi2() << ", " << edge_gnss0->error().transpose();
        LOG(INFO) << "gnss1: " << edge_gnss1->chi2() << ", " << edge_gnss1->error().transpose();
        LOG(INFO) << "bias: " << edge_gyro_rw->chi2() << "/" << edge_acc_rw->error().transpose();
        LOG(INFO) << "prior: " << edge_prior->chi2() << "/" << edge_prior->error().transpose();
        if (edge_odom) {
            LOG(INFO) << "body vel: " << (v1_pose->estimate().so3().inverse() * v1_vel->estimate()).transpose();
            LOG(INFO) << "meas: " << vel_odom.transpose();
            LOG(INFO) << "odom: " << edge_odom->chi2() << "/" << edge_odom->error().transpose();
        }
    }

    last_frame_->R_ = v0_pose->estimate().so3();
    last_frame_->p_ = v0_pose->estimate().translation();
    last_frame_->v_ = v0_vel->estimate();
    last_frame_->bg_ = v0_bg->estimate();
    last_frame_->ba_ = v0_ba->estimate();

    this_frame_->R_ = v1_pose->estimate().so3();
    this_frame_->p_ = v1_pose->estimate().translation();
    this_frame_->v_ = v1_vel->estimate();
    this_frame_->bg_ = v1_bg->estimate();
    this_frame_->ba_ = v1_ba->estimate();

    // 重置integ
    options_.preinteg_options_.init_bg_ = this_frame_->bg_;
    options_.preinteg_options_.init_ba_ = this_frame_->ba_;
    pre_integ_ = std::make_shared<IMUPreintegration>(options_.preinteg_options_);
}

NavStated GinsPreInteg::GetState() const {
    if (this_frame_ == nullptr) {
        return {};
    }

    if (pre_integ_ == nullptr) {
        return *this_frame_;
    }

    return pre_integ_->Predict(*this_frame_, options_.gravity_);
}

}  // namespace sad