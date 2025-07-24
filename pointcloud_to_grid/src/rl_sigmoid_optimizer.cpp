#include <ros/ros.h>
#include <pointcloud_to_grid/RLSigmoidParams.h>

class RLSigmoidOptimizer {
private:
    ros::NodeHandle nh_;
    ros::Subscriber rl_params_sub_;
    
    // RL에서 받은 최적 파라미터
    float optimal_M_, optimal_k_, optimal_c_;
    bool params_received_;

public:
    RLSigmoidOptimizer() : params_received_(false) {
        rl_params_sub_ = nh_.subscribe("/rl_sigmoid_params", 1, 
                                     &RLSigmoidOptimizer::rlParamsCallback, this);
    }
    
    void rlParamsCallback(const pointcloud_to_grid::RLSigmoidParams::ConstPtr& msg) {
        optimal_M_ = msg->M;
        optimal_k_ = msg->k;
        optimal_c_ = msg->c;
        params_received_ = true;
        
        ROS_INFO("Received RL optimized params: M=%.2f, k=%.2f, c=%.2f", 
                 optimal_M_, optimal_k_, optimal_c_);
    }
    
    std::vector<waypoint_maker::Waypoint> generateOptimizedPath() {
        if (!params_received_) {
            ROS_WARN("RL parameters not received yet, using default values");
            return {};
        }
        
        std::vector<waypoint_maker::Waypoint> waypoints_sig;
        waypoint_maker::Waypoint sig_point;
        
        // RL로 최적화된 파라미터로 경로 생성
        for (float x = 50; x > -60; x -= 0.5) {
            double point_y = optimal_M_ * (1.0 / (1.0 + exp(-optimal_k_ * (x - optimal_c_))));
            
            sig_point.pose.pose.position.x = x;
            sig_point.pose.pose.position.y = point_y;
            waypoints_sig.push_back(sig_point);
        }
        
        return waypoints_sig;
    }
};
