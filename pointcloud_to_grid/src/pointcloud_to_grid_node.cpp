#include <ros/ros.h>
#include <nav_msgs/OccupancyGrid.h>
#include <sensor_msgs/PointCloud.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl/PCLPointCloud2.h>
#include <pcl/conversions.h>
#include <pcl_ros/point_cloud.h>
#include <pointcloud_to_grid/pointcloud_to_grid_core.hpp>
#include <pointcloud_to_grid/MyParamsConfig.h>
#include <dynamic_reconfigure/server.h>

nav_msgs::OccupancyGridPtr intensity_grid(new nav_msgs::OccupancyGrid); // Intensity 정보 저장 포인터
nav_msgs::OccupancyGridPtr height_grid(new nav_msgs::OccupancyGrid);    // 높이 정보 저장 포인터
// nav_msgs::OccupancyGridPtr path_grid(new nav_msgs::OccupancyGrid); // 경로 정보 저장


GridMap grid_map; // 객체 선언(hpp에 있음)
ros::Publisher pub_igrid, pub_hgrid; // pub intensity, pub height
ros::Publisher pub_path_; // pub intensity, pub height
ros::Publisher pub_path_grid;
ros::Publisher pub_best_path_;
ros::Publisher pub_global_path;
ros::Publisher pub_global_path_v;
ros::Publisher pub_flag;
ros::Publisher pub_G2L_path;
ros::Publisher pub_G2L_path2;

ros::Subscriber sub_pc2;             // sub pointcloud2
ros::Subscriber obs_sub;     
ros::Subscriber odom_sub_;
ros::Subscriber course_sub_;
ros::Subscriber enddist_sub_;
ros::Subscriber state_sub_;
ros::Subscriber lane_sub_;


// path planning
int bestPathIndex;
bool avoidance = false;
bool one_time = true;
waypoint_maker::Lane final_path;
std::vector<double> waypoints_x, waypoints_y;
std::vector<std::vector<waypoint_maker::Waypoint>> waypoints_group;
std::vector<waypoint_maker::Waypoint> waypoints_sig;
std::vector<float> waypoints_k_group;
std::vector<int> waypoints_idx_group;


//callback
geometry_msgs::PoseStamped cur_pose_;
double cur_course_ = 0.0;
bool obs = false;
float end_dist = 10000.0;
int state_ = 0;
int waypoints_size_;
std::vector<waypoint_maker::Waypoint> waypoints_;

// Decision
  bool sig_obs = false;
  bool reference_obs = false;



std::string frame_id = "os_sensor";

PointXY getIndex(double x, double y) { // (x, y)를 OccupancyGrid의 셀 인덱스로 변환
    PointXY ret;
    ret.x = int(fabs(x - grid_map.topleft_x) / grid_map.cell_size);
    ret.y = int(fabs(y - grid_map.topleft_y) / grid_map.cell_size);
    return ret;
}




void paramsCallback(my_dyn_rec::MyParamsConfig &config, uint32_t level) {
    grid_map.cell_size = config.cell_size;
    grid_map.position_x = config.position_x;
    grid_map.position_y = config.position_y;
    grid_map.cell_size = config.cell_size;
    grid_map.length_x = config.length_x;
    grid_map.length_y = config.length_y;
    grid_map.cloud_in_topic = config.cloud_in_topic;
    grid_map.intensity_factor = config.intensity_factor;
    grid_map.height_factor = config.height_factor;
    grid_map.mapi_topic_name = config.mapi_topic_name;
    grid_map.maph_topic_name = config.maph_topic_name;

    grid_map.initGrid(intensity_grid); // intensity_grid 초기화
    grid_map.initGrid(height_grid);   // height_grid 초기화
    grid_map.paramRefresh();          // 변수 업데이트
}

void OdomCallback(const nav_msgs::Odometry::ConstPtr &odom_msg) {
	cur_pose_.header = odom_msg->header;
	cur_pose_.pose.position = odom_msg->pose.pose.position;
}

void CourseCallback(const std_msgs::Float64::ConstPtr &course_msg){
	cur_course_ = course_msg -> data;
}

void DistCallback(const std_msgs::Float64::ConstPtr &dist_msg){
  end_dist = dist_msg->data;
}

void ObstacleCallback(const std_msgs::Bool::ConstPtr &obs_msg){
  obs = obs_msg->data;
}


void StateCallback(const waypoint_maker::State::ConstPtr &state_msg)
{
 state_ = state_msg->current_state;
}

void LaneCallback(const waypoint_maker::Lane::ConstPtr &lane_msg)
{
	waypoints_.clear();
	std::vector<waypoint_maker::Waypoint>().swap(waypoints_);
	waypoints_ = lane_msg->waypoints;
	waypoints_size_ = waypoints_.size();
}

void PathCheck(const std::vector<std::vector<waypoint_maker::Waypoint>> &waypoints_group, nav_msgs::OccupancyGridPtr &grid, std::vector<signed char> hpoints) {
    waypoints_idx_group.clear();
    std::vector<int> path;
    path.clear();
    

    int w_idx = 0;
    if(waypoints_group.size() != 0 && hpoints.size() != 0){
      for (const auto &waypoints : waypoints_group) {
          bool k = false;
          for (const auto &waypoint : waypoints) {
              double x = waypoint.pose.pose.position.x;
              double y = waypoint.pose.pose.position.y;
              PointXY cell = getIndex(x, y);

              if (cell.x < grid_map.cell_num_x && cell.y < grid_map.cell_num_y) {
                  int idx = cell.y * grid_map.cell_num_x + cell.x;

                  path.push_back(idx);

                  if(hpoints[idx] == -128){
                    k = false;
                  }
                  else{
                    k = true;
                    break;
                  }
              }
          }
        if(!k){
          waypoints_idx_group.push_back(w_idx);
        }

        w_idx ++;  
      }
    }
}

void global_PathCheck(const std::vector<waypoint_maker::Waypoint> &G2L_path, nav_msgs::OccupancyGridPtr &grid, std::vector<signed char> hpoints, bool &obstacleDetected) {
    std::vector<int> path;
    path.clear();
    
    if(G2L_path.size() != 0 && hpoints.size() != 0){
      bool k = false;
      for (const auto &waypoints : G2L_path) {
        double x = waypoints.pose.pose.position.x;
        double y = waypoints.pose.pose.position.y;
        PointXY cell = getIndex(x, y);

        if (cell.x < grid_map.cell_num_x && cell.y < grid_map.cell_num_y) {
            int idx = cell.y * grid_map.cell_num_x + cell.x;

            path.push_back(idx);

            if(hpoints[idx] == -128){
              k = false;
            }
            else{
              k = true;
              break;
            }
          }
      
        }
        if(!k){
          obstacleDetected = false;
        }
        else{
          obstacleDetected = true;
        }
    }
}

std::vector<std::pair<int, int>> generateDirections(int n) {
    std::vector<std::pair<int, int>> directions;
    for (int dx = -n; dx <= n; ++dx) {
        for (int dy = -n; dy <= n; ++dy) {
            directions.emplace_back(dx, dy);
        }
    }
    return directions;
}

void Local_to_Global(std::vector<std::vector<waypoint_maker::Waypoint>> waypoints_group, int bestPathIndex){
  std::vector<waypoint_maker::Waypoint> global_path;
  global_path.clear();
	double yaw = cur_course_ * M_PI / 180;
	if(waypoints_group.size() != 0 && bestPathIndex >= 0){
		for(const auto &waypoint : waypoints_group[bestPathIndex]){
      waypoint_maker::Waypoint point;
		
			point.pose.pose.position.x = cur_pose_.pose.position.x +  ((waypoint.pose.pose.position.x+1.04) * cos(yaw)) - ((waypoint.pose.pose.position.y) * sin(yaw));
			point.pose.pose.position.y = cur_pose_.pose.position.y +  ((waypoint.pose.pose.position.x+1.04) * sin(yaw)) + ((waypoint.pose.pose.position.y) * cos(yaw));
		
			global_path.push_back(point);
		}
	}
  final_path.waypoints = global_path;
}


void Global_to_Local(std::vector<waypoint_maker::Waypoint> global_waypoints, std::vector<waypoint_maker::Waypoint> &Local_path){
  Local_path.clear();
	double yaw = cur_course_ * M_PI / 180;
	if(global_waypoints.size() != 0){
		for(const auto &waypoint : global_waypoints){
      waypoint_maker::Waypoint point;
      double dx = waypoint.pose.pose.position.x - cur_pose_.pose.position.x;
      double dy = waypoint.pose.pose.position.y - cur_pose_.pose.position.y;
      

      point.pose.pose.position.x = ((dx) * cos(yaw) + dy * sin(yaw))-1.04;
      point.pose.pose.position.y = -(dx) * sin(yaw) + dy * cos(yaw);
		
			Local_path.push_back(point);

		}
	}
}

void visualize_G2L_path(std::vector<waypoint_maker::Waypoint> path){
    nav_msgs::Path path_msg_;
    path_msg_.header.stamp = ros::Time::now();
    path_msg_.header.frame_id = "os_sensor";
    path_msg_.poses.clear();

    // if(!obs){
    //   path_msg_.poses.clear();
    // }

    // else{
      for(int j=0; j<path.size(); j++){
        geometry_msgs::PoseStamped pose;
        pose.pose.position.x = path[j].pose.pose.position.x;
        pose.pose.position.y = path[j].pose.pose.position.y;
        pose.pose.position.z = path[j].pose.pose.position.z;
        pose.pose.orientation = path[j].pose.pose.orientation;
        path_msg_.poses.push_back(pose);
      }
      for(int k=path.size()-1; k>=0; k--){
        geometry_msgs::PoseStamped pose;
        pose.pose.position.x = path[k].pose.pose.position.x;
        pose.pose.position.y = path[k].pose.pose.position.y;
        pose.pose.position.z = path[k].pose.pose.position.z;
        pose.pose.orientation = path[k].pose.pose.orientation;
        path_msg_.poses.push_back(pose);
      }
    // }  

    // 통합된 Path 메시지 퍼블리시
    pub_G2L_path.publish(path_msg_);
}

void visualize_G2L_path2(std::vector<waypoint_maker::Waypoint> path){
    nav_msgs::Path path_msg_;
    path_msg_.header.stamp = ros::Time::now();
    path_msg_.header.frame_id = "os_sensor";
    path_msg_.poses.clear();

    if(!avoidance){
      path_msg_.poses.clear();
    }

    // else{
      for(int j=0; j<path.size(); j++){
        geometry_msgs::PoseStamped pose;
        pose.pose.position.x = path[j].pose.pose.position.x;
        pose.pose.position.y = path[j].pose.pose.position.y;
        pose.pose.position.z = path[j].pose.pose.position.z;
        pose.pose.orientation = path[j].pose.pose.orientation;
        path_msg_.poses.push_back(pose);
      }
      for(int k=path.size()-1; k>=0; k--){
        geometry_msgs::PoseStamped pose;
        pose.pose.position.x = path[k].pose.pose.position.x;
        pose.pose.position.y = path[k].pose.pose.position.y;
        pose.pose.position.z = path[k].pose.pose.position.z;
        pose.pose.orientation = path[k].pose.pose.orientation;
        path_msg_.poses.push_back(pose);
      }
    // }  

    // 통합된 Path 메시지 퍼블리시
    pub_G2L_path2.publish(path_msg_);
}

void Decision(std::vector<waypoint_maker::Waypoint> path, std::vector<signed char> hpoints, bool &obs){  // obs : output
  std::vector<waypoint_maker::Waypoint> Local_path;
  Local_path.clear();
  if(path.size() != 0){
    Global_to_Local(path, Local_path);
    visualize_G2L_path(Local_path);
    global_PathCheck(Local_path, height_grid, hpoints, obs);
  }
  else{
    ROS_WARN("path is empty");
  }
}

void pointcloudCallback(const sensor_msgs::PointCloud2::ConstPtr &msg) {
    pcl::PointCloud<pcl::PointXYZI> out_cloud;
    pcl::fromROSMsg(*msg, out_cloud);
    // Initialize grid
    grid_map.initGrid(intensity_grid);
    grid_map.initGrid(height_grid);
    std::vector<signed char> hpoints(grid_map.cell_num_x * grid_map.cell_num_y, -128);
    std::vector<signed char> ipoints(grid_map.cell_num_x * grid_map.cell_num_y, -128);
    for (auto &out_point : out_cloud.points) {
        if (out_point.x > 0.01 || out_point.x < -0.01) {
            if (out_point.x > grid_map.bottomright_x && out_point.x < grid_map.topleft_x &&
                out_point.z > -0.1 && out_point.z < 4.0 && out_point.y > grid_map.bottomright_y && out_point.y < grid_map.topleft_y) {
                PointXY cell = getIndex(out_point.x, out_point.y);
                if (cell.x < grid_map.cell_num_x && cell.y < grid_map.cell_num_y) {
                    int idx = cell.y * grid_map.cell_num_x + cell.x;
                    ipoints[idx] = out_point.intensity * grid_map.intensity_factor;
                    hpoints[idx] = out_point.z * grid_map.height_factor;
                }
            }
        }
    }
    int n = 8; // 얼만큼 불릴건가
    std::vector<std::pair<int, int>> directions = generateDirections(n);
    std::vector<signed char> updated_hpoints = hpoints;
    for (int y = 0; y < grid_map.cell_num_y; ++y) {
        for (int x = 0; x < grid_map.cell_num_x; ++x) {
            int idx = y * grid_map.cell_num_x + x;
            if (hpoints[idx] != -128) { // 장애물이 있는 셀
                for (const auto &dir : directions) {
                    int nx = x + dir.first;
                    int ny = y + dir.second;
                    if (nx >= 0 && nx < grid_map.cell_num_x && ny >= 0 && ny < grid_map.cell_num_y) {
                        int neighbor_idx = ny * grid_map.cell_num_x + nx;
                        // 주변 셀만 0으로 설정 (이미 값이 설정된 셀은 변경하지 않음)
                        if (updated_hpoints[neighbor_idx] == -128) {
                            updated_hpoints[neighbor_idx] = 70;
                        }
                    }
                }
            }
        }
    }
    hpoints = updated_hpoints;
    // hpoints 결과 디버깅 출력clear
    // ROS_INFO_STREAM("Modified hpoints:");
    for (int y = 0; y < grid_map.cell_num_y; ++y) {
        std::ostringstream row;
        for (int x = 0; x < grid_map.cell_num_x; ++x) {
            int idx = y * grid_map.cell_num_x + x;
            row << static_cast<int>(hpoints[idx]) << " ";
        }
        // ROS_INFO_STREAM(row.str());
    }
    intensity_grid->header.stamp = ros::Time::now();
    intensity_grid->header.frame_id = "os_sensor";
    intensity_grid->info.map_load_time = ros::Time::now();
    intensity_grid->data = ipoints;
    height_grid->header.stamp = ros::Time::now();
    height_grid->header.frame_id = "os_sensor";
    height_grid->info.map_load_time = ros::Time::now();
    height_grid->data = hpoints;
    
    waypoints_idx_group.clear();
    if(hpoints.size() != 0){
      if(waypoints_size_ != 0){
        Decision(waypoints_, hpoints, reference_obs);
      }
      if(reference_obs){
        std::vector<waypoint_maker::Waypoint> sig_Local_path;
        sig_Local_path.clear();
        PathCheck(waypoints_group, height_grid, hpoints);
        Global_to_Local(final_path.waypoints, sig_Local_path);
        global_PathCheck(sig_Local_path, height_grid, hpoints, sig_obs);
        visualize_G2L_path2(sig_Local_path);
      }
    }    

    pub_igrid.publish(intensity_grid);
    pub_hgrid.publish(height_grid);
}

void sigmoid(){
  double point_y;
  float x_0;  // x_0 : y=0일 때 x 값
  float slope;
  float c_start = -10.0, c_end = 10.0, c_step = 0.1; // c 값 탐색 범위
  float straight = 10.0;
  waypoint_maker::Waypoint sig_point;

  waypoints_group.clear();
  waypoints_sig.clear();
  waypoints_k_group.clear();
  for(float st = straight; st > 2.0; st--){
    for(float M = -6.0; M <= 6.0; M += 0.5) {
      for(float k = 0.2; k < 3.0; k += slope) {
        waypoints_sig.clear();
        if(k < 0.3){
          slope = 0.05;
        }
        else{
          slope = 0.1;
        }

        waypoints_x.clear();
        waypoints_y.clear();

        float best_c = c_start;
        float min_error = 1e6;  // 1e6 = 10^6 

        // c 값을 탐색하여 최적의 시작점 찾기(c 값에 따라 0.1 안으로 들어오는 값이 없을 수 있기 때문에 있는 c 값 찾기 위해)
        for (float c = c_start; c <= c_end; c += c_step) {
          float temp_x_0 = 0.0;

          for (float x = 50; x > -60; x -= 0.5) {
            point_y = M * (1 / (1 + exp(-k * (x - c))));
            if (fabs(point_y) < 0.2) {
              temp_x_0 = x;
              break;
            }
          }

          if (fabs(temp_x_0) < min_error) {   // 너무 큰 수 방지
            min_error = fabs(temp_x_0);
            best_c = c;
          }
        }

        // 최적의 c 값으로 경로 생성
        float b_c = best_c;
        for (float x = 50; x > -60; x -= 0.5) {
          point_y = M * (1 / (1 + exp(-k * (x - b_c))));
          waypoints_x.push_back(x);
          waypoints_y.push_back(point_y);

          if (fabs(point_y) < 0.2) {
            x_0 = x;
            break;
          }
        }

        for (int i = 0; i < waypoints_x.size(); i++) {
          waypoints_x[i] -= x_0-st;

          if (waypoints_x[i] >= 0.0 && waypoints_x[i] < 35.0) {
            sig_point.pose.pose.position.x = waypoints_x[i];
            sig_point.pose.pose.position.y = waypoints_y[i];
            waypoints_sig.push_back(sig_point);
          }
        }
        if(waypoints_sig.size() != 0){
          for(float i=0; i <= st; i+=0.5){
            sig_point.pose.pose.position.x = waypoints_x.back()-i;
            if(waypoints_y.size()!=0){
              if(waypoints_y.back() > 0.0){
                sig_point.pose.pose.position.y = 0.2;
              }
              else{
                sig_point.pose.pose.position.y = -0.2;
              }
            }
            waypoints_sig.push_back(sig_point);
          }
          std::reverse(waypoints_sig.begin(), waypoints_sig.end());
          if(waypoints_sig[0].pose.pose.position.y <= 0.2 && waypoints_sig.size() >= 25){
              waypoints_group.push_back(waypoints_sig);
              waypoints_k_group.push_back(k);
          }
        }
      }
    }
  }
  std::cout << "---sig---" << std::endl;
  one_time = false;
}


void visualize_path_best(std::vector<std::vector<waypoint_maker::Waypoint>> waypoints_group, std::string id, int best_idx){
    nav_msgs::Path best_msg_;
    best_msg_.header.stamp = ros::Time::now();
    best_msg_.header.frame_id = frame_id;
    best_msg_.poses.clear();

    if(!avoidance){
      best_msg_.poses.clear();
    }
    else{
      const auto& waypoint = waypoints_group[best_idx];
      for(int j=0; j<waypoint.size(); j++){
        geometry_msgs::PoseStamped pose;
        pose.pose.position.x = waypoint[j].pose.pose.position.x;
        pose.pose.position.y = waypoint[j].pose.pose.position.y;
        pose.pose.position.z = waypoint[j].pose.pose.position.z;
        pose.pose.orientation = waypoint[j].pose.pose.orientation;
        best_msg_.poses.push_back(pose);
      }
      for(int k=waypoint.size()-1; k>=0; k--){
        geometry_msgs::PoseStamped pose;
        pose.pose.position.x = waypoint[k].pose.pose.position.x;
        pose.pose.position.y = waypoint[k].pose.pose.position.y;
        pose.pose.position.z = waypoint[k].pose.pose.position.z;
        pose.pose.orientation = waypoint[k].pose.pose.orientation;
        best_msg_.poses.push_back(pose);
      }
    }

    // 통합된 Path 메시지 퍼블리시
    pub_best_path_.publish(best_msg_);
}
// 최적의 경로 선택 함수
// void findBestPath(std::vector<std::vector<waypoint_maker::Waypoint>> path_group, std::vector<int> idx, float alpha , float beta) {

//   float slopeSum = 0.0;
//   float minCost = 1e6; // 초기 최소 비용
//   float totalCost = 1e6;
//   if(path_group.size() == 0 || idx.size() == 0){
//     // bestPathIndex = -1;
//   }
//   else{
//     for(auto i: idx){
//       slopeSum = 0.0;
//       std::vector<waypoint_maker::Waypoint> path = path_group[i];
//       int N = path.size();
//       if (N < 2){
//         totalCost = 1e9; // 데이터가 부족하면 큰 값 반환
//       }
//       else{
//         // SlopeCost 계산
//         for (int i = 0; i < N - 1; i++) {
//           float dx = (float)path[i + 1].pose.pose.position.x - (float)path[i].pose.pose.position.x;
//           float dy = (float)path[i + 1].pose.pose.position.y - (float)path[i].pose.pose.position.y;
          
//           if (fabs(dx) > 1e-6) { // dx가 0이 아닌 경우
//             slopeSum += fabs(dy / dx);
//           }
//         }
//       }

//       float slopecost = slopeSum / (N - 1); // 평균 기울기 변화량
//       float max_y = fabs(path[N-1].pose.pose.position.y); // last_y : max_y;
//       totalCost = alpha * slopecost + beta * max_y;

//       minCost = totalCost;
//       bestPathIndex = i;
      
//     }
//     // std::cout << "bestPathIndex : " << bestPathIndex << std::endl;
//     if(bestPathIndex != -1){
//       for(int i=0; i<path_group[bestPathIndex].size(); i++){
//         waypoint_maker::Waypoint tmp_path;
//         tmp_path.pose.pose.position.x = path_group[bestPathIndex][i].pose.pose.position.x;
//         tmp_path.pose.pose.position.y = path_group[bestPathIndex][i].pose.pose.position.y;
//       }
//       // best_path = path_group[bestPathIndex];
      
//     }
//   }

// }

void findBestPath(std::vector<std::vector<waypoint_maker::Waypoint>> path_group, std::vector<int> idx, std::vector<float> k_group) {
    if (path_group.size() == 0 || idx.size() == 0 || k_group.size() == 0) {
        ROS_WARN("Invalid path group or k_group size. Best path cannot be determined.");
        bestPathIndex = -1;
        return;
    }

    int best_k_idx = 0;
    float minCost = std::numeric_limits<float>::max();  // 초기 최소 비용을 무한대로 설정

    for (const auto i: idx) {
        float k = k_group[i];
        if (k < minCost) {
            minCost = k;
            bestPathIndex = i;  // 최적의 k 인덱스 갱신
        }
    }

    std::cout << "bestPathIndex : " << bestPathIndex << std::endl;
}

void visualize_path(std::vector<std::vector<waypoint_maker::Waypoint>> waypoints_group, std::string id, std::vector<int> idx){
    nav_msgs::Path path_msg_;
    path_msg_.header.stamp = ros::Time::now();
    path_msg_.header.frame_id = frame_id;
    path_msg_.poses.clear();

    // if(!obs){
    //   path_msg_.poses.clear();
    // }

    // else{
      for (const auto& index : idx) {
          const auto& waypoint = waypoints_group[index];
          for(int j=0; j<waypoint.size(); j++){
            geometry_msgs::PoseStamped pose;
            pose.pose.position.x = waypoint[j].pose.pose.position.x;
            pose.pose.position.y = waypoint[j].pose.pose.position.y;
            pose.pose.position.z = waypoint[j].pose.pose.position.z;
            pose.pose.orientation = waypoint[j].pose.pose.orientation;
            path_msg_.poses.push_back(pose);
          }
          for(int k=waypoint.size()-1; k>=0; k--){
            geometry_msgs::PoseStamped pose;
            pose.pose.position.x = waypoint[k].pose.pose.position.x;
            pose.pose.position.y = waypoint[k].pose.pose.position.y;
            pose.pose.position.z = waypoint[k].pose.pose.position.z;
            pose.pose.orientation = waypoint[k].pose.pose.orientation;
            path_msg_.poses.push_back(pose);
          }
      }  
    // }

    // 통합된 Path 메시지 퍼블리시
    pub_path_.publish(path_msg_);
}

void visualize_global_path(waypoint_maker::Lane path){
    nav_msgs::Path path_msg_;
    path_msg_.header.stamp = ros::Time::now();
    path_msg_.header.frame_id = "map";
    path_msg_.poses.clear();

    // if(!obs){
    //   path_msg_.poses.clear();
    // }

    // else{
      for(int j=0; j<path.waypoints.size(); j++){
        geometry_msgs::PoseStamped pose;
        pose.pose.position.x = path.waypoints[j].pose.pose.position.x;
        pose.pose.position.y = path.waypoints[j].pose.pose.position.y;
        pose.pose.position.z = path.waypoints[j].pose.pose.position.z;
        pose.pose.orientation = path.waypoints[j].pose.pose.orientation;
        path_msg_.poses.push_back(pose);
      }
      for(int k=path.waypoints.size()-1; k>=0; k--){
        geometry_msgs::PoseStamped pose;
        pose.pose.position.x = path.waypoints[k].pose.pose.position.x;
        pose.pose.position.y = path.waypoints[k].pose.pose.position.y;
        pose.pose.position.z = path.waypoints[k].pose.pose.position.z;
        pose.pose.orientation = path.waypoints[k].pose.pose.orientation;
        path_msg_.poses.push_back(pose);
      }
    // }  

    // 통합된 Path 메시지 퍼블리시
    pub_global_path_v.publish(path_msg_);
}

void Pub_flag(bool flag){
  std_msgs::Bool flag_msg;
  flag_msg.data = flag;
  pub_flag.publish(flag_msg);
}

void run(){
  static bool first_time = true;
  if(reference_obs){
    if(end_dist > 10.0){
    avoidance = true;
      if(first_time){
        first_time = false;
        std::cout << "first_time" << std::endl;
        findBestPath(waypoints_group, waypoints_idx_group, waypoints_k_group);
        if(bestPathIndex != -1){
          Local_to_Global(waypoints_group, bestPathIndex);  
        }
      }
      if(sig_obs){
        std::cout << "sig_obs" << sig_obs << std::endl;
        first_time = true;
        end_dist = 10000.0;
      }
    }
    else{
      avoidance = false;
      first_time = true;
      end_dist = 10000.0;
    }
  }
  else{
    avoidance = false;
    first_time = true;
    end_dist = 10000.0;
  }

  Pub_flag(avoidance);
  pub_global_path.publish(final_path);
  visualize_global_path(final_path);
  visualize_path(waypoints_group, frame_id, waypoints_idx_group);
  if(bestPathIndex != -1){
    visualize_path_best(waypoints_group, frame_id, bestPathIndex);
  }
}


int main(int argc, char **argv) {
    ros::init(argc, argv, "pointcloud_to_grid_node");
    dynamic_reconfigure::Server<my_dyn_rec::MyParamsConfig> server;
    dynamic_reconfigure::Server<my_dyn_rec::MyParamsConfig>::CallbackType f;
    f = boost::bind(&paramsCallback, _1, _2);
    server.setCallback(f);

    ros::NodeHandle nh;
    pub_igrid = nh.advertise<nav_msgs::OccupancyGrid>(grid_map.mapi_topic_name, 1);
    pub_hgrid = nh.advertise<nav_msgs::OccupancyGrid>(grid_map.maph_topic_name, 1);
    pub_path_ = nh.advertise<nav_msgs::Path>("/path", 1, true);
    pub_global_path_v = nh.advertise<nav_msgs::Path>("/global_path_v", 1, true);
    pub_best_path_ = nh.advertise<nav_msgs::Path>("/bestpath", 1, true);
    pub_G2L_path = nh.advertise<nav_msgs::Path>("/G2Lpath", 1, true);
    pub_G2L_path2 = nh.advertise<nav_msgs::Path>("/G2Lpath2", 1, true);
    pub_global_path = nh.advertise<waypoint_maker::Lane>("/lidar_path",1, true);
    pub_flag = nh.advertise<std_msgs::Bool>("/lidar_path_ctrl", 1, true);

    sub_pc2 = nh.subscribe(grid_map.cloud_in_topic, 1, pointcloudCallback);
    obs_sub = nh.subscribe("/obs_flag", 1, ObstacleCallback);
	  odom_sub_ = nh.subscribe("odom", 1, OdomCallback);
  	course_sub_ = nh.subscribe("course_path", 1, CourseCallback);
  	enddist_sub_ = nh.subscribe("/lidar_remain_dist", 1, DistCallback);
		state_sub_ = nh.subscribe("gps_state",1,StateCallback);
	  lane_sub_ = nh.subscribe("final_waypoints", 1, LaneCallback);
    

    ros::Rate loop_rate(10); // 10 Hz 루프

    while (ros::ok()) {
        if (one_time) {
          sigmoid();
        }
        ros::spinOnce(); // 콜백 처리
        run();
        loop_rate.sleep(); // 지정된 주기로 대기
    }

    return 0;
}
