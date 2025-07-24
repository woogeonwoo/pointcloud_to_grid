#pragma once  // 헤더 파일 중복 포함 방지
#include <ros/ros.h>
#include <nav_msgs/OccupancyGrid.h>
#include <queue>
#include <stack>
#include <vector>
#include <limits>
#include <std_msgs/Float64.h>
#include <std_msgs/Bool.h>
#include <nav_msgs/Path.h>
#include <nav_msgs/Odometry.h>
#include "waypoint_maker/Waypoint.h"
#include "waypoint_maker/Lane.h"
#include "waypoint_maker/State.h"


class PointXY{
public: 
  int x;
  int y;
};

class PointXYZI{
public: 
  double x;
  double y;
  double z;
  double intensity;
};

class PointXYZD {
public:
    double x, y, distance;
    signed char height;

    PointXYZD(){
      x=0;
      y=0;
      height=-128;
      distance=0;
    }
    PointXYZD(double x_, double y_, signed char h_, double d_){
      x=x_;
      y=y_;
      height=h_;
      distance=d_;
    }
     // 새로운 생성자
};

class GridMap{
  public: 
    float position_x; // GridMap의 중심 위치, 파라미터 조정 값(이 값에 따라 중심이 변함)
    float position_y;
    float cell_size;  // 각 셀의 크기(해상도)를 나타냅니다. 단위는 미터
    float length_x; // GridMap의 가로(폭)와 세로(높이) 길이
    float length_y;
    std::string cloud_in_topic; // pointcloud2 topic
    std::string frame_out;      // OccupancyGrid 메시지의 좌표계를 정의할 프레임 ID
    std::string mapi_topic_name;  // OccupancyGrid 메시지를 퍼블리시할 토픽
    std::string maph_topic_name;
    float topleft_x;  // GridMap의 좌측 상단 좌표를 계산하여 저장
    float topleft_y;  
    float bottomright_x;  // GridMap의 우측 하단 좌표를 계산하여 저장
    float bottomright_y;
    int cell_num_x; // GridMap의 가로 방향과 세로 방향의 셀 개수
    int cell_num_y;
    float intensity_factor; // PointCloud에서 얻은 intensity 값을 OccupancyGrid로 변환할 때 사용하는 보정 계수
    float height_factor;    // PointCloud의 높이 값을 OccupancyGrid로 변환할 때 사용하는 보정 계수
    


    void initGrid(nav_msgs::OccupancyGridPtr grid) {
      grid->header.seq = 1;
      grid->header.frame_id = GridMap::frame_out; // TODO
      grid->info.origin.position.z = 0;  // 2D 라서 z=0
      grid->info.origin.orientation.w = 0; // 초기 회전
      grid->info.origin.orientation.x = 0;
      grid->info.origin.orientation.y = 0;
      grid->info.origin.orientation.z = 1;  // 쿼터니언 벡터의 크기는 항상 1로 만들기 위해
      grid->info.origin.position.x = position_x + length_x / 2; // 중심으로 부터 map 크기
      grid->info.origin.position.y = position_y + length_y / 2;
      grid->info.width = length_x / cell_size;  // 가로 방향의 셀 개수
      grid->info.height = length_y /cell_size;  // 세로 방향의 셀 개수
      grid->info.resolution = cell_size;        // 해상도
      // resolution/grid size [m/cell]
    }
    
    void paramRefresh(){  // 변수 업데이트
      topleft_x = position_x + length_x / 2;  // map 경계 설정
      bottomright_x = position_x - length_x / 2;
      topleft_y = position_y + length_y / 2;
      bottomright_y = position_y - length_y / 2;
      cell_num_x = int(length_x / cell_size); // 셀 개수
      cell_num_y = int(length_y / cell_size);
      if(cell_num_x > 0){
        ROS_INFO_STREAM("Cells: " << cell_num_x << "*" << cell_num_y << "px, subscribed to " << GridMap::cloud_in_topic << " [" << topleft_x << ", " << topleft_y << "]" << " [" << bottomright_x << ", " << bottomright_y << "]");
      }
      std::cout << "topleft_x : " << topleft_x << std::endl;
      std::cout << "topleft_y : " << topleft_y << std::endl;
      std::cout << "bottomright_x : " << bottomright_x << std::endl;
      std::cout << "bottomright_y : " << bottomright_y << std::endl;
    }

    // number of cells
    int getSize(){
      return cell_num_x * cell_num_y;
    }
    
    // number of cells
    int getSizeX(){
      return cell_num_x;
    }

    // number of cells
    int getSizeY(){
      return cell_num_y;
    }

    // length [m] meters
    double getLengthX(){
      return length_x;
    }

    // length [m] meters
    double getLengthY(){
      return length_y;
    }

    // resolution [m/cell] size of a single cell
    double getResolution(){
      return cell_size;
    }
};