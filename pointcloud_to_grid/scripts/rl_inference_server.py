#!/usr/bin/env python3

import rospy
import numpy as np
from stable_baselines3 import PPO
from pointcloud_to_grid.msg import RLSigmoidParams
from nav_msgs.msg import OccupancyGrid

class RLSigmoidServer:
    def __init__(self):
        rospy.init_node('rl_sigmoid_server')
        
        # 훈련된 모델 로드
        model_path = rospy.get_param('~model_path', 
                                   '$(find pointcloud_to_grid)/models/trained_sigmoid_model.zip')
        self.model = PPO.load(model_path)
        
        # ROS 통신
        self.grid_sub = rospy.Subscriber('/height_grid', OccupancyGrid, self.grid_callback)
        self.params_pub = rospy.Publisher('/rl_sigmoid_params', RLSigmoidParams, queue_size=1)
        
    def grid_callback(self, grid_msg):
        # occupancy grid를 RL 모델 입력으로 변환
        state = self.preprocess_grid(grid_msg)
        
        # RL 모델로 최적 sigmoid 파라미터 예측
        action, _ = self.model.predict(state, deterministic=True)
        
        # 결과 발행
        params_msg = RLSigmoidParams()
        params_msg.M = float(action[0])
        params_msg.k = float(action[1])
        params_msg.c = float(action[2])
        params_msg.confidence = 0.95  # 모델 신뢰도
        
        self.params_pub.publish(params_msg)
        
    def preprocess_grid(self, grid_msg):
        # grid 데이터를 numpy 배열로 변환
        grid_data = np.array(grid_msg.data, dtype=np.float32)
        grid_2d = grid_data.reshape((grid_msg.info.height, grid_msg.info.width))
        
        # 정규화 및 전처리
        grid_normalized = (grid_2d + 128) / 256.0  # -128~127 -> 0~1
        
        return grid_normalized

if __name__ == '__main__':
    server = RLSigmoidServer()
    rospy.spin()
