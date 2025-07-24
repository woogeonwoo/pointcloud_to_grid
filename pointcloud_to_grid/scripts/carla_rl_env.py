#!/usr/bin/env python3

import gymnasium as gym
import numpy as np
import carla
import random
import time
import math
from collections import deque

class CarlaRLEnvironment(gym.Env):
    def __init__(self):
        super().__init__()
        
        # CARLA 연결
        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(10.0)
        self.world = self.client.get_world()
        
        # Action space: sigmoid 파라미터 [M, k, c]
        self.action_space = gym.spaces.Box(
            low=np.array([-6.0, 0.2, -10.0]), 
            high=np.array([6.0, 3.0, 10.0]), 
            dtype=np.float32
        )
        
        # State space: occupancy grid (100x100)
        self.observation_space = gym.spaces.Box(
            low=0, high=1, 
            shape=(100, 100, 1),
            dtype=np.float32
        )
        
        self.vehicle = None
        self.sensors = []
        self.collision_sensor = None
        self.lidar_data = None
        self.collision_detected = False
        
        # 경로 관련
        self.waypoints = []
        self.current_waypoint_idx = 0
        self.max_steps = 1000
        self.current_step = 0
        
        # 성능 메트릭
        self.path_efficiency = 0.0
        self.path_smoothness = 0.0
        self.collision_count = 0
        
    def reset(self, seed=None):
        super().reset(seed=seed)
        
        # 기존 액터들 정리
        self._cleanup()
        
        # 차량 스폰
        self._spawn_vehicle()
        
        # 센서 설정
        self._setup_sensors()
        
        # 목표점 설정
        self._set_random_goal()
        
        # 초기 상태
        self.current_step = 0
        self.collision_detected = False
        self.collision_count = 0
        
        # 초기 관측값 반환
        time.sleep(0.5)  # 센서 데이터 수집 대기
        return self._get_observation(), {}
    
    def step(self, action):
        self.current_step += 1
        
        # Sigmoid 파라미터 적용
        M, k, c = action
        
        # Sigmoid 경로 생성
        path = self._generate_sigmoid_path(M, k, c)
        
        # 경로 실행 및 평가
        reward, done = self._execute_and_evaluate_path(path, M, k, c)
        
        # 다음 관측값
        next_obs = self._get_observation()
        
        # 에피소드 종료 조건
        if self.current_step >= self.max_steps:
            done = True
            
        info = {
            'collision_count': self.collision_count,
            'path_efficiency': self.path_efficiency,
            'path_smoothness': self.path_smoothness,
            'sigmoid_params': {'M': M, 'k': k, 'c': c}
        }
        
        return next_obs, reward, done, False, info
    
    def _spawn_vehicle(self):
        """차량 스폰"""
        blueprint_library = self.world.get_blueprint_library()
        vehicle_bp = blueprint_library.find('vehicle.tesla.model3')
        
        # 랜덤 스폰 포인트
        spawn_points = self.world.get_map().get_spawn_points()
        spawn_point = random.choice(spawn_points)
        
        self.vehicle = self.world.spawn_actor(vehicle_bp, spawn_point)
        
    def _setup_sensors(self):
        """센서 설정"""
        blueprint_library = self.world.get_blueprint_library()
        
        # LiDAR 센서
        lidar_bp = blueprint_library.find('sensor.lidar.ray_cast')
        lidar_bp.set_attribute('channels', '64')
        lidar_bp.set_attribute('range', '50')
        lidar_bp.set_attribute('points_per_second', '100000')
        lidar_bp.set_attribute('rotation_frequency', '10')
        
        lidar_transform = carla.Transform(carla.Location(x=0, z=2.5))
        self.lidar = self.world.spawn_actor(lidar_bp, lidar_transform, attach_to=self.vehicle)
        self.lidar.listen(self._lidar_callback)
        self.sensors.append(self.lidar)
        
        # 충돌 센서
        collision_bp = blueprint_library.find('sensor.other.collision')
        self.collision_sensor = self.world.spawn_actor(collision_bp, carla.Transform(), attach_to=self.vehicle)
        self.collision_sensor.listen(self._collision_callback)
        self.sensors.append(self.collision_sensor)
    
    def _lidar_callback(self, data):
        """LiDAR 데이터 처리"""
        points = np.frombuffer(data.raw_data, dtype=np.float32)
        points = points.reshape((-1, 4))  # x, y, z, intensity
        self.lidar_data = points
    
    def _collision_callback(self, event):
        """충돌 감지"""
        self.collision_detected = True
        self.collision_count += 1
    
    def _generate_sigmoid_path(self, M, k, c):
        """Sigmoid 함수로 경로 생성 (기존 C++ 로직 포팅)"""
        waypoints = []
        
        # 기존 C++ 코드와 동일한 로직
        x_0 = 0.0
        for x in np.arange(50, -60, -0.5):
            point_y = M * (1 / (1 + np.exp(-k * (x - c))))
            waypoints.append([x, point_y])
            
            if abs(point_y) < 0.2:
                x_0 = x
                break
        
        # 경로 보정
        corrected_waypoints = []
        for i, (x, y) in enumerate(waypoints):
            corrected_x = x - x_0 + 10.0  # straight 거리
            if corrected_x >= 0.0 and corrected_x < 35.0:
                corrected_waypoints.append([corrected_x, y])
        
        return corrected_waypoints
    
    def _execute_and_evaluate_path(self, path, M, k, c):
        """경로 실행 및 평가"""
        if len(path) == 0:
            return -100.0, True  # 잘못된 경로
        
        reward = 0.0
        
        # 경로를 차량에 적용 (단순화된 버전)
        for i, (x, y) in enumerate(path[:min(10, len(path))]):  # 처음 10개 점만 실행
            # 차량 위치 기준으로 월드 좌표 변환
            vehicle_transform = self.vehicle.get_transform()
            
            # 로컬 좌표를 월드 좌표로 변환
            world_x = vehicle_transform.location.x + x * math.cos(math.radians(vehicle_transform.rotation.yaw)) - y * math.sin(math.radians(vehicle_transform.rotation.yaw))
            world_y = vehicle_transform.location.y + x * math.sin(math.radians(vehicle_transform.rotation.yaw)) + y * math.cos(math.radians(vehicle_transform.rotation.yaw))
            
            # 차량을 해당 위치로 이동
            new_transform = carla.Transform(
                carla.Location(x=world_x, y=world_y, z=vehicle_transform.location.z),
                vehicle_transform.rotation
            )
            self.vehicle.set_transform(new_transform)
            
            # 짧은 대기
            time.sleep(0.1)
            
            # 충돌 검사
            if self.collision_detected:
                reward -= 100.0
                return reward, True
        
        # 보상 계산
        reward += self._calculate_path_reward(path, M, k, c)
        
        return reward, False
    
    def _calculate_path_reward(self, path, M, k, c):
        """경로 품질 기반 보상 계산"""
        if len(path) == 0:
            return -50.0
        
        reward = 0.0
        
        # 1. 경로 길이 보상 (적절한 길이)
        path_length = len(path)
        if 20 <= path_length <= 50:
            reward += 10.0
        else:
            reward -= abs(path_length - 35) * 0.5
        
        # 2. 경로 부드러움 (곡률 변화)
        smoothness = self._calculate_smoothness(path)
        reward += smoothness * 5.0
        
        # 3. 경로 효율성 (목표 방향성)
        efficiency = self._calculate_efficiency(path)
        reward += efficiency * 10.0
        
        # 4. 파라미터 합리성
        if 0.5 <= abs(M) <= 5.0:
            reward += 5.0
        if 0.3 <= k <= 2.5:
            reward += 5.0
        if -8.0 <= c <= 8.0:
            reward += 5.0
        
        return reward
    
    def _calculate_smoothness(self, path):
        """경로 부드러움 계산"""
        if len(path) < 3:
            return -1.0
        
        curvatures = []
        for i in range(1, len(path) - 1):
            p1, p2, p3 = path[i-1], path[i], path[i+1]
            
            # 곡률 계산 (간단한 각도 변화)
            v1 = [p2[0] - p1[0], p2[1] - p1[1]]
            v2 = [p3[0] - p2[0], p3[1] - p2[1]]
            
            angle_change = abs(math.atan2(v2[1], v2[0]) - math.atan2(v1[1], v1[0]))
            curvatures.append(angle_change)
        
        # 곡률 변화의 표준편차 (작을수록 부드러움)
        if len(curvatures) > 0:
            smoothness = 1.0 / (1.0 + np.std(curvatures))
        else:
            smoothness = 0.0
            
        return smoothness
    
    def _calculate_efficiency(self, path):
        """경로 효율성 계산"""
        if len(path) < 2:
            return -1.0
        
        # 시작점에서 끝점까지의 직선 거리
        start, end = path[0], path[-1]
        straight_distance = math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
        
        # 실제 경로 길이
        actual_distance = 0.0
        for i in range(1, len(path)):
            p1, p2 = path[i-1], path[i]
            actual_distance += math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
        
        # 효율성 = 직선거리 / 실제거리
        if actual_distance > 0:
            efficiency = straight_distance / actual_distance
        else:
            efficiency = 0.0
            
        return efficiency
    
    def _get_observation(self):
        """현재 상태 관측값 생성"""
        # LiDAR 데이터를 occupancy grid로 변환
        if self.lidar_data is None:
            return np.zeros((100, 100, 1), dtype=np.float32)
        
        # 간단한 occupancy grid 생성
        grid = np.zeros((100, 100), dtype=np.float32)
        
        for point in self.lidar_data:
            x, y, z = point[0], point[1], point[2]
            
            # 그리드 좌표로 변환 (50m 범위, 0.5m 해상도)
            grid_x = int((x + 25) / 0.5)
            grid_y = int((y + 25) / 0.5)
            
            if 0 <= grid_x < 100 and 0 <= grid_y < 100 and -0.5 < z < 3.0:
                grid[grid_y, grid_x] = 1.0
        
        return grid.reshape((100, 100, 1))
    
    def _set_random_goal(self):
        """랜덤 목표점 설정"""
        # 현재는 단순히 전방 랜덤 위치로 설정
        self.goal_x = random.uniform(20, 40)
        self.goal_y = random.uniform(-10, 10)
    
    def _cleanup(self):
        """리소스 정리"""
        if self.vehicle is not None:
            self.vehicle.destroy()
            
        for sensor in self.sensors:
            if sensor is not None:
                sensor.destroy()
                
        self.sensors.clear()
        self.vehicle = None
        self.collision_sensor = None
        self.lidar_data = None
        self.collision_detected = False

    def close(self):
        self._cleanup()
