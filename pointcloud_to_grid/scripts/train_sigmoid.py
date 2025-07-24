#!/usr/bin/env python3

import os
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.vec_env import VecNormalize
import argparse
import yaml

from carla_rl_env import CarlaRLEnvironment

class TensorboardCallback(BaseCallback):
    """훈련 진행상황을 로깅하는 콜백"""
    
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        
    def _on_step(self) -> bool:
        # 에피소드 완료시 메트릭 로깅
        if self.locals['dones'][0]:
            info = self.locals['infos'][0]
            
            # 보상 및 길이 기록
            episode_reward = np.sum(self.locals['rewards'])
            episode_length = info.get('episode_length', 0)
            
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(episode_length)
            
            # Tensorboard에 로깅
            self.logger.record("episode/reward", episode_reward)
            self.logger.record("episode/length", episode_length)
            self.logger.record("episode/collision_count", info.get('collision_count', 0))
            self.logger.record("episode/path_efficiency", info.get('path_efficiency', 0))
            self.logger.record("episode/path_smoothness", info.get('path_smoothness', 0))
            
            # Sigmoid 파라미터 로깅
            sigmoid_params = info.get('sigmoid_params', {})
            if sigmoid_params:
                self.logger.record("sigmoid/M", sigmoid_params.get('M', 0))
                self.logger.record("sigmoid/k", sigmoid_params.get('k', 0))
                self.logger.record("sigmoid/c", sigmoid_params.get('c', 0))
        
        return True

def train_ppo(config):
    """PPO 알고리즘으로 훈련"""
    print("Starting PPO training...")
    
    # 환경 생성
    env = make_vec_env(CarlaRLEnvironment, n_envs=config['n_envs'])
    env = VecNormalize(env, norm_obs=True, norm_reward=True)
    
    # 모델 생성
    model = PPO(
        policy="CnnPolicy",
        env=env,
        learning_rate=config['learning_rate'],
        n_steps=config['n_steps'],
        batch_size=config['batch_size'],
        n_epochs=config['n_epochs'],
        gamma=config['gamma'],
        gae_lambda=config['gae_lambda'],
        clip_range=config['clip_range'],
        ent_coef=config['ent_coef'],
        vf_coef=config['vf_coef'],
        max_grad_norm=config['max_grad_norm'],
        verbose=1,
        tensorboard_log="./tensorboard_logs/"
    )
    
    # 콜백 설정
    callback = TensorboardCallback()
    
    # 훈련
    model.learn(
        total_timesteps=config['total_timesteps'],
        callback=callback,
        progress_bar=True
    )
    
    return model, env

def train_sac(config):
    """SAC 알고리즘으로 훈련"""
    print("Starting SAC training...")
    
    # 환경 생성
    env = make_vec_env(CarlaRLEnvironment, n_envs=1)  # SAC는 보통 단일 환경
    env = VecNormalize(env, norm_obs=True, norm_reward=True)
    
    # 모델 생성
    model = SAC(
        policy="CnnPolicy",
        env=env,
        learning_rate=config['learning_rate'],
        buffer_size=config['buffer_size'],
        learning_starts=config['learning_starts'],
        batch_size=config['batch_size'],
        tau=config['tau'],
        gamma=config['gamma'],
        train_freq=config['train_freq'],
        gradient_steps=config['gradient_steps'],
        verbose=1,
        tensorboard_log="./tensorboard_logs/"
    )
    
    # 콜백 설정
    callback = TensorboardCallback()
    
    # 훈련
    model.learn(
        total_timesteps=config['total_timesteps'],
        callback=callback,
        progress_bar=True
    )
    
    return model, env

def evaluate_model(model, env, n_episodes=10):
    """모델 성능 평가"""
    print(f"Evaluating model for {n_episodes} episodes...")
    
    episode_rewards = []
    collision_counts = []
    efficiency_scores = []
    smoothness_scores = []
    
    for episode in range(n_episodes):
        obs = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            episode_reward += reward[0]
            
            if done[0]:
                episode_rewards.append(episode_reward)
                collision_counts.append(info[0].get('collision_count', 0))
                efficiency_scores.append(info[0].get('path_efficiency', 0))
                smoothness_scores.append(info[0].get('path_smoothness', 0))
    
    # 결과 출력
    print(f"Average Reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"Average Collisions: {np.mean(collision_counts):.2f}")
    print(f"Average Efficiency: {np.mean(efficiency_scores):.3f}")
    print(f"Average Smoothness: {np.mean(smoothness_scores):.3f}")
    
    return {
        'rewards': episode_rewards,
        'collisions': collision_counts,
        'efficiency': efficiency_scores,
        'smoothness': smoothness_scores
    }

def visualize_training_results(results, save_path=None):
    """훈련 결과 시각화"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 보상 히스토그램
    axes[0, 0].hist(results['rewards'], bins=20, alpha=0.7)
    axes[0, 0].set_title('Episode Rewards Distribution')
    axes[0, 0].set_xlabel('Reward')
    axes[0, 0].set_ylabel('Frequency')
    
    # 충돌 횟수
    axes[0, 1].hist(results['collisions'], bins=10, alpha=0.7, color='red')
    axes[0, 1].set_title('Collision Count Distribution')
    axes[0, 1].set_xlabel('Collisions per Episode')
    axes[0, 1].set_ylabel('Frequency')
    
    # 효율성
    axes[1, 0].hist(results['efficiency'], bins=20, alpha=0.7, color='green')
    axes[1, 0].set_title('Path Efficiency Distribution')
    axes[1, 0].set_xlabel('Efficiency Score')
    axes[1, 0].set_ylabel('Frequency')
    
    # 부드러움
    axes[1, 1].hist(results['smoothness'], bins=20, alpha=0.7, color='orange')
    axes[1, 1].set_title('Path Smoothness Distribution')
    axes[1, 1].set_xlabel('Smoothness Score')
    axes[1, 1].set_ylabel('Frequency')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Training results saved to {save_path}")
    
    plt.show()

def load_config(config_path):
    """설정 파일 로드"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def main():
    parser = argparse.ArgumentParser(description='Train RL model for sigmoid path optimization')
    parser.add_argument('--algorithm', choices=['ppo', 'sac'], default='ppo',
                        help='RL algorithm to use')
    parser.add_argument('--config', default='../config/rl_params.yaml',
                        help='Config file path')
    parser.add_argument('--output_dir', default='../models/',
                        help='Output directory for trained models')
    parser.add_argument('--evaluate', action='store_true',
                        help='Evaluate trained model')
    
    args = parser.parse_args()
    
    # 설정 로드
    config = load_config(args.config)
    
    # 출력 디렉토리 생성
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 훈련 또는 평가
    if args.evaluate:
        # 기존 모델 로드 및 평가
        model_path = os.path.join(args.output_dir, f'trained_sigmoid_{args.algorithm}.zip')
        if args.algorithm == 'ppo':
            model = PPO.load(model_path)
        else:
            model = SAC.load(model_path)
        
        env = make_vec_env(CarlaRLEnvironment, n_envs=1)
        results = evaluate_model(model, env)
        visualize_training_results(results, 
                                 os.path.join(args.output_dir, f'evaluation_results_{args.algorithm}.png'))
    else:
        # 훈련
        if args.algorithm == 'ppo':
            model, env = train_ppo(config[args.algorithm])
        else:
            model, env = train_sac(config[args.algorithm])
        
        # 모델 저장
        model_path = os.path.join(args.output_dir, f'trained_sigmoid_{args.algorithm}')
        model.save(model_path)
        env.save(os.path.join(args.output_dir, f'env_normalize_{args.algorithm}.pkl'))
        
        print(f"Model saved to {model_path}")
        
        # 평가
        results = evaluate_model(model, env)
        visualize_training_results(results, 
                                 os.path.join(args.output_dir, f'training_results_{args.algorithm}.png'))

if __name__ == "__main__":
    main()
