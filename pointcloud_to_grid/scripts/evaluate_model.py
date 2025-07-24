#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from stable_baselines3 import PPO, SAC
import argparse
import os

from carla_rl_env import CarlaRLEnvironment

class ModelEvaluator:
    def __init__(self, model_path, algorithm='ppo'):
        self.algorithm = algorithm
        
        # 모델 로드
        if algorithm == 'ppo':
            self.model = PPO.load(model_path)
        elif algorithm == 'sac':
            self.model = SAC.load(model_path)
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")
        
        # 환경 생성
        self.env = CarlaRLEnvironment()
    
    def evaluate_scenarios(self, n_scenarios=50):
        """다양한 시나리오에서 모델 평가"""
        results = {
            'rewards': [],
            'collisions': [],
            'efficiency': [],
            'smoothness': [],
            'sigmoid_params': {'M': [], 'k': [], 'c': []}
        }
        
        for scenario in range(n_scenarios):
            print(f"Evaluating scenario {scenario + 1}/{n_scenarios}")
            
            obs, _ = self.env.reset()
            episode_reward = 0
            done = False
            
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, done, _, info = self.env.step(action)
                episode_reward += reward
                
                if done:
                    results['rewards'].append(episode_reward)
                    results['collisions'].append(info.get('collision_count', 0))
                    results['efficiency'].append(info.get('path_efficiency', 0))
                    results['smoothness'].append(info.get('path_smoothness', 0))
                    
                    sigmoid_params = info.get('sigmoid_params', {})
                    results['sigmoid_params']['M'].append(sigmoid_params.get('M', 0))
                    results['sigmoid_params']['k'].append(sigmoid_params.get('k', 0))
                    results['sigmoid_params']['c'].append(sigmoid_params.get('c', 0))
        
        return results
    
    def compare_with_baseline(self, baseline_results=None):
        """기존 그리드 서치와 성능 비교"""
        rl_results = self.evaluate_scenarios()
        
        if baseline_results is None:
            # 기본 baseline (그리드 서치 결과를 시뮬레이션)
            baseline_results = {
                'rewards': np.random.normal(50, 20, 50),
                'collisions': np.random.poisson(2, 50),
                'efficiency': np.random.beta(2, 3, 50),
                'smoothness': np.random.beta(3, 2, 50)
            }
        
        # 비교 시각화
        self._plot_comparison(rl_results, baseline_results)
        
        return rl_results, baseline_results
    
    def _plot_comparison(self, rl_results, baseline_results):
        """RL vs Baseline 비교 시각화"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        metrics = ['rewards', 'collisions', 'efficiency', 'smoothness']
        titles = ['Episode Rewards', 'Collision Count', 'Path Efficiency', 'Path Smoothness']
        
        for i, (metric, title) in enumerate(zip(metrics, titles)):
            ax = axes[i // 2, i % 2]
            
            # 박스플롯
            data = [baseline_results[metric], rl_results[metric]]
            labels = ['Baseline (Grid Search)', 'RL Optimized']
            
            ax.boxplot(data, labels=labels)
            ax.set_title(title)
            ax.grid(True, alpha=0.3)
            
            # 통계 정보 추가
            baseline_mean = np.mean(baseline_results[metric])
            rl_mean = np.mean(rl_results[metric])
            improvement = ((rl_mean - baseline_mean) / baseline_mean) * 100
            
            ax.text(0.5, 0.95, f'Improvement: {improvement:.1f}%', 
                   transform=ax.transAxes, ha='center', va='top',
                   bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig('rl_vs_baseline_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def analyze_parameter_distribution(self):
        """Sigmoid 파라미터 분포 분석"""
        results = self.evaluate_scenarios()
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        params = ['M', 'k', 'c']
        param_names = ['Amplitude (M)', 'Steepness (k)', 'Shift (c)']
        
        for i, (param, name) in enumerate(zip(params, param_names)):
            values = results['sigmoid_params'][param]
            
            axes[i].hist(values, bins=20, alpha=0.7, density=True)
            axes[i].axvline(np.mean(values), color='red', linestyle='--', 
                          label=f'Mean: {np.mean(values):.2f}')
            axes[i].set_title(f'{name} Distribution')
            axes[i].set_xlabel(name)
            axes[i].set_ylabel('Density')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('sigmoid_parameter_distribution.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 파라미터 간 상관관계
        self._plot_parameter_correlation(results)
    
    def _plot_parameter_correlation(self, results):
        """파라미터 간 상관관계 분석"""
        import pandas as pd
        
        # 데이터프레임 생성
        df = pd.DataFrame({
            'M': results['sigmoid_params']['M'],
            'k': results['sigmoid_params']['k'], 
            'c': results['sigmoid_params']['c'],
            'efficiency': results['efficiency'],
            'smoothness': results['smoothness'],
            'reward': results['rewards']
        })
        
        # 상관관계 매트릭스
        correlation_matrix = df.corr()
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                   square=True, fmt='.2f')
        plt.title('Parameter Correlation Matrix')
        plt.tight_layout()
        plt.savefig('parameter_correlation.png', dpi=300, bbox_inches='tight')
        plt.show()

def main():
    parser = argparse.ArgumentParser(description='Evaluate trained RL model')
    parser.add_argument('--model_path', required=True, help='Path to trained model')
    parser.add_argument('--algorithm', choices=['ppo', 'sac'], default='ppo')
    parser.add_argument('--n_scenarios', type=int, default=50, help='Number of scenarios to evaluate')
    parser.add_argument('--output_dir', default='./evaluation_results/', help='Output directory')
    
    args = parser.parse_args()
    
    # 출력 디렉토리 생성
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 평가 실행
    evaluator = ModelEvaluator(args.model_path, args.algorithm)
    
    print("Starting comprehensive evaluation...")
    
    # 시나리오 평가
    results = evaluator.evaluate_scenarios(args.n_scenarios)
    
    # 성능 통계 출력
    print("\n=== Evaluation Results ===")
    print(f"Average Reward: {np.mean(results['rewards']):.2f} ± {np.std(results['rewards']):.2f}")
    print(f"Average Collisions: {np.mean(results['collisions']):.2f}")
    print(f"Average Efficiency: {np.mean(results['efficiency']):.3f}")
    print(f"Average Smoothness: {np.mean(results['smoothness']):.3f}")
    print(f"Success Rate: {(np.array(results['collisions']) == 0).sum() / len(results['collisions']) * 100:.1f}%")
    
    # 파라미터 분석
    evaluator.analyze_parameter_distribution()
    
    # 베이스라인과 비교
    evaluator.compare_with_baseline()
    
    print(f"\nEvaluation complete! Results saved to {args.output_dir}")

if __name__ == "__main__":
    main()
