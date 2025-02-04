import gymnasium as gym
import numpy as np
import time
from dynamixel_sdk import PortHandler, PacketHandler

class MyRealEnv(gym.Env):
    def __init__(self):
        super(MyRealEnv, self).__init__()
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(7,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(480, 640, 3), dtype=np.uint8)
        
        self.dxl_ids = [10, 11, 12, 20, 21, 22, 30, 31, 32]

    def reset(self):
        # 重置環境狀態
        self.state = np.random.rand(5)  # 假設初始狀態為隨機值
        return self.state

    def step(self, action):
        # 根據動作更新狀態
        # 執行你的硬體控制邏輯
        # 獲取當前狀態，獎勵和是否完成的標誌
        reward = self.compute_reward(action)
        done = False  # 根據你的邏輯決定是否完成
        self.state = np.random.rand(5)  # 更新狀態
        return self.state, reward, done, {}

    def compute_reward(self, action):
        # 計算獎勵
        return np.random.rand()  # 假設獎勵為隨機值

    def render(self, mode='human'):
        # 可選：渲染環境
        pass

# 使用 Stable Baselines3 訓練模型
from stable_baselines3 import PPO

env = MyRealEnv()
model = PPO("MlpPolicy", env, verbose=1)

# 訓練模型
model.learn(total_timesteps=10000)

# 測試模型
obs = env.reset()
for _ in range(1000):
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    env.render()
    time.sleep(0.1)  # 模擬延遲
    if done:
        obs = env.reset()

# 清理
env.close()
