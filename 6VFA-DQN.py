import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

# 하이퍼파라미터 설정
num_episodes = 1000
epsilon_start = 1.0
epsilon_end = 0.1
epsilon_decay = 0.995
gamma = 0.99  # 할인율
batch_size = 32
learning_rate = 0.0001
update_target_every = 10  # 타겟 네트워크 업데이트 주기

# 환경 설정 (예시로 OpenAI Gym 환경을 사용)
import gym
env = gym.make('CartPole-v1')

# Q 네트워크 정의
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Q 네트워크와 타겟 네트워크 초기화
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
q_network = QNetwork(state_dim, action_dim)
target_network = QNetwork(state_dim, action_dim)
target_network.load_state_dict(q_network.state_dict())  # 타겟 네트워크 초기화
target_network.eval()  # 타겟 네트워크는 평가 모드

# 옵티마이저
optimizer = optim.Adam(q_network.parameters(), lr=learning_rate)

# 경험 리플레이 버퍼
memory = deque(maxlen=10000)

# epsilon-greedy 정책
def epsilon_greedy(state, epsilon):
    if random.random() < epsilon:
        return env.action_space.sample()  # 랜덤 행동
    else:
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        q_values = q_network(state_tensor)  # Q값 예측
        return torch.argmax(q_values).item()  # 가장 큰 Q값을 가지는 행동 선택

# 경험 리플레이에서 미니 배치 샘플링
def get_mini_batch(batch_size):
    batch = random.sample(memory, batch_size)
    states, actions, rewards, next_states, dones = zip(*batch)
    return torch.FloatTensor(states), torch.LongTensor(actions), torch.FloatTensor(rewards), torch.FloatTensor(next_states), torch.BoolTensor(dones)

# 학습 루프
epsilon = epsilon_start
for episode in range(num_episodes):
    state = env.reset()
    total_reward = 0
    done = False

    while not done:
        action = epsilon_greedy(state, epsilon)
        next_state, reward, done, _ = env.step(action)
        memory.append((state, action, reward, next_state, done))  # 경험 리플레이 버퍼에 저장

        state = next_state
        total_reward += reward

        # 미니 배치로 학습
        if len(memory) > batch_size:
            states, actions, rewards, next_states, dones = get_mini_batch(batch_size)
            
            # 현재 Q 네트워크에서 Q값 계산
            q_values = q_network(states)
            q_value = q_values.gather(1, actions.unsqueeze(1))  # 현재 행동에 대한 Q값

            # 타겟 네트워크에서 Q값 계산
            next_q_values = target_network(next_states)
            next_q_value = next_q_values.max(1)[0]  # 가장 큰 Q값

            # 타겟 Q값 계산
            target = rewards + gamma * next_q_value * (~dones)

            # 손실 함수 계산
            loss = nn.MSELoss()(q_value.squeeze(), target)

            # 역전파
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # epsilon 값 감소
    epsilon = max(epsilon_end, epsilon * epsilon_decay)

    # 주기적으로 타겟 네트워크 업데이트
    if episode % update_target_every == 0:
        target_network.load_state_dict(q_network.state_dict())

    print(f"Episode {episode+1}/{num_episodes}, Total Reward: {total_reward}")

print("학습 완료!")
