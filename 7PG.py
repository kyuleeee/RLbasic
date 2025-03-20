
#### 1.  Monte Carlo Control Policy #### 
#==============================================================================

import numpy as np

num_states = 5
num_actions = 2
num_episodes = 5000

theta = np.random.randn(num_states, num_actions)  # 각 상태에서 행동별 가중치
#원래 사실 theta를 어떻게 initialize 해야하는지 감이 안잡혔었음.

# Softmax 정책 함수
def softmax_policy(state, theta):
    preferences = theta[state]  # 해당 상태의 행동별 가중치 가져오기
    exp_values = np.exp(preferences - np.max(preferences))  # 안정적인 softmax
    return exp_values / np.sum(exp_values)

# 확률에 따라 행동 선택
def choose_action(state, theta):
    probs = softmax_policy(state, theta)
    return np.random.choice(num_actions, p=probs)

# 한 번의 에피소드를 실행하면서 데이터를 모으는 코드 (Monte Carlo)
def run_episode(theta, max_steps=10):
    episode = []
    state = np.random.randint(0, num_states)  # 랜덤한 초기 상태

    for _ in range(max_steps):  # max_steps 이후 종료
        action = choose_action(state, theta)
        next_state = np.random.randint(0, num_states)
        reward = np.random.randn()
        
        episode.append((state, action, reward))
        state = next_state

    return episode

alpha = 0.01  # 학습률
gamma = 0.9   # 감가율 (할인율)

# GLIE Monte Carlo Control 학습
for episode_num in range(1, num_episodes + 1):
    episode = run_episode(theta) 
    
    G = 0  # 리턴값 초기화

    for state, action, reward in reversed(episode):
        G = reward + gamma * G  # 할인된 보상 계산
        probs = softmax_policy(state, theta)

        grad = np.zeros_like(theta[state])
        #gradient를 저장하는 저장소! 
        grad[action] = 1 - probs[action]  # log 확률의 gradient가 이렇게 계산이 되고 
        #여기서 헷갈렸던 건, 선택한 행동 선택한 행동 a에 대해서만 기울기를 계산
        theta[state] += alpha * grad * G  # 정책 파라미터 업데이트
        #alpha * log gradient * G 이렇게 되는 것임. 

# 학습된 파라미터 출력
print("Theta:")
print(theta)
