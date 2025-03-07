"""
THis is a code for Model Free Control in RL. 
"""

# ###1. Monte Carlo Value Function Approximation ###
# #==============================================================================
import numpy as np

num_states = 5    # 상태 개수
num_actions = 2   # 행동 개수
num_episodes = 5000  

# 가중치 초기화 (VFA용)
w = np.zeros(num_states)  # feature vector의 크기만큼 가중치 생성

# ε-greedy policy [만약, 저 epsilon이 고정이라면 그냥 exploration만 계속하게 됨]
def epsilon_greedy_policy(w, state, epsilon):
    # 행동에 대한 VFA 값 계산
    action_values = [VFA(state, w) for _ in range(num_actions)]
    
    # 가장 높은 VFA 값을 가지는 행동 선택
    best_action = np.argmax(action_values)
    
    # ε-greedy 정책
    policy = np.full(num_actions, epsilon / num_actions)   # 모든 행동에 대해 ε/m 확률 부여
    policy[best_action] += 1 - epsilon # 가장 좋은 행동에는 1 - ε 추가 확률 
    
    return np.random.choice(np.arange(num_actions), p=policy)

# 한 번의 에피소드를 실행하면서 데이터를 모으는 코드 (Monte Carlo)
def run_episode(epsilon, max_steps=10):
    episode = []  
    state = np.random.randint(0, num_states)  # 랜덤한 초기 상태

    while True:
        action = epsilon_greedy_policy(w, state, epsilon) #epsilon_greedy_policy로 탐색과정! 
        next_state = np.random.randint(0, num_states)  #
        reward = np.random.randn()
        
        episode.append((state, action, reward)) 
        state = next_state

        if len(episode) >= max_steps:  # 최대 10번 이동 후 종료
            break
    return episode
def feature_vector(state, num_states):
    vec = np.zeros(num_states)
    vec[state] = 1  
    return vec
  
def VFA(state, w):
    return np.dot(w, feature_vector(state, num_states))  # 상태에 대한 가중치 적용
  
# GLIE Monte Carlo Control 학습
for episode_num in range(1, num_episodes+1):
    epsilon = 1 / episode_num # episode를 점점 늘려갈 때마다 epsilon값이 줄어든다. 
    episode = run_episode(epsilon) #매번 epsilon greedy policy로 탐색을 해서 episode를 만들어낸다. 
    
    G = 0  # 매번 그냥 0에서부터 시작. 그 에피소드에서는 G=0인 것임.
    visited = set()  # 첫 방문 판별용
    
    for state, action, reward in reversed(episode):
        G = reward + 0.9 * G  # 감가율  = 0.9 
        
        if (state, action) not in visited:  # 첫 방문인지 확인 
            visited.add((state, action)) 
            state_vec = feature_vector(state,num_states) # 상태 벡터 생성
            alpha = 1 / (episode_num + 1)
            w += alpha * (G - VFA(state, w)) * state_vec  # w 업데이트
            #Q[state, action] += alpha * (G - Q[state, action])여기서는 이걸 할 필요가 없음. 단순히 w만 업데이트 하면 됨. 
            
print("학습된 Q 테이블:")
print(Q)


#==============================================================================

#### 2.SARSA ####
w = np.zeros(num_states * num_actions) # feature vector의 크기만큼 가중치 생성
def epsilon_greedy_policy(w, state, epsilon):
    # 행동에 대한 VFA 값 계산
    action_values = [VFA(state, w) for _ in range(num_actions)]
    
    # 가장 높은 VFA 값을 가지는 행동 선택
    best_action = np.argmax(action_values)
    
    # ε-greedy 정책
    policy = np.full(num_actions, epsilon / num_actions)   # 모든 행동에 대해 ε/m 확률 부여
    policy[best_action] += 1 - epsilon # 가장 좋은 행동에는 1 - ε 추가 확률 
    
    return np.random.choice(np.arange(num_actions), p=policy)
def run_episode_next(epsilon, max_steps=10):
    episode = []  
    state = np.random.randint(0, num_states)  # 랜덤한 초기 상태
    action = epsilon_greedy_policy(w, state, epsilon)  # 초기 행동 선택

    while True:
        next_state = np.random.randint(0, num_states)  # 임의의 환경 이동
        reward = np.random.randn()
        next_action = epsilon_greedy_policy(w, next_state, epsilon) # 다음 행동 선택
        
        episode.append((state, action, reward, next_state, next_action)) 
        
        state, action = next_state, next_action  # SARSA는 (s, a, r, s', a') 방식으로 진행!

        if len(episode) >= max_steps:  # 최대 10번 이동 후 종료
            break
    return episode   


def feature_vector(state,action,num_states,num_actions):
    vec = np.zeros(num_states * num_actions)
    vec[state * num_actions + action] = 1  
    return vec
  
def VFA(state, action, w):
    return np.dot(w, feature_vector(state, action, num_states, num_actions))  # state-action에 대한 가중치 적용
  
for episode_num in range(1, num_episodes+1):
    epsilon = 1 / episode_num # episode를 점점 늘려갈 때마다 epsilon값이 줄어든다. 
    episode = run_episode_next(epsilon)  
    
    visited = set()  # 첫 방문 판별용
    
    for state, action, reward, next_state, next_action in reversed(episode):
        if (state, action) not in visited:  
            visited.add((state, action)) 
            state_action_vec = feature_vector(state,action, num_states,num_actions) # 상태 벡터 생성
            alpha = 1 / (episode_num + 1)
            w += alpha * (reward + 0.9 * VFA(next_state, next_action, w) - VFA(state, action, w)) * state_action_vec
            
print("학습된 가중치 벡터 w:")
print(w)


#### 3.n-step SARSA ####
n_step = 3  
alpha = 0.1

for episode_num in range(1, num_episodes + 1):
    epsilon = 1 / episode_num  # Exploration 감소
    episode = run_episode_next(epsilon)  # 에피소드 진행

    for t in range(len(episode)):  # 각 time step에 대해 반복
        G = 0  # n-step return 초기화

        # 1. n-step 동안 보상 계산
        for i in range(n_step):  # n-step동안 보상 계산
            if t + i < len(episode):
                G += (0.9 ** i) * episode[t + i][2]  # 보상 추가 n-step 동안 보상 계산 : 이거는 건들 수 없고, 건들 필요가 없음. 

        # 2. n-step 후의 Q 값 추가 여기서 건든다.
        if t + n_step < len(episode):
            G += (0.9 ** n_step) * VFA(episode[t + n_step][3], episode[t + n_step][4], w)

        # 3. Q 업데이트
        state, action = episode[t][:2]  # 현재 상태, 행동
        state_action_vec = feature_vector(state, action, num_states, num_actions)  # 상태-행동 벡터 생성
        w += alpha * (G - VFA(state, action, w)) * state_action_vec  # 가중치 업데이트

print("학습된 가중치 벡터 w:")
print(w)


#### 4.backward SARSA ####

alpha = 0.1  # 학습률
gamma = 0.9  # 할인율
lmbda = 0.8  # Eligibility Trace 감쇠율

for episode_num in range(1, num_episodes + 1):
    #원래는
    E = np.zeros((num_states, num_actions)) 
    #Q = np.zeros((num_states, num_actions)) 
    w = np.zeros(num_states * num_actions)  
    epsilon = 1 / episode_num  # Exploration 감소
    episode = run_episode_next(epsilon)

    for state, action, reward, next_state, next_action in reversed(episode):
        d = reward + gamma * VFA(next_state, next_action, w) - VFA(state, action, w#d = reward + 0.9 * Q[next_state,next_action] - Q[state,action]
        
        E[state,action] = E[state,action] + 1 #일단 방문했으니까 +1 증가시킴
        #흠 여기까지는 그래도 어느정도 이해가 가는데,밑에는 잘 몰라서
        for s in range(num_states):
            for a in range(num_actions):
                w += alpha * d * E[s, a]  # Q 업데이트
                E[s, a] *= gamma * lmbda  #(시간이 지나면서 감소)
            """
            1. 기본 SARSA와의 비교 
            Q[state, action] += alpha * (reward + 0.9 * Q[next_state, next_action] - Q[state,action])  
            => 이게 기본 SARSA알고리즘
            => Q[state, action] += alpha * (d) 인 것임.  
            따라서, E[s,a]를 단순히 곱하면
            Q[state, action] += alpha * (d) * E[s,a]임 
            
            
            2. 왜 여기서 num_states와 num_actions에 대해 다 계산하니? 
            Q 값을 업데이트할 때는 현재 상태뿐만 아니라 과거에 방문했던 상태들도 보상의 영향을 받음 
            이걸 처리하기 위해 모든 상태-행동 쌍에 대해 Eligibility Trace를 곱해서 업데이트
            """
print(w)

#### 5. Q-Learning ####


def greedy_policy_val_weight(Q, state,action): #state가 주어질 때 best action을 선택해서 값을 찾는 greedy
    #return np.max(VFA(state,action,w)) #현실적으로 greedy가 안될 거 같다? 이부분이? 
    return np.max([VFA(state, action, w) for action in range(num_actions)])


    #ver2. 정리 버전 - 37p : Q(S,A) ←Q(S,A) + α R + γ maxa′Q(S′,a′)−Q(S,A)
for episode_num in range(1, num_episodes+1):
    epsilon = 1 / episode_num # episode를 점점 늘려갈 때마다 epsilon값이 줄어든다. 
    #behaviro policy ( 내가 관찰한 policy )
    episode = run_episode_next(epsilon) 
    
    visited = set()  # 첫 방문 판별용
    
    for state, action, reward, next_state, next_action in reversed(episode): 
        if (state, action) not in visited:  
            visited.add((state, action))  
            
            N[state, action] += 1  # 방문 횟수 증가
            alpha = 1 / N[state, action]  # 학습률 감소 (GLIE 조건)
            
            # target policy에서의 action 추출? 
            # 어떻게 target policy를 특정하지? 일단 greedy라고 나와있으니, greedy하게 episode를 뽑아보자. 
            #next_val = greedy_policy_val(Q, next_state)
            next_val =greedy_policy_val_weight(state,action,w) #이게 맞는건가?
            w += alpha * (reward + 0.9 * next_val - VFA(state, action, w)) * feature_vector(state, action, num_states, num_actions)
            
            #여기서 매우 중요한 건, 똑같이 저렇게 reversed(episode)를 했다고 하더라도, 저 위에서는 G를 썼지만, 여기서는 그때의 그 reward만을 써서 update를 한다는 것
print("w")
print(w)
'''
Q-learning에서는, 그 다음에 최적의 행동을 선택할 때, 
가장 그 이후의 최적의 행동이어야 하는데 과연 이걸 어떻게 weight로 표현할 수 있을까?

(이전에는 단순히 Q[state]의 max를 뽑았는데?)
=> 일단 state가 고정되어 있으니까, 너무 걱정하지 말고 그 state에 해당하는 action을 뽑아서 그 action에 대한 가중치를 뽑아보자.
'''
