"""
THis is a code for Model Free Control in RL. 
"""

#### 1. GLIE Monte Carlo Control #### 
#==============================================================================
import numpy as np

num_states = 5    # 상태 개수
num_actions = 2   # 행동 개수
num_episodes = 5000  

# Q-테이블 초기화 (모든 Q(s,a)를 0으로 시작) 
# Q(s,a) ==  E[ Gt | St= s , At = a]
Q = np.zeros((num_states, num_actions)) 

# 각 상태에서 방문 횟수 저장
N = np.zeros((num_states, num_actions))

# ε-greedy policy [만약, 저 epsilon이 고정이라면 그냥 exploration만 계속하게 됨]
def epsilon_greedy_policy(Q, state, epsilon):
    m = len(Q[state])
    best_action = np.argmax(Q[state])  # Q 값이 가장 높은 행동 선택
    
    policy = np.full(m, epsilon / m)   # 모든 행동에 대해 ε/m 확률 부여
    policy[best_action] += 1 - epsilon # 가장 좋은 행동에는 1 - ε 추가 확률
    
    return np.random.choice(np.arange(m), p=policy)

# 한 번의 에피소드를 실행하면서 데이터를 모으는 코드 (Monte Carlo)
def run_episode(epsilon, max_steps=10):
    episode = []  
    state = np.random.randint(0, num_states)  # 랜덤한 초기 상태

    while True:
        action = epsilon_greedy_policy(Q, state, epsilon) #epsilon_greedy_policy로 탐색과정! 
        next_state = np.random.randint(0, num_states)  #
        reward = np.random.randn()
        
        episode.append((state, action, reward)) 
        state = next_state

        if len(episode) >= max_steps:  # 최대 10번 이동 후 종료
            break
    return episode

    
  
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
            N[state, action] += 1  # 특정 state, 특정 action에 대해 방문 횟수 증가
            alpha = 1 / N[state, action]  # GLIE 조건: α → 0 (학습율 감소)
            Q[state, action] += alpha * (G - Q[state, action])  # Q(s,a) 업데이트
            
            
print("학습된 Q 테이블:")
print(Q)


#==============================================================================

#### 2.SARSA ####
def run_episode_next(epsilon, max_steps=10):
    episode = []  
    state = np.random.randint(0, num_states)  # 랜덤한 초기 상태
    action = epsilon_greedy_policy(Q, state, epsilon)  # 초기 행동 선택

    while True:
        next_state = np.random.randint(0, num_states)  # 임의의 환경 이동
        reward = np.random.randn()
        next_action = epsilon_greedy_policy(Q, next_state, epsilon) # 다음 행동 선택
        
        episode.append((state, action, reward, next_state, next_action)) 
        
        state, action = next_state, next_action  # SARSA는 (s, a, r, s', a') 방식으로 진행!

        if len(episode) >= max_steps:  # 최대 10번 이동 후 종료
            break
    return episode   


for episode_num in range(1, num_episodes+1):
    epsilon = 1 / episode_num # episode를 점점 늘려갈 때마다 epsilon값이 줄어든다. 
    episode = run_episode_next(epsilon)  
    
    visited = set()  # 첫 방문 판별용
    
    for state, action, reward, next_state, next_action in reversed(episode):
        if (state, action) not in visited:  
            visited.add((state, action))  
            
            N[state, action] += 1  # 방문 횟수 증가
            alpha = 1 / N[state, action]  # 학습률 감소 (GLIE 조건)
            
            Q[state, action] += alpha * (reward + 0.9 * Q[next_state, next_action] - Q[state, action])
            #여기서 매우 중요한 건, 똑같이 저렇게 reversed(episode)를 했다고 하더라도, 저 위에서는 G를 썼지만, 여기서는 그때의 그 reward만을 써서 update를 한다는 것
print("학습된 Q 테이블:")
print(Q)


#### 3.n-step SARSA ####
n_step = 3  
alpha = 0.1
for episode_num in range(1, num_episodes + 1):
    epsilon = 1 / episode_num  # Exploration 감소
    episode = run_episode_next(epsilon)  

    for t in range(len(episode)):  # 각 time step에 대해 반복
        G = 0  # n-step return 초기화

        # 1. n-step 동안 보상 계산
        for i in range(n_step): #3번 step이라고 하면 3번 반복해야함.
            if t + i < len(episode):
                G += (0.9 ** i) * episode[t + i][2]  # 이건 보상!!!! 

        # 2. n-step 후의 Q 값 추가
        if t + n_step < len(episode):
            G += (0.9 ** n_step) * Q[episode[t + n_step][3], episode[t + n_step][4]]

        # 3. Q 업데이트
        state, action = episode[t][:2]  # 현재 상태, 행동
        Q[state, action] += alpha * (G - Q[state, action])  
print("학습된 Q 테이블:")
print(Q)



#### 4.backward SARSA ####

alpha = 0.1  # 학습률
gamma = 0.9  # 할인율
lmbda = 0.8  # Eligibility Trace 감쇠율

for episode_num in range(1, num_episodes + 1):
    E = np.zeros((num_states, num_actions))
    Q = np.zeros((num_states, num_actions))
    epsilon = 1 / episode_num  # Exploration 감소
    episode = run_episode_next(epsilon)

    for state, action, reward, next_state, next_action in reversed(episode):
        d = reward + 0.9 * Q[next_state,next_action] - Q[state,action]
        
        E[state,action] = E[state,action] + 1 #일단 방문했으니까 +1 증가시킴
        #흠 여기까지는 그래도 어느정도 이해가 가는데,밑에는 잘 몰라서
        for s in range(num_states):
            for a in range(num_actions):
                Q[s, a] += alpha * d * E[s, a]  # Q 업데이트
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

#### 5. Q-Learning ####

def greedy_policy_action(Q, state): #state만 주어질 때 best action을 찾는 greedy
    return np.argmax(Q[state])

    #ver1. 정석 버전 - 35p : Q(St ,At ) ←Q(St ,At ) + α Rt+1 + γQ(St+1,A′)−Q(St ,At ) 이 식 그대로 
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
            next_action_target = greedy_policy_action(Q, next_state)  # 목표 정책의 행동 선택
            
            Q[state, action] += alpha * (reward + 0.9 * Q[next_state, next_action_target] - Q[state, action])
    
print("학습된 Q 테이블:")
print(Q)



def greedy_policy_val(Q, state): #state가 주어질 때 best action을 선택해서 값을 찾는 greedy
    return np.max(Q[state])

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
            next_val = greedy_policy_val(Q, next_state)  # 최대 Q 값 사용
            
            Q[state, action] += alpha * (reward + 0.9 * next_val - Q[state, action])
            #여기서 매우 중요한 건, 똑같이 저렇게 reversed(episode)를 했다고 하더라도, 저 위에서는 G를 썼지만, 여기서는 그때의 그 reward만을 써서 update를 한다는 것
print("학습된 Q 테이블:")
print(Q)

'''
- SARSA vs Q-Learning
    - next_state, next_action은 어디서?
        - next_state : 둘다 기존의 policy(behavior policy)에서 뽑는다.
        - next_action :
            - SARSA : from behavior policy *(run_episode_next 함수 활용)*
            - Q-Learning : from target policy *(greedy_policy_action 함수 or greedy_policy_val)*
                
                ⇒ 그렇다면, 언제 greedy_policy_action을? 언제 greedy_policy_val을?
                
                greedy_policy_action : 
                
                최적의 행동(next action)을 직접 선택
                
                `argmax(Q[state])`를 사용해 가장 높은 Q 값을 가진 행동 선택
                
                greedy_policy_val : 
                
                특정 상태에서 최대 Q 값을 가져오기 위해 사용
                
                `max(Q[state])`를 사용해 가능한 행동 중 최대 Q 값 반환
- epsilon_greedy_policy vs greedy_policy
    - e_greedy : 탐색(exploration)까지 고려해야 할 때
        → 그래서 저기에 np.random.choice를 넣어두는 것임.
    - - greedy_policy : 가장 좋은 행동, 좋은 값만 선택할 때
        → 그래서 저기에 np.random.choice가 없는 것
'''