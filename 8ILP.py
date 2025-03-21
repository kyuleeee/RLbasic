#This is for simple maze RL with Dyna-Q! 

import numpy as np
import matplotlib.pyplot as plt
import time

import argparse

# 미로 크기 설정 (7x7 미로)
maze = np.array([
    [0, 0, 0, 0, 0, 0, 0, 1, 0],  # 벽 (1)
    [0, 0, 1, 0, 0, 0, 0, 1, 0],  # 길 (0)
    [0, 0, 1, 0, 0, 0, 0, 1, 0],
    [0, 0, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0]
])

# 시작점과 목표점
start = (2,0)
goal = (0,8)

# 미로 출력 함수 (matplotlib을 사용한 미로 출력)
def plot_maze(maze, agent_position):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(maze, cmap='binary')  # 벽은 흰색, 길은 검정색
    ax.scatter(agent_position[1], agent_position[0], c='red', s=100, marker='*')  # 에이전트 위치

    ax.set_xticks(np.arange(-0.5, len(maze[0]), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(maze), 1), minor=True)
    ax.grid(which='minor', color='black', linestyle='-', linewidth=2)

    # 목표 위치 표시
    ax.scatter(goal[1], goal[0], c='green', s=100, marker='x')

    plt.draw()
    plt.pause(0.1)  # 0.1초 동안 화면을 멈춤 (애니메이션 효과)

# 에이전트 이동 함수
def move_agent(agent_position, direction):
    x, y = agent_position
    if direction == 'up' and x > 0 and maze[x-1, y] != 1:
        return (x-1, y)
    elif direction == 'down' and x < maze.shape[0] - 1 and maze[x+1, y] != 1:
        return (x+1, y)
    elif direction == 'left' and y > 0 and maze[x, y-1] != 1:
        return (x, y-1)
    elif direction == 'right' and y < maze.shape[1] - 1 and maze[x, y+1] != 1:
        return (x, y+1)
    return agent_position  # 이동 불가능할 경우 기존 위치 반환

# 게임 시작
current_position = start
steps = 0

print("Start Maze Game:")
plt.ion()  # 인터랙티브 모드 시작
plot_maze(maze, current_position)

# 에이전트가 목표에 도달할 때까지 반복
while current_position != goal:
    direction = input("Move (up/down/left/right): ")
    current_position = move_agent(current_position, direction)
    steps += 1
    print(f"Step {steps}:")
    plot_maze(maze, current_position)

print("You reached the goal!")
plt.ioff()  # 인터랙티브 모드 종료
plt.show()


## 1. SARSA ####
num_states = 54   # 상태 개수
num_actions = 4  # 행동 개수

# Q-테이블 초기화 (모든 Q(s,a)를 0으로 시작) 
# Q(s,a) ==  E[ Gt | St= s , At = a]
Q = np.zeros((num_states, num_actions)) 

# 각 상태에서 방문 횟수 저장
N = np.zeros((num_states, num_actions))

# ε-greedy policy
def epsilon_greedy_policy(Q, state, epsilon):
    m = len(Q[state])
    best_action = np.argmax(Q[state])  # Q 값이 가장 높은 행동 선택
    
    policy = np.full(m, epsilon / m)   # 모든 행동에 대해 ε/m 확률 부여
    policy[best_action] += 1 - epsilon # 가장 좋은 행동에는 1 - ε 추가 확률
    
    return np.random.choice(np.arange(m), p=policy)

def run_episode_next(epsilon, max_steps=10):
    episode = []  
    state = start
    action = epsilon_greedy_policy(Q, state, epsilon)  # 초기 행동 선택

    while True:
        next_state = move_agent(state,action)  # 임의의 환경 이동
        reward = np.random.randn()
        next_action = epsilon_greedy_policy(Q, next_state, epsilon) # 다음 행동 선택
        
        episode.append((state, action, reward, next_state, next_action)) 
        
        state, action = next_state, next_action  # SARSA는 (s, a, r, s', a') 방식으로 진행!

        if len(episode) >= max_steps:  # 최대 10번 이동 후 종료
            break
    return episode   

def greedy_policy_action(Q, state): #state만 주어질 때 best action을 찾는 greedy
    return np.argmax(Q[state])

for episode_num in range(1, num_episodes+1):
    epsilon = 1 / episode_num # episode를 점점 늘려갈 때마다 epsilon값이 줄어든다. 
    episode = run_episode_next(epsilon)  
    
    visited = set()  # 첫 방문 판별용
    
    for state, action, reward, next_state, next_action in reversed(episode):
        #현실세계의 에피소드 인거지. 
        if (state, action) not in visited:  
            visited.add((state, action))  
            
            N[state, action] += 1  # 방문 횟수 증가
            alpha = 1 / N[state, action]  # 학습률 감소 (GLIE 조건)
            
            Q[state, action] += alpha * (reward + 0.9 * Q[next_state, next_action] - Q[state, action])
            # target policy에서의 action 추출? 
            # 어떻게 target policy를 특정하지? 일단 greedy라고 나와있으니, greedy하게 episode를 뽑아보자. 
            next_val = greedy_policy_val(Q, next_state)  # 최대 Q 값 사용
            
            
            Q[state, action] += alpha * (reward + 0.9 * next_val - Q[state, action])
            
            
            #여기서 매우 중요한 건, 똑같이 저렇게 reversed(episode)를 했다고 하더라도, 저 위에서는 G를 썼지만, 여기서는 그때의 그 reward만을 써서 update를 한다는 것
print("학습된 Q 테이블:")
print(Q)


def main():
    parser = argparse.ArgumentParser(description="Simple maze with Dyna-Q")
    parser.add_argument("--planning_step", type=int, default=5, help="Number of states")
    parser.add_argument("--num_episodes", type=int, default=5000, help="Number of training episodes")

    args = parser.parse_args()

    if args.planning_step == 0 :
        Q = SARSA(args.num_episode)
    else :  
        Q = DynaQ(args.planning_step,args.num_episodes)
    
    print("\nFinal Q-table:")
    print(Q)

if __name__ == "__main__":
    main()