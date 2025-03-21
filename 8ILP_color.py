import numpy as np
import matplotlib.pyplot as plt
import time
import argparse

# 미로 크기 설정 (6x9 미로)
maze = np.array([
    [0, 0, 0, 0, 0, 0, 0, 1, 0],  # 벽 (1)
    [0, 0, 1, 0, 0, 0, 0, 1, 0],  # 길 (0)
    [0, 0, 1, 0, 0, 0, 0, 1, 0],
    [0, 0, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0]
])

start = (2, 0)
goal = (0, 8)

actions = ['up', 'down', 'left', 'right']
action_map = {0: 'up', 1: 'down', 2: 'left', 3: 'right'}

# 미로 출력 함수
def plot_maze(maze, agent_position, ax=None, agent_plot=None, title=None, is_planning=False):
    if ax is None or agent_plot is None:
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(maze, cmap='binary')
        ax.scatter(goal[1], goal[0], c='green', s=100, marker='x')
        agent_color = 'blue' if is_planning else 'red'
        agent_plot = ax.scatter(agent_position[1], agent_position[0], c=agent_color, s=100, marker='*')
        ax.set_xticks(np.arange(-0.5, maze.shape[1], 1), minor=True)
        ax.set_yticks(np.arange(-0.5, maze.shape[0], 1), minor=True)
        ax.grid(which='minor', color='black', linestyle='-', linewidth=2)
        if title:
            ax.set_title(title, fontsize=14)
        plt.ion()
        plt.show()
        return ax, agent_plot
    
    # 에이전트 컬러 업데이트
    agent_color = 'blue' if is_planning else 'red'
    agent_plot.set_color(agent_color)
    agent_plot.set_offsets([agent_position[1], agent_position[0]])
    
    # 타이틀 업데이트
    if title:
        ax.set_title(title, fontsize=14)
    
    plt.draw()
    plt.pause(0.01)
    return ax, agent_plot

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
    return agent_position

# 상태 인덱스 변환
def state_index(state):
    return state[0] * maze.shape[1] + state[1]

# Q-테이블 및 모델 초기화
num_states = maze.shape[0] * maze.shape[1]
num_actions = len(actions)
Q = np.zeros((num_states, num_actions))
Model = {}

# ε-greedy 정책
def epsilon_greedy_policy(Q, state, epsilon):
    if np.random.rand() < epsilon:
        return np.random.choice(num_actions)
    return np.argmax(Q[state])

# Dyna-Q 알고리즘
def DynaQ(planning_step, num_episodes, alpha=0.1, gamma=0.9, epsilon=0.1):
    global Q, Model
    
    ax, agent_plot = plot_maze(maze, start, title="Maze")
    
    for episode in range(num_episodes):
        state = start
        state_idx = state_index(state)
        num_step = 0
        
        while state != goal:
            action = epsilon_greedy_policy(Q, state_idx, epsilon)
            next_state = move_agent(state, action_map[action])
            next_state_idx = state_index(next_state)
            reward = 1 if next_state == goal else -0.01
            
            # Q 업데이트
            Q[state_idx, action] += alpha * (reward + gamma * np.max(Q[next_state_idx]) - Q[state_idx, action])
            
            # 환경 모델 업데이트
            Model[(state_idx, action)] = (next_state_idx, reward)
            
            # 실제 에이전트 이동 (빨간색으로 표시)
            state = next_state
            state_idx = next_state_idx
            title = f"episode {episode + 1}, step {num_step+1}"
            ax, agent_plot = plot_maze(maze, state, ax, agent_plot, title, is_planning=False)
            num_step += 1
            
            # Planning 단계 (파란색으로 표시)
            if len(Model) > 0:
                for p in range(planning_step):
                    sampled_state_idx, sampled_action = list(Model.keys())[np.random.randint(len(Model))]
                    sampled_next_state_idx, sampled_reward = Model[(sampled_state_idx, sampled_action)]
                    
                    # Q-값 업데이트
                    Q[sampled_state_idx, sampled_action] += alpha * (sampled_reward + gamma * np.max(Q[sampled_next_state_idx]) - Q[sampled_state_idx, sampled_action])
                    
                    # 기획 단계 시각화 (옵션)
                    sampled_state_row = sampled_state_idx // maze.shape[1]
                    sampled_state_col = sampled_state_idx % maze.shape[1]
                    planning_title = f"episode {episode + 1}, step {num_step}, planning {p+1}/{planning_step}"
                    ax, agent_plot = plot_maze(maze, (sampled_state_row, sampled_state_col), ax, agent_plot, planning_title, is_planning=True)
            
            # 목표에 도달했는지 확인
            if state == goal:
                title = f"episode {episode + 1} complete! ({num_step} step)"
                ax, agent_plot = plot_maze(maze, state, ax, agent_plot, title, is_planning=False)
                time.sleep(1)  # 목표 도달 시 잠시 보여줌
        
        print(f"episode {episode + 1} complete: {num_step} step!")
    
    return Q

# 메인 함수
def main():
    parser = argparse.ArgumentParser(description="Maze for Dyna-Q")
    parser.add_argument("--planning_step", type=int, default=5, help="planning_step")
    parser.add_argument("--num_episodes", type=int, default=10, help="num_episode")
    args = parser.parse_args()
    
    global Q, Model
    Q = np.zeros((num_states, num_actions))  # Q-테이블 초기화
    Model = {}  # 환경 모델 초기화
    
    final_Q = DynaQ(args.planning_step, args.num_episodes)
    
    print("\n최종 Q-테이블:")
    print(final_Q)

if __name__ == "__main__":
    main()