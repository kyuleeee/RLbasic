import numpy as np
import matplotlib.pyplot as plt
import time
import argparse
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.patches import Rectangle, Circle

# Maze size configuration (6x9 maze)
maze = np.array([
    [0, 0, 0, 0, 0, 0, 0, 1, 0],  # Wall (1)
    [0, 0, 1, 0, 0, 0, 0, 1, 0],  # Path (0)
    [0, 0, 1, 0, 0, 0, 0, 1, 0],
    [0, 0, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0]
])

start = (2, 0)
goal = (0, 8)

actions = ['up', 'down', 'left', 'right']
action_map = {0: 'up', 1: 'down', 2: 'left', 3: 'right'}

# Cute maze display function
def plot_maze(maze, agent_position, ax=None, agent_plot=None, title=None, is_planning=False):
    if ax is None or agent_plot is None:
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Setting background color
        ax.set_facecolor('#F0F8FF')  # Sky blue background
        
        # Drawing maze - pretty colors for walls and paths
        for i in range(maze.shape[0]):
            for j in range(maze.shape[1]):
                if maze[i, j] == 1:  # Wall
                    wall = Rectangle((j-0.5, i-0.5), 1, 1, color='#8B4513', alpha=0.8)  # Brown walls
                    ax.add_patch(wall)
                else:  # Path
                    path = Rectangle((j-0.5, i-0.5), 1, 1, color='#E0FFFF', alpha=0.5)  # Light sky blue paths
                    ax.add_patch(path)
        
        # Drawing grid
        ax.set_xticks(np.arange(-0.5, maze.shape[1], 1), minor=True)
        ax.set_yticks(np.arange(-0.5, maze.shape[0], 1), minor=True)
        ax.grid(which='minor', color='#D3D3D3', linestyle='-', linewidth=1)
        
        # Goal point (displayed as trophy)
        treasure = Circle((goal[1], goal[0]), 0.3, color='#FFD700', alpha=0.8)  # Gold circle
        ax.add_patch(treasure)
        ax.text(goal[1], goal[0], 'G', fontsize=15, ha='center', va='center', fontweight='bold')
        
        # Agent display
        if is_planning:
            agent_color = '#1E90FF'  # Deep blue
            agent_marker = '?' if is_planning else 'A'  # Question mark vs Agent
        else:
            agent_color = '#FF6347'  # Tomato color
            agent_marker = 'A'  # Agent
            
        agent_circle = Circle((agent_position[1], agent_position[0]), 0.3, color=agent_color, alpha=0.7)
        ax.add_patch(agent_circle)
        agent_plot = ax.text(agent_position[1], agent_position[0], agent_marker, 
                             fontsize=15, ha='center', va='center', fontweight='bold')
        
        # Start point display
        start_point = Circle((start[1], start[0]), 0.3, color='#90EE90', alpha=0.6)  # Light green
        ax.add_patch(start_point)
        ax.text(start[1], start[0], 'S', fontsize=15, ha='center', va='center', fontweight='bold')
        
        # Setting axis range and removing ticks
        ax.set_xlim(-0.5, maze.shape[1] - 0.5)
        ax.set_ylim(maze.shape[0] - 0.5, -0.5)
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Setting title
        if title:
            ax.set_title(title, fontsize=16, color='#4B0082', fontweight='bold')
        else:
            ax.set_title("Maze Exploration Adventure", fontsize=16, color='#4B0082', fontweight='bold')
        
        plt.tight_layout()
        plt.ion()
        plt.show()
        return ax, agent_plot
    
    # Update agent position
    agent_plot.set_position((agent_position[1], agent_position[0]))
    
    # Change agent expression (based on planning state)
    if is_planning:
        agent_plot.set_text('?')  # Thinking
    else:
        agent_plot.set_text('A')  # Agent
    
    # Update title
    if title:
        ax.set_title(title, fontsize=16, color='#4B0082', fontweight='bold')
    
    plt.draw()
    plt.pause(0.05)  # Slightly slower to see movement
    
    return ax, agent_plot

# Agent movement function
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

# State index conversion
def state_index(state):
    return state[0] * maze.shape[1] + state[1]

# Q-table and model initialization
num_states = maze.shape[0] * maze.shape[1]
num_actions = len(actions)
Q = np.zeros((num_states, num_actions))
Model = {}

# ε-greedy policy
def epsilon_greedy_policy(Q, state, epsilon):
    if np.random.rand() < epsilon:
        return np.random.choice(num_actions)
    return np.argmax(Q[state])

# Message generation function
def get_message(episode, step, is_planning=False, is_complete=False):
    if is_complete:
        messages = [
            f"Success! Episode {episode + 1} completed! (Total: {step} steps)",
            f"Great! Episode {episode + 1} successful! (Total: {step} steps)",
            f"Episode {episode + 1} cleared! (Total: {step} steps)"
        ]
    elif is_planning:
        messages = [
            f"Thinking... Episode {episode + 1}, Step {step}",
            f"Planning... Episode {episode + 1}, Step {step}",
            f"Predicting future... Episode {episode + 1}, Step {step}"
        ]
    else:
        messages = [
            f"Exploring! Episode {episode + 1}, Step {step}",
            f"Moving... Episode {episode + 1}, Step {step}",
            f"Searching... Episode {episode + 1}, Step {step}"
        ]
    
    return np.random.choice(messages)

# Dyna-Q algorithm
def DynaQ(planning_step, num_episodes, alpha=0.1, gamma=0.9, epsilon=0.1):
    global Q, Model
    
    ax, agent_plot = plot_maze(maze, start, title="The Agent's Maze Adventure Begins!")
    time.sleep(1)  # Show starting screen briefly
    
    for episode in range(num_episodes):
        state = start
        state_idx = state_index(state)
        num_step = 1
        
        while state != goal:
            # Select action and move
            action = epsilon_greedy_policy(Q, state_idx, epsilon)
            next_state = move_agent(state, action_map[action])
            next_state_idx = state_index(next_state)
            reward = 1 if next_state == goal else -0.01
            
            # Update Q
            Q[state_idx, action] += alpha * (reward + gamma * np.max(Q[next_state_idx]) - Q[state_idx, action])
            
            # Update environment model
            Model[(state_idx, action)] = (next_state_idx, reward)
            
            # Actual agent movement
            state = next_state
            state_idx = next_state_idx
            title = get_message(episode, num_step)
            ax, agent_plot = plot_maze(maze, state, ax, agent_plot, title, is_planning=False)
            
            # Check goal reached
            if state == goal:
                complete_title = get_message(episode, num_step, is_complete=True)
                ax, agent_plot = plot_maze(maze, state, ax, agent_plot, complete_title)
                time.sleep(1.5)  # Show a bit longer when goal is reached
                break
            
            # Planning step (thinking agent)
            if len(Model) > 0 and planning_step > 0:
                # Randomly visualize only some planning (for speed improvement)
                if np.random.rand() < 0.3:  # 30% chance to visualize planning
                    sampled_state_idx, sampled_action = list(Model.keys())[np.random.randint(len(Model))]
                    sampled_next_state_idx, sampled_reward = Model[(sampled_state_idx, sampled_action)]
                    
                    # Calculate state coordinates for visualization
                    sampled_state_row = sampled_state_idx // maze.shape[1]
                    sampled_state_col = sampled_state_idx % maze.shape[1]
                    
                    # Visualize planning
                    planning_title = get_message(episode, num_step, is_planning=True)
                    ax, agent_plot = plot_maze(maze, (sampled_state_row, sampled_state_col), 
                                              ax, agent_plot, planning_title, is_planning=True)
                
                # Run all planning steps (visualize only some)
                for _ in range(planning_step):
                    sampled_state_idx, sampled_action = list(Model.keys())[np.random.randint(len(Model))]
                    sampled_next_state_idx, sampled_reward = Model[(sampled_state_idx, sampled_action)]
                    Q[sampled_state_idx, sampled_action] += alpha * (sampled_reward + gamma * np.max(Q[sampled_next_state_idx]) - Q[sampled_state_idx, sampled_action])
            
            num_step += 1
        
        print(f"Episode {episode + 1} completed: {num_step} steps!")
    
    # Learning complete message
    ax.set_title("Learning complete! The agent has mastered the maze!", fontsize=16, color='#4B0082', fontweight='bold')
    plt.draw()
    plt.pause(2)
    
    return Q

# Main function
def main():
    parser = argparse.ArgumentParser(description="Agent's Dyna-Q Maze Exploration")
    parser.add_argument("--planning_step", type=int, default=5, help="Number of planning steps")
    parser.add_argument("--num_episodes", type=int, default=10, help="Number of learning episodes")
    args = parser.parse_args()
    
    global Q, Model
    Q = np.zeros((num_states, num_actions))  # Initialize Q-table
    Model = {}  # Initialize environment model
    
    final_Q = DynaQ(args.planning_step, args.num_episodes)
    
    print("\nFinal Q-table:")
    print(final_Q)

if __name__ == "__main__":
    main()