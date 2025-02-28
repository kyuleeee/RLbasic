import numpy as np
import argparse

def epsilon_greedy_policy(Q, state, epsilon):
    """ ε-greedy 정책을 적용하여 행동을 선택 """
    m = len(Q[state])
    best_action = np.argmax(Q[state])  
    policy = np.full(m, epsilon / m)  
    policy[best_action] += 1 - epsilon 
    return np.random.choice(np.arange(m), p=policy)

def env_step(num_states):
    """ 환경에서 다음 상태와 보상을 반환 (임의의 환경 가정) """
    next_state = np.random.randint(0, num_states)
    reward = np.random.randn()
    return next_state, reward

def run_episode(Q, epsilon, num_states, num_actions, max_steps=10):
    """ GLIE Monte Carlo Control을 위한 에피소드 실행 """
    episode = []
    state = np.random.randint(0, num_states)  

    while len(episode) < max_steps:
        action = epsilon_greedy_policy(Q, state, epsilon)
        next_state, reward = env_step(num_states)
        episode.append((state, action, reward))
        state = next_state
    return episode

def monte_carlo_control(num_states, num_actions, num_episodes):
    """ GLIE Monte Carlo Control 학습 수행 """
    Q = np.zeros((num_states, num_actions)) 
    N = np.zeros((num_states, num_actions))  

    for episode_num in range(1, num_episodes+1):
        epsilon = 1 / episode_num
        episode = run_episode(Q, epsilon, num_states, num_actions)

        G = 0  
        visited = set()  

        for state, action, reward in reversed(episode):
            G = reward + 0.9 * G  
            if (state, action) not in visited:
                visited.add((state, action))
                N[state, action] += 1
                alpha = 1 / N[state, action]
                Q[state, action] += alpha * (G - Q[state, action])  
    return Q

def run_episode_sarsa(Q, epsilon, num_states, num_actions, max_steps=10):
    """ SARSA 학습을 위한 에피소드 실행 """
    episode = []
    state = np.random.randint(0, num_states)
    action = epsilon_greedy_policy(Q, state, epsilon)

    while len(episode) < max_steps:
        next_state, reward = env_step(num_states)
        next_action = epsilon_greedy_policy(Q, next_state, epsilon)
        episode.append((state, action, reward, next_state, next_action))
        state, action = next_state, next_action  
    return episode  

def sarsa(num_states, num_actions, num_episodes):
    """ SARSA 학습 수행 """
    Q = np.zeros((num_states, num_actions))
    N = np.zeros((num_states, num_actions))

    for episode_num in range(1, num_episodes+1):
        epsilon = 1 / episode_num
        episode = run_episode_sarsa(Q, epsilon, num_states, num_actions)

        visited = set()  
        for state, action, reward, next_state, next_action in reversed(episode):
            if (state, action) not in visited:
                visited.add((state, action))
                N[state, action] += 1
                alpha = 1 / N[state, action]
                Q[state, action] += alpha * (reward + 0.9 * Q[next_state, next_action] - Q[state, action])
    return Q

def n_step_sarsa(num_states, num_actions, num_episodes, n_step):
    """ n-step SARSA 학습 수행 """
    Q = np.zeros((num_states, num_actions))
    alpha = 0.1  

    for episode_num in range(1, num_episodes+1):
        epsilon = 1 / episode_num
        episode = run_episode_sarsa(Q, epsilon, num_states, num_actions)

        for t in range(len(episode)):
            G = 0  
            for i in range(n_step):
                if t + i < len(episode):
                    G += (0.9 ** i) * episode[t + i][2]  
            if t + n_step < len(episode):
                next_state = episode[t + n_step][3]
                next_action = episode[t + n_step][4]
                G += (0.9 ** n_step) * Q[next_state, next_action]
            state, action = episode[t][:2]
            Q[state, action] += alpha * (G - Q[state, action])
    return Q

def main():
    parser = argparse.ArgumentParser(description="Model-Free Control in Reinforcement Learning")
    parser.add_argument("--num_states", type=int, default=5, help="Number of states")
    parser.add_argument("--num_actions", type=int, default=2, help="Number of actions")
    parser.add_argument("--num_episodes", type=int, default=5000, help="Number of training episodes")
    parser.add_argument("--n_step", type=int, default=3, help="Number of steps for n-step SARSA")
    parser.add_argument("--method", type=str, choices=["mc", "sarsa", "nstep"], default="mc", help="Training method: mc (Monte Carlo), sarsa, nstep (n-step SARSA)")

    args = parser.parse_args()

    if args.method == "mc":
        Q = monte_carlo_control(args.num_states, args.num_actions, args.num_episodes)
    elif args.method == "sarsa":
        Q = sarsa(args.num_states, args.num_actions, args.num_episodes)
    elif args.method == "nstep":
        Q = n_step_sarsa(args.num_states, args.num_actions, args.num_episodes, args.n_step)
    
    print("\nFinal Q-table:")
    print(Q)

if __name__ == "__main__":
    main()
