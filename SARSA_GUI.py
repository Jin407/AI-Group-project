import tkinter as tk
import gym
import numpy as np
import random
import time

def run_SARSA():
    alpha = float(learning_rate_entry.get())
    gamma = float(discount_factor_entry.get())
    num_episodes = int(num_episodes_entry.get())

    epsilon = 1.0  # Exploration factor (start with high exploration)
    epsilon_decay = 0.995  # Decay factor for epsilon
    epsilon_min = 0.01  # Minimum epsilon value

    # Initialize the Taxi-v3 environment
    env = gym.make('Taxi-v3').env
    q_table = np.zeros([env.observation_space.n, env.action_space.n])

    all_rewards = []
    average_rewards = []

    # Training Phase
    # SARSA algorithm
    for i in range(num_episodes):
        state = env.reset()[0]  # Reset the environment for a new episode
        done = False
        total_reward = 0  # Initialize total reward for this episode

        # Initial action selection based on epsilon-greedy policy
        action = env.action_space.sample() if random.uniform(0, 1) < epsilon else np.argmax(q_table[state])

        while not done:
            # Take the action and observe the outcome
            next_state, reward, done, info, _ = env.step(action)
            total_reward += reward  # Accumulate reward

            # Select next action based on epsilon-greedy policy
            next_action = env.action_space.sample() if random.uniform(0, 1) < epsilon else np.argmax(q_table[next_state])

            # Update Q-value for the current state-action pair using SARSA update rule
            old_value = q_table[state, action]
            next_value = q_table[next_state, next_action]
            q_table[state, action] = old_value + alpha * (reward + gamma * next_value - old_value)

            # Transition to the next state and action
            state, action = next_state, next_action

        # Decay epsilon
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        # Store total reward and calculate average reward
        all_rewards.append(total_reward)
        average_reward = np.mean(all_rewards)
        average_rewards.append(average_reward)

    # Testing Phase
    total_epochs, total_penalties, total_rewards = 0, 0, 0
    episodes = 5  # Number of episodes for testing
    counter = 0

    for ep in range(episodes):
        if counter > 100:
            result_label.config(text="System time out")
            break
        state = env.reset()[0]
        epochs, penalties, episode_reward = 0, 0, 0
        done = False

        while not done:
            action = np.argmax(q_table[state])
            state, reward, done, info, _ = env.step(action)

            if reward == -10:
                penalties += 1

            episode_reward += reward
            epochs += 1
            counter += 1
            if counter > 100:
                break

        total_penalties += penalties
        total_epochs += epochs
        total_rewards += episode_reward

    avg_timesteps = total_epochs / episodes
    avg_penalties = total_penalties / episodes
    avg_reward = total_rewards / episodes

    result_label.config(text=f"Results after {episodes} episodes:\n"
                             f"Average timesteps per episode: {avg_timesteps:.2f}\n"
                             f"Average penalties per episode: {avg_penalties:.2f}\n"
                             f"Average reward per episode: {avg_reward:.2f}")

    env.close()

# Create the main window
root = tk.Tk()
root.title("Taxi-v3 SARSA")

# Create labels and entry widgets for parameters
learning_rate_label = tk.Label(root, text="Learning Rate (α):")
learning_rate_label.grid(row=0, column=0, padx=10, pady=10)
learning_rate_entry = tk.Entry(root)
learning_rate_entry.grid(row=0, column=1, padx=10, pady=10)
learning_rate_entry.insert(0, "0.2")

discount_factor_label = tk.Label(root, text="Discount Factor (γ):")
discount_factor_label.grid(row=1, column=0, padx=10, pady=10)
discount_factor_entry = tk.Entry(root)
discount_factor_entry.grid(row=1, column=1, padx=10, pady=10)
discount_factor_entry.insert(0, "0.9")

num_episodes_label = tk.Label(root, text="Number of Episodes:")
num_episodes_label.grid(row=2, column=0, padx=10, pady=10)
num_episodes_entry = tk.Entry(root)
num_episodes_entry.grid(row=2, column=1, padx=10, pady=10)
num_episodes_entry.insert(0, "2000")

# Run button to start the training and testing
run_button = tk.Button(root, text="Run SARSA", command=run_SARSA)
run_button.grid(row=3, columnspan=2, pady=20)

# Label to display the result
result_label = tk.Label(root, text="")
result_label.grid(row=4, columnspan=2, pady=10)

# Run the application
root.mainloop()