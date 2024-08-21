import tkinter as tk
from tkinter import ttk
import gymnasium as gym
from stable_baselines3 import DQN
import numpy as np

def run_dqn():
    gamma = float(gamma_entry.get())
    learning_rate = float(learning_rate_entry.get())
    total_timesteps = int(total_timesteps_entry.get())

    environment = gym.make("Taxi-v3")

    # Initialize model with user-specified parameters
    model = DQN(
        "MlpPolicy",
        environment,
        gamma=gamma,
        learning_rate=learning_rate,
        learning_starts=500,
        target_update_interval=100,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.1,
        batch_size=32,
        train_freq=1,
        gradient_steps=1,
        exploration_fraction=0.5,
        verbose=1
    )

    # Train the model
    model.learn(total_timesteps=total_timesteps, log_interval=4)

    # Testing Phase
    total_epochs, total_penalties, total_rewards = 0, 0, 0
    episodes = 5  # Number of episodes for testing
    counter = 0

    for ep in range(episodes):
        if counter > 100:
            result_label.config(text="System time out")
            break
        state = environment.reset()[0]
        epochs, penalties, episode_reward = 0, 0, 0
        done = False

        while not done:
            action, _states = model.predict(state, deterministic=True)
            action = int(action)  # Ensure action is a scalar
            state, reward, done, info, _ = environment.step(action)

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

    result_label.config(text=f"Training completed.\n"
                             f"Results after {episodes} episodes:\n"
                             f"Average timesteps per episode: {avg_timesteps:.2f}\n"
                             f"Average penalties per episode: {avg_penalties:.2f}\n"
                             f"Average reward per episode: {avg_reward:.2f}")

    environment.close()

# Create the main window
root = tk.Tk()
root.title("Taxi-v3 DQN")

# Create labels and entry widgets for parameters
learning_rate_label = tk.Label(root, text="Learning Rate (α):")
learning_rate_label.grid(row=1, column=0, padx=10, pady=10)
learning_rate_entry = tk.Entry(root)
learning_rate_entry.grid(row=1, column=1, padx=10, pady=10)
learning_rate_entry.insert(0, "1e-3")

gamma_label = tk.Label(root, text="Gamma (γ):")
gamma_label.grid(row=0, column=0, padx=10, pady=10)
gamma_entry = tk.Entry(root)
gamma_entry.grid(row=0, column=1, padx=10, pady=10)
gamma_entry.insert(0, "0.9")

total_timesteps_label = tk.Label(root, text="Total Timesteps:")
total_timesteps_label.grid(row=2, column=0, padx=10, pady=10)
total_timesteps_entry = tk.Entry(root)
total_timesteps_entry.grid(row=2, column=1, padx=10, pady=10)
total_timesteps_entry.insert(0, "80000")


# Run button to start the training and testing
run_button = tk.Button(root, text="Run DQN", command=run_dqn)
run_button.grid(row=4, columnspan=2, pady=20)

# Label to display the result
result_label = tk.Label(root, text="")
result_label.grid(row=5, columnspan=2, pady=10)

# Run the application
root.mainloop()
