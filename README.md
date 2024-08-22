# AI-Group-project
This repository is for the group project we did in CSCI 323. In this project, by referring to several guides, we implemented 3 reinforcement learning algorithm Q-learning, SARSA and DQN to attempt to train an agent to navigate OpenAI's Taxi-V3 environment. There are four files which includes a jupyther notebook for our code execution and 3 python programme which is the GUI to play with the 3 reinforcement learning algorithm we implemented.

In the Jupyther notebook, you will find the training rate and performance of all 3 different reinforcement learning algorithm.

# Prerequisites

Before running the program, ensure you have the following installed on your machine:

Python 3
Git

# To clone the repository, open terminal or cmd and run the following command:

git clone https://github.com/Jin407/AI-Group-project

Alternatively, you can download the files directly from GitHub.

# To run the programme, you will need to pip install the dependecies:

pip install -r requirements.txt

To run Q-learning algorithm GUI:

python3 Q-learning_GUI.py

To run DQN algorithm GUI:

python3 DQN_GUI.py

To run SARSA_GUI.py:

python3 SARSA_GUI.py

# WARNING: DQN algorithm takes a while to run as the number of timesteps required to train our DQN agent is very large.

In conlusion to run our programme, simply follow these commands on terminal or cmd:

1. git clone https://github.com/Jin407/AI-Group-project

2. cd AI-Group-project

3. pip install -r requirements.txt

4. python3 Q-learning_GUI.py (To run Q-learning GUI)

5. python3 DQN_GUI.py (To run DQN GUI)

6. python3 SARSA_GUI.py (To run SARSA GUI)

Alternatively, you can download the programmes directly from Github and run them on your preferred IDE, the dependecies used in these programme include gym, gymnasium and stable_baselines3 so kindly ensure to pip install these packages before running the programme.

The GUI provide a way to edit the learning rate, discount factor and number of training episodes/timesteps of our reinforcement learning algorithm and see how it impacts the performance of our model.


