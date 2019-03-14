# Continuous Control Implementation
## Deep Deterministic Policy Gradient Reinforcement Learning Agent

### Synopsis:
Utilizing a Deep Deterministic Policy on the Unity ML ‘reacher’ environment I achieved a single agent solution in 150 turns by averaging a score >30 across 100 episodes.

### Environment:
The reacher environment utilizes a double-jointed arm to move to a target location based an Agent’s policy (it’s Brain).  The agent is able to earn +0.1 points each step that it correctly reaches out to its target location.  The reacher consists of 33 variables controlling position, rotation, velocity, and angular velocities, an action is then carried out by an agent by describing the corresponding torque applicable to 2 joints of the reacher (2 actions per joint for 4 total).
Environment Goal: 
The reacher environment’s benchmark implementation maintains an average of 30pts per episode across 100 episodes – thus I set out to implement a benchmark DDPG reinforcement model which achieved benchmark results after tuning hyperparameters appropriately.


### DDPG Algorithm Background:
This implementation pf the actor critic method of deep reinforcement learning utilizes a Deep Deterministic Policy Gradient (DDPG) to evaluate a continuous action space.  DDPG is based on the papers ‘Deterministic Policy Gradient Algorithms’ published in 2014 by David Silver and ‘Continuous Control with Deep Reinforcement Learning’ published by Tomothy P. Lillicrap in 2015.
Unlike other actor critic methods that rely on stochastic distributions to return probabilities across a discreet action space, DDPG utilizes a deterministic policy to directly estimate a set of actions actions based on the environment’s current state.  As a result, DDPG is able to take advantage of Q values (much like DQN) which allows the estimation of rewards by maximizing Q via a feed-forward Critic network.  The actor feed-forward network then is able to use the critic’s value estimates to choose the action that maximizes Q via back-propagation (stochastic gradient decent of the deterministic policy gradient allows optimizing for Q by minimizing the MSE of the objective).

Like DQN, DDPG requires us to explore the environment to determine the correct policy – this is accomplished by adding noise via the Ornstein-Uhlenbeck process (ON) to explore the environment controlled by some value of Epsilon controlling how greedy the policy is.


### How to run:
1.	Clone this repository onto your machine and unzip into a directory of your choice
2.	Download and install Anaconda if needed
3.	Create a new environment and install the package requirements listed below
4.	Download the Unity environment and unzip the executable into the project’s parent directory.
        a.	It is recommended that your video card be CUDA compatible for optimal performance
5.	Using Continuous_Control.ipynb in a Jupyter notebook train and/or play the agent within the Unity Environment after loading saved critic and actor checkpoints saved during training.


### Requirements:
1.	python 3.6+
2.	pytorch 1.0+ (Instructions: https://pytorch.org/)
3.	CUDA 9.0+
4.	UnityAgent (Instructions: https://github.com/Unity-Technologies/ml-agents)
5.	Jupyter NoteBook
6.	Numpy
7.	Matplotlib
8.	Reacher Unity Environment (Linux),(OSX), (Win64),(Win32)



### Running the Code:
1.	Select either a 1 agent or a 20 agent Unity Environment and examine the state and action spaces of the environment by taking random steps as defined by the OU noise added at each step.
2.	Train the agent by running the appropriate training loop within Continuous_Control.ipynb until obtaining a 100 episode moving average score greater than 30.
3.	Show the results by graphing each episode’s average score as well as the 100 episode moving average
4.	Watch the actions taken by your newly trained smart agent by loading saved checkpoints into the DDPG training loop and setting the train variable to False.


