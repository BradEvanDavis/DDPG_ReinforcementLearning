# Continuous Control Implementation
## Deep Deterministic Policy Gradient Reinforcement Learning Agent

### Synopsis:
Utilizing a Deep Deterministic Policy on the Unity ML ‘reacher’ environment I achieved a single agent solution in 150 turns by averaging a score >30 across 100 episodes.

### Environment:
The reacher environment utilizes a double-jointed arm to move to a target location based an Agent’s policy (it’s Brain).  The agent is able to earn +0.1 points each step that it correctly reaches out to its target location.  The reacher consists of 33 variables controlling position, rotation, velocity, and angular velocities, an action is then carried out by an agent by describing the corresponding torque applicable to 2 joints of the reacher (2 actions per joint for 4 total).
Environment Goal: 
The reacher environment’s benchmark implementation maintains an average of 30pts per episode across 100 episodes – thus I set out to implement a benchmark DDPG reinforcement model which achieved benchmark results after tuning hyperparameters appropriately.


### DDPG Algorithm Background:
This actor-critic implementation utilizes deep reinforcement learning known as Deep Deterministic Policy Gradient (DDPG) to evaluate a continuous action space.  DDPG is based on the papers ‘Deterministic Policy Gradient Algorithms’ published in 2014 by David Silver and ‘Continuous Control with Deep Reinforcement Learning’ published by Tomothy P. Lillicrap in 2015.

Unlike other actor-critic methods that rely on stochastic distributions to return probabilities across a discreet action space, DDPG utilizes a deterministic policy to directly estimate a set of continuous actions based on the environment’s current state.  As a result, DDPG is able to take advantage of Q values (much like DQN) which allows the estimation of rewards by maximizing Q via a feed-forward Critic network.  The actor feed-forward network then is able to use the critic’s value estimates to choose the action that maximizes Q via back-propagation (stochastic gradient decent of the deterministic policy gradient allows optimizing for Q by minimizing MSE).

Like DQN, DDPG requires the agent to explore the environment in-order to determine an optimal policy – this is accomplished by adding noise via the Ornstein-Uhlenbeck process (ON) to explore the environment.  This implementation additionally adds a value of Epsilon to control how greedy the policy is, as well as a learning modifier to control when the model implements a learning step.  

To better control learning this implementation decays the value of Epsilon for each action.  At the beginning of training the model Epsilon is set to 1 and adds ON noise to every action.  Epsilon is then decayed over time so that the policy gradually becomes more greedy to exploit value maximizing actions.  

Specifically, within this implementation learning via gradient descent is conducted after each step within the environment for the first 200 turns to help speed up training.  After 200 episodes the model continues to train at a changed rate, then based on the accumulation of experiences from 20 steps the model learns over 10 passes – this ultimately helps stabilize scores over time.

Of note, using this methodology in some instances resulted in stalled learning via a ‘learning ceiling’.  The model ultimately selected was able to break this barrier based on the addition of noise after training several models, therefore a good way to choose an optimal implementation is to train multiple independent agents and then to select the top performing ones from that set of trained models.

In order to further improve the implemented DDPG model several improvements could be made by implementing updated features including prioritized replay, training multiple agents across independently generated policies, and through dynamic epsilon values that choose greedy policies or exploration dependent on the performance of the network and observations within the environment. 


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


