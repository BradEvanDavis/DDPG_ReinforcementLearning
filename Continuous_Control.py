
#get_ipython().system('pip -q install ./python')

from unityagents import UnityEnvironment
import numpy as np
from agent import Agent
import random
import torch
from collections import deque
import matplotlib.pyplot as plt
import time
%matplotlib inline

# select this option to load version 1 (with a single agent) of the environment
env = UnityEnvironment(file_name='.\Reacher1\Reacher.exe')

# select this option to load version 2 (with 20 agents) of the environment
# env = UnityEnvironment(file_name='.\Reacher20\Reacher.exe')

# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

# reset the environment
env_info = env.reset(train_mode=True)[brain_name]

# number of agents
num_agents = len(env_info.agents)
print('Number of agents:', num_agents)

# size of each action
action_size = brain.vector_action_space_size
print('Size of each action:', action_size)

# examine the state space 
states = env_info.vector_observations
state_size = states.shape[1]
print('There are {} agents. Each observes a state with length: {}'.format(states.shape[0], state_size))
print('The state for the first agent looks like:', states[0])

mean_scores = []
min_scores = []
max_scores = []
best_score = -np.inf
scores_window = deque(maxlen=100)
moving_avgs = []

def ddpg(n_episodes=300, max_t=700, train=True, num_agents=1, print_every=20, train_mode=True):
    
    mean_scores = []
    min_scores = []
    max_scores = []
    best_score = -np.inf
    scores_window = deque(maxlen=100)
    moving_avgs = []

    
    #for i in range(num_agents):
     #   agents.append(Agent(state_size, action_size, random_seed=44))
    
    for i_episode in range(0, n_episodes):
        env_info = env.reset(train_mode=True)[brain_name]
        states = env_info.vector_observations
        scores = np.zeros(num_agents)
        agent.reset()
        start_time = time.time()

        for t in range(max_t): 
            actions = agent.act(states, add_noise=True)
            env_info = env.step(actions)[brain_name]        # send the action to the environment
            next_states = env_info.vector_observations   # get the next state
            rewards = env_info.rewards                  # get the reward
            dones = env_info.local_done                # see if episode has finished
        
            for state, action, reward, next_state, done in zip(states, actions, rewards, next_states, dones):
                agent.step(state, action, rewards, next_state, done, t)
                
            states = next_state
            scores += rewards
            if np.any(dones):
                break

        duration = time.time() - start_time
        min_scores.append(np.min(scores))
        max_scores.append(np.max(scores))
        mean_scores.append(np.mean(scores))
        scores_window.append(mean_scores[-1])
        moving_avgs.append(np.mean(scores_window))
                
        if i_episode % print_every == 0:
            print('\rEpisode {}, Mean last 100: {:.2f}, Mean current: {:.2f}, Max: {:.2f}, Min: {:.2f}, Time: {:.2f}'\
                .format(i_episode, moving_avgs[-1], mean_scores[-1], max_scores[-1], min_scores[-1], duration, end="\n"))
            torch.save(agent.actor_local.state_dict(), 'checkpoint_actor.pth')
            torch.save(agent.critic_local.state_dict(), 'checkpoint_critic.pth')

        
        if moving_avgs[-1]>=30.0:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode, moving_avg[-1]))
            torch.save(agent.actor_local.state_dict(), 'checkpoint_actor.pth')
            torch.save(agent.critic_local.state_dict(), 'checkpoint_critic.pth')
            
    return scores



agent = Agent(state_size=state_size, action_size=action_size, random_seed=44)
scores = ddpg()


fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(1, len(scores)+1), scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.show()

