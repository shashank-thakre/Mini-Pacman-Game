#!/usr/bin/env python
# coding: utf-8

# In[1]:


from mini_pacman import test, random_strategy, naive_strategy


# In[2]:


import random
import gc
import time
import numpy as np

from tensorflow.keras.models import Sequential, clone_model
from tensorflow.keras.layers import Dense, InputLayer
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import CSVLogger, TensorBoard
import tensorflow.keras.backend as K


# In[3]:


import json
from mini_pacman import PacmanGame

with open('test_params.json', 'r') as file:
    read_params = json.load(file)
game_params = read_params['params']
env = PacmanGame(**game_params)


# In[4]:


obs = env.get_obs()
obs


# In[5]:


def get_state(obs):
    v = []
    x,y = obs['player']
    v.append(x)
    v.append(y)
    for x, y in obs['monsters']:
        v.append(x)
        v.append(y)
    for x, y in obs['diamonds']:
        v.append(x)
        v.append(y)
    for x, y in obs['walls']:
        v.append(x)
        v.append(y)
    return v


# Create a function constructing DQN with 3 hidden layers of 8 units each, input with the shape of observation of the environment and output with the shape of available actions.

# In[6]:


def create_dqn_model(input_shape, nb_actions):
    model = Sequential()
    model.add(Dense(units=32, input_shape=input_shape, activation='relu'))
    model.add(Dense(units=64, activation='relu'))
    model.add(Dense(units=128, activation='relu'))
    model.add(Dense(units=72, activation='relu'))
    model.add(Dense(units=36, activation='relu'))
    model.add(Dense(nb_actions, activation='relu'))
    return model


# Compile the online network using Adam optimizer and loss function of type mse. Clone the online network as target network fixing the same weights as in online network.

# In[7]:


input_shape = (32,)
nb_actions = 9
print('input_shape: ',input_shape)
print('nb_actions: ',nb_actions)

online_network = create_dqn_model(input_shape, nb_actions)
online_network.compile(optimizer=Adam(0.001), loss='mse') #0.001 is the learning rate
target_network = clone_model(online_network)
target_network.set_weights(online_network.get_weights())


# Create a function selecting action according to  𝜀
#  -greedy algorithm, i.e. select action randomly with probability  𝜀
#   and action with best known Q-value with probability  1−𝜀
# 

# In[8]:


def epsilon_greedy(q_values, epsilon, n_outputs):
    if random.random() < epsilon:
        return random.choice(n_outputs)  # random action
    else:
        return np.argmax(q_values)+1  # q-optimal action


# In[9]:


n_steps = 300000 # number of times 
warmup = 10000 # first iterations after random initiation before training starts
training_interval = 4 # number of steps after which dqn is retrained
copy_steps = 10_000 # number of steps after which weights of 
                   # online network copied into target network
gamma = 0.999999 # discount rate
batch_size = 64 # size of batch from replay memory 
eps_max = 1.0 # parameters of decaying sequence of eps
eps_min = 0.05
eps_decay_steps = 75_000


# Create replay_memory - a storage of experienced transitions - as deque.

# In[10]:


from collections import deque
replay_memory_maxlen = 1_000_000
replay_memory = deque([], maxlen=replay_memory_maxlen)


# Running them for large enough number of steps n_steps allows training DQN for prediction of the Q-values.

# In[11]:


step = 0
iteration = 0
done = True

while step < n_steps:
    if done:
        obs = env.reset()
        old_state = get_state(obs)
    iteration += 1
    q_values = online_network.predict(np.array([old_state]))[0]  
    epsilon = max(eps_min, eps_max - (eps_max-eps_min) * step/eps_decay_steps)
    action = epsilon_greedy(q_values, epsilon, obs['possible_actions'])
    next_obs = env.make_action(action)
    new_state = get_state(next_obs)
    reward = next_obs['reward']
    done = next_obs["end_game"]
    replay_memory.append((old_state, action, reward, new_state, done))
    old_state = new_state

    if iteration >= warmup and iteration % training_interval == 0:
        step += 1
        minibatch = random.sample(replay_memory, batch_size)
        replay_state = np.array([x[0] for x in minibatch])
        replay_action = np.array([x[1] for x in minibatch])
        replay_rewards = np.array([x[2] for x in minibatch])
        replay_next_state = np.array([x[3] for x in minibatch])
        replay_done = np.array([x[4] for x in minibatch], dtype=int)
        #target_for_action = replay_rewards + (1-replay_done) * gamma * np.amax(target_network.predict(replay_next_state), axis=1)
        # Double DQN Method below
        best_actions = np.argmax(online_network.predict(replay_next_state), axis=1)
        target_for_action = replay_rewards + (1-replay_done) * gamma * \
                                    target_network.predict(replay_next_state)[np.arange(batch_size), best_actions]
        # Double DQN Method above
        target = online_network.predict(replay_state)  # targets coincide with predictions ...
        replay_action = replay_action - 1
        target[np.arange(batch_size), replay_action] = target_for_action  #...except for targets with actions from replay
        online_network.fit(replay_state, target, epochs=step, verbose=0, initial_epoch=step-1)
        if step % copy_steps == 0:
            target_network.set_weights(online_network.get_weights())
            


# In[12]:


online_network.save('sv_pacman_dbl_dqn_model.h5')


# In[14]:


from tensorflow.keras.models import load_model
pacman_dqn_model = load_model('sv_pacman_dbl_dqn_model_300k.h5')


# Output

# In[15]:


#test(strategy=random_strategy, log_file='test_pacman_log_RS.json')


# In[17]:


#test(strategy=naive_strategy, log_file='test_pacman_log_NS.json')


# In[18]:


def dqn_strategy(obs):
    new_state = get_state(obs)
    q_values = pacman_dqn_model.predict(np.array([new_state]))[0]  
    epsilon = max(eps_min, eps_max - (eps_max-eps_min) * step/eps_decay_steps)
    action = epsilon_greedy(q_values, epsilon, obs['possible_actions'])
    return action


# In[19]:


test(strategy=dqn_strategy, log_file='test_pacman_log_dbl_DQN.json')


# In[ ]:


env.close()

