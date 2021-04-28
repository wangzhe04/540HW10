import gym
import random
import numpy as np
import time
from collections import deque
import pickle


from collections import defaultdict


EPISODES =   20000
LEARNING_RATE = .1
DISCOUNT_FACTOR = .99
EPSILON = 1
EPSILON_DECAY = .999



def default_Q_value():
    return 0


if __name__ == "__main__":



    random.seed(1)
    np.random.seed(1)
    env = gym.envs.make("FrozenLake-v0")
    env.seed(1)
    env.action_space.np_random.seed(1)
    total_reward = 0


    Q_table = defaultdict(default_Q_value) # starts with a pessimistic estimate of zero reward for each state.

    #Q_table = np.zeros([env.observation_space.n, env.action_space.n])

    episode_reward_record = deque(maxlen=100)


    for i in range(EPISODES):
        # episode_reward = 0
        done = False
        state = env.reset()
        reward = 0

        #TODO PERFORM Q LEARNING

        while not done:
            if random.uniform(0,1) < EPSILON:
                action = env.action_space.sample()
            else:
                prediction = np.array([Q_table[(state, i)] for i in range(env.action_space.n)])
                action = np.argmax(prediction)

            # Take action
            next_state, reward, done, info = env.step(action)

            if done == False:
                q_value = Q_table[state, action]
                max_value = np.max(Q_table[next_state])


                # print(Q_table[next_state])
                new_q_value = (1 - LEARNING_RATE) * q_value + LEARNING_RATE * (reward + DISCOUNT_FACTOR * max_value)

                Q_table[state, action] = new_q_value
                state = next_state
            else:
                q_value = Q_table[state, action]
                max_value = np.max(Q_table[next_state])
                new_q_value = (1 - LEARNING_RATE) * q_value + LEARNING_RATE * reward

                Q_table[state, action] = new_q_value
                state = next_state

        episode_reward_record.append(reward)
        EPSILON = EPSILON * EPSILON_DECAY







        total_reward += reward



        if i%100 ==0 and i>0:
            print("LAST 100 EPISODE AVERAGE REWARD: " + str(sum(list(episode_reward_record))/100))
            print("EPSILON: " + str(EPSILON) )
    
    
    ####DO NOT MODIFY######
    model_file = open('Q_TABLE.pkl' ,'wb')
    pickle.dump([Q_table,EPSILON],model_file)
    #######################







