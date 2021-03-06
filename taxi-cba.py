# TODO: 
# 1. implement classifier(), update_classifier(), update_threshhold()


import gym
import numpy as np
#from IPython.display import clear_output
from sklearn.neighbors import NearestNeighbors
from sklearn.svm import SVC
from classifiers import classifier, update_classifier, process_state, nearest_neighbor, update_threshold

env = gym.make("Taxi-v2").env

########## Training the agent #################################################
# Hyperparameters
t_conf_gamma = 2
t_dist_gamma = 2

# For plotting metrics
all_epochs = []
svm = SVC(decision_function_shape='ovr')
#q_table = np.zeros([env.observation_space.n, env.action_space.n])
nn = NearestNeighbors()

for i in range(1, 4):
    state = env.reset()
    epochs = 0
    done = False
    states = []
    actions = []
    action_space = [0,1,2,3,4,5]
    t_conf = np.zeros(len(action_space))
    for i in range(len(action_space)):
        t_conf[i] = float("inf")
    t_dist = 0

    while not done:
        # 1. get sensor data step (i.e. the current state.  this is different for continuous actions)
        # 2. ask if expert wants to perform the corrective demonstration
        pres = "o"
        env.render()
        while pres != "y" and pres != "n":
            pres = input("Do you want to correct the last action taken? [y/n]: ")
        
        if pres == "n":     # 3. execute the Confident Execution step
            # 5. Put current state into classifier to get a_p, c, and db
            # 6. Get nearest neighbor for state
            a_p, c, db = classifier(svm, process_state(env.decode(state)))
            d = nearest_neighbor(nn, states, process_state(env.decode(state)))
            #states.append(process_state(env.decode(state)))

            if c > t_conf[a_p] and d < t_dist:
                state, reward, done, info = env.step(a_p)
                states.append(process_state(env.decode(state)))
                actions.append(a_p)
            else:
                pres = -1
                while pres not in [0, 1, 2, 3, 4, 5]:
                    pres = int(input("Expert: enter the next action I should take [0-5]: "))
                states.append(process_state(env.decode(state)))
                actions.append(pres)
                svm = update_classifier(states, actions)
                t_conf, t_dist = update_threshold(states, actions, action_space, t_dist_gamma, t_conf_gamma)


                state, reward, done, info = env.step(pres)
        else:   # 4. execute the Corrective Demonstration step
            pres = -1
            env.render()
            while pres not in [0, 1, 2, 3, 4, 5]:
                pres = int(input("Expert: enter the next action I should take [0-5]: "))
            states.append(process_state(env.decode(state)))
            actions.append(pres)
            svm = update_classifier(states, actions)
            t_conf, t_dist = update_threshold(states, actions, action_space, t_dist_gamma, t_conf_gamma)

            if (len(actions) < 5):
                state, reward, done, info = env.step(pres)
        
    if i % 100 == 0:
        #clear_output(wait=True)
        print(f"Episode: {i}")

print("Training finished.\n")


######### Evaluate agent's performance after CBA Training #####################

total_epochs, total_penalties = 0, 0
episodes = 100

for _ in range(episodes):
    state = env.reset()
    epochs, penalties, reward = 0, 0, 0
    
    done = False
    
    while not done:
        action = np.argmax(q_table[state])
        state, reward, done, info = env.step(action)

        if reward == -10:
            penalties += 1

        epochs += 1

    total_penalties += penalties
    total_epochs += epochs

print(f"Results after {episodes} episodes:")
print(f"Average timesteps per episode: {total_epochs / episodes}")
print(f"Average penalties per episode: {total_penalties / episodes}")
