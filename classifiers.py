import numpy as np
from scipy.spatial import cKDTree

from sklearn.neighbors import NearestNeighbors
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

# Note for later: for us, the actions is the vector/direction of the pan

def classifier(svm: SVC, feature):
    #taxirow, taxicol, passloc, destidx = state
    #feature = [taxirow, taxicol, passloc, destidx]
    c = svm.decision_function(feature)
    a_p = svm.predict(feature)
    return a_p, c, None


def update_classifier(nn, states, actions):
    svm = SVC(decision_function_shape='ovr')
    if (len(actions) < 5):
        return svm
    svm.fit(states, actions)
    nn = NearestNeighbors()
    nn.fit(states, actions)
    return svm, nn


def nearest_neighbor(nn, states, state):
    # nn.fit(T)
    # total_distance = 0
    # for state_action_pair_datum in T:
    #     distance, neighbor = nn.kneighbors(state_action_pair_datum, 1, return_distance=True)
    #     total_distance += distance
    #
    # return t_dist_gamma * (total_distance / len(T))
    nn.fit(states)
    distance, neighbor = nn.kneighbors(state, 1, return_distance=True)
    return distance


def update_threshold(states, actions, action_space, t_dist_gamma, t_conf_gamma):
    if (len(states) < 5):
        t_conf = np.zeros(len(action_space))
        for i in range(len(action_space)):
            t_conf[i] = float("inf")
        t_dist = 0
        return t_conf, t_dist


    # t_dist update
    kdt = cKDTree(states)
    dists, neighs = kdt.query(states, 2)

    dists = np.sum(dists, axis=1)
    avg_dist = np.mean(dists)
    t_dist = t_dist_gamma * avg_dist

    # t_conf update
    # 1. split dataset into training and test
    X_train, X_test, y_train, y_test = train_test_split(states, actions, test_size = 0.33, random_state = 42)
    svm_temp = SVC(decision_function_shape='ovr')
    svm_temp.fit(X_train, y_train)
    y_test_pred = svm_temp.predict(X_test)
    y_test_conf = svm_temp.decision_function(X_test)

    # 2. determine the # of incorrectly classified points per class
    miss_classified_num = np.zeros(len(action_space))
    miss_classified_conf = np.zeros(len(action_space))

    for i in range(len(y_test_pred)):
        if y_test[i] != y_test_pred[i]:
            miss_classified_num[y_test[i]] += 1
            miss_classified_conf[y_test[i]] += y_test_conf[i]

    # 3. evaluate confidence thresholds for each class
    t_conf = np.zeros(len(action_space))
    for i in range(len(action_space)):
        if miss_classified_num[i] != 0:
            t_conf[i] = t_conf_gamma * (miss_classified_conf[i] / miss_classified_num[i])

    return t_conf, t_dist



def process_state(state):
    taxirow, taxicol, passloc, destidx = state
    return taxirow, taxicol, passloc, destidx