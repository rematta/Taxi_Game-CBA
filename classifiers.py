import numpy as np
from scipy.spatial import cKDTree
import copy

from sklearn.neighbors import NearestNeighbors
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split


# Note for later: for us, the actions is the vector/direction of the pan

def classifier(svm: SVC, feature):
    # taxirow, taxicol, passloc, destidx = state
    # feature = [taxirow, taxicol, passloc, destidx]
    a_p = svm.predict([feature])
    c = svm.decision_function([feature])
    c = c[0][a_p[0]]
    return a_p[0], c, None


def update_classifier(states, actions):
    # add dummy values to make sure there is 1 state for each action in our action space
    st = copy.deepcopy(states)
    at = copy.deepcopy(actions)
    st.append((-100, -100, -100, -100))
    st.append((-101, -100, -100, -100))
    st.append((-102, -100, -100, -100))
    st.append((-103, -100, -100, -100))
    st.append((-104, -100, -100, -100))
    st.append((-105, -100, -100, -100))
    at.append(0)
    at.append(1)
    at.append(2)
    at.append(3)
    at.append(4)
    at.append(5)

    svm = SVC(decision_function_shape='ovr')
    if len(at) < 5:
        return svm
    svm.fit(st, at)

    return svm


def nearest_neighbor(nn, states, state):
    # nn.fit(T)
    # total_distance = 0
    # for state_action_pair_datum in T:
    #     distance, neighbor = nn.kneighbors(state_action_pair_datum, 1, return_distance=True)
    #     total_distance += distance
    #
    # return t_dist_gamma * (total_distance / len(T))
    nn.fit(states)
    distance, neighbor = nn.kneighbors([state], 1, return_distance=True)
    return distance[0][0]


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
    X_train, X_test, y_train, y_test = train_test_split(states, actions, test_size=0.33, random_state=42)
    # add dummy values to make sure there is 1 state for each action in our action space
    X_train.append((-100, -100, -100, -100))
    X_train.append((-101, -100, -100, -100))
    X_train.append((-102, -100, -100, -100))
    X_train.append((-103, -100, -100, -100))
    X_train.append((-104, -100, -100, -100))
    X_train.append((-105, -100, -100, -100))
    y_train.append(0)
    y_train.append(1)
    y_train.append(2)
    y_train.append(3)
    y_train.append(4)
    y_train.append(5)

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
            miss_classified_conf[y_test[i]] += y_test_conf[i][y_test[i]]

    # 3. evaluate confidence thresholds for each class
    t_conf = np.zeros(len(action_space))
    for i in range(len(action_space)):
        if miss_classified_num[i] != 0:
            t_conf[i] = t_conf_gamma * (miss_classified_conf[i] / miss_classified_num[i])

    return t_conf, t_dist


def process_state(state):
    taxirow, taxicol, passloc, destidx = state
    return taxirow, taxicol, passloc, destidx
