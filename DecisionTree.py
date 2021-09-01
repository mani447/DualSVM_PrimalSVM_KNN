import numpy as np
import pandas as pd

def get_accuracy(estimated, actual):
    dataset_size = len(estimated)
    bool_list = [1 if estimated[i] == actual[i] else 0 for i in range(0, dataset_size)]
    acc = sum(bool_list) / dataset_size
    return acc * 100
	
def predict(root, X):
    dataset_size = X.shape[0]
    predictions = list()
    for k in range(0, dataset_size):
        temp_node = root
        feature_idx = temp_node.feature_index
        split_variables = temp_node.split_conditions
        split_cond = X.loc[k, feature_idx]
        idx = split_variables.index(split_cond)
        temp_node = temp_node.children[idx]
        predictions.append(temp_node.predicted_class)
    return predictions
	
def conditional_entropy(feature_vector, Y):
    unique_features = list(set(feature_vector))
    label_space = list(set(Y))
    prior_probs = dict()
    cond_probs = dict()
    cond_entropy = 0
    for feature in unique_features:
        prior_probs[feature] = len(np.where(feature_vector == feature)[0]) / len(feature_vector)
        y_wrt_feature = Y[feature_vector.index[np.where(feature_vector == feature)[0]]]
        temp = dict()
        for label in label_space:
            if len(y_wrt_feature) != 0:
                temp[label] = len(np.where(y_wrt_feature == label)[0]) / len(y_wrt_feature)
                cond_entropy = cond_entropy + prior_probs[feature] * np.log2(temp[label] ** temp[label])
            else:
                temp[label] = 0
        cond_probs[feature] = temp
    cond_entropy = -1 * cond_entropy
    return cond_entropy
	
def get_best_attribute(X, Y):
    cond_entropy = dict()
    for col in X.columns:
        cond_entropy[col] = conditional_entropy(X[col], Y)
    idx = min(cond_entropy, key=lambda k: cond_entropy[k])
    min_entropy = min(cond_entropy.values())
    return idx, min_entropy
	
class Node:
    def __init__(self, entropy, no_of_samples, no_of_samples_per_class, predicted_class):
        self.entropy = entropy
        self.no_of_samples = no_of_samples
        self.no_of_samples_per_class = no_of_samples_per_class
        self.predicted_class = predicted_class
        self.feature_index = 0
        self.IG = 0
        self.split_conditions = []
        self.children = None
		
def get_entropy(Y):
    label_space = list(set(Y))
    entropy = 0
    for label in label_space:
        no_wrt_label = len(np.where(Y == label)[0])
        prob = no_wrt_label / len(Y)
        entropy = entropy - np.log2(prob ** prob)
    return entropy
	
def Decision_Tree_One_Split(dataX, dataY, parent_class, depth):
    samples_per_class = {i: sum(dataY == i) for i in set(dataY)}
    if len(dataY) == 0:
        max_class = parent_class
    else:
        max_class = max(samples_per_class, key=lambda k: samples_per_class[k])
        
    node = Node(entropy=get_entropy(dataY), no_of_samples=len(dataY), no_of_samples_per_class=samples_per_class,
                predicted_class=max_class)
    if(depth == 0):
        return node
    if (get_entropy(dataY) != 0) & (dataX.shape[1] != 1) & (len(dataY) != 0):
        idx, entropy_min = get_best_attribute(dataX, dataY)
        node.IG = node.entropy - entropy_min
        node.feature_index = idx
        unique_features = feature_space[idx]
        node.split_conditions = unique_features
        child_list = []
        for feature in unique_features:
            feature_indexes = dataX.index[np.where(dataX[idx] == feature)[0]]
            dataX_new = dataX.loc[feature_indexes, dataX.columns != idx]
            dataY_new = dataY[feature_indexes]
            child_list.append(Decision_Tree_One_Split(dataX_new, dataY_new, max_class, depth - 1))
        node.children = child_list
    return node
	
def Decision_Tree_Last_Split(dataX, dataY, parent_class):
    samples_per_class = {i: sum(dataY == i) for i in set(dataY)}
    if len(dataY) == 0:
        max_class = parent_class
    else:
        max_class = max(samples_per_class, key=lambda k: samples_per_class[k])
        
    node = Node(entropy=get_entropy(dataY), no_of_samples=len(dataY), no_of_samples_per_class=samples_per_class,
                predicted_class=max_class)
    
    if (get_entropy(dataY) != 0) & (dataX.shape[1] != 1) & (len(dataY) != 0):
        idx = dataX.shape[1]-1
        idx_entropy = conditional_entropy(dataX[idx], dataY)
        node.IG = node.entropy - idx_entropy
        node.feature_index = idx
        unique_features = feature_space[idx]
        node.split_conditions = unique_features
        child_list = []
        for feature in unique_features:
            feature_indexes = dataX.index[np.where(dataX[idx] == feature)[0]]
            dataX_new = dataX.loc[feature_indexes, dataX.columns != idx]
            dataY_new = dataY[feature_indexes]
            child_list.append(Decision_Tree_Last_Split(dataX_new, dataY_new, max_class))
        node.children = child_list
    return node
	
def getInformationGainList(root, NodeIGL):
    NodeIGL[root.feature_index] = root.IG
    
    if(root.children == None):
        return
    
    for i in root.children:
        getInformationGainList(i, NodeIGL)
    return
	
train_file = r'C:\Users\Manid\Music\Study\CourseWork\Semester1\ML_Ruozzi\Assignments\Project2\mush_train.data'
with open(train_file, 'r') as csv_file:
    df = pd.read_csv(csv_file, sep=',', header=None)
    mush_train_in = df[df.columns[1:]]
    mush_train_out = df[0]

test_file = r"C:\Users\Manid\Music\Study\CourseWork\Semester1\ML_Ruozzi\Assignments\Project2\mush_test.data"
with open(test_file, 'r') as csv_file:
    df = pd.read_csv(csv_file, sep=',', header=None)
    mush_test_in = df[df.columns[1:]]
    mush_test_out = df[0]

feature_space = {1: ['b', 'c', 'x', 'f', 'k', 's'],
                 2: ['f', 'g', 'y', 's'],
                 3: ['n', 'b', 'c', 'g', 'r', 'p', 'u', 'e', 'w', 'y'],
                 4: ['t', 'f'],
                 5: ['a', 'l', 'c', 'y', 'f', 'm', 'n', 'p', 's'],
                 6: ['a', 'd', 'f', 'n'],
                 7: ['c', 'w', 'd'],
                 8: ['b', 'n'],
                 9: ['k', 'n', 'b', 'h', 'g', 'r', 'o', 'p', 'u', 'e', 'w', 'y'],
                 10: ['e', 't'],
                 11: ['b', 'c', 'u', 'e', 'z', 'r', '?'],
                 12: ['f', 'y', 'k', 's'],
                 13: ['f', 'y', 'k', 's'],
                 14: ['n', 'b', 'c', 'g', 'o', 'p', 'e', 'w', 'y'],
                 15: ['n', 'b', 'c', 'g', 'o', 'p', 'e', 'w', 'y'],
                 16: ['p', 'u'],
                 17: ['n', 'o', 'w', 'y'],
                 18: ['n', 'o', 't'],
                 19: ['c', 'e', 'f', 'l', 'n', 'p', 's', 'z'],
                 20: ['k', 'n', 'b', 'h', 'r', 'o', 'u', 'w', 'y'],
                 21: ['a', 'c', 'n', 's', 'v', 'y'],
                 22: ['g', 'l', 'm', 'p', 'u', 'w', 'd']}
				 
root = Decision_Tree_Last_Split(mush_train_in, mush_train_out, 'e')
output_predicted = predict(root, mush_test_in)
accuracy = get_accuracy(output_predicted, mush_test_out)
print("Accuracy using last attribute for split is: ",accuracy)

NodeIGL = dict()
getInformationGainList(root, NodeIGL)
print("Information Gain for each node is:")
print(NodeIGL)

root = Decision_Tree_One_Split(mush_train_in, mush_train_out, 'e', 2)
output_predicted = predict(root, mush_test_in)
accuracy = get_accuracy(output_predicted, mush_test_out)
print("Accuracy using last attribute for split is: ",accuracy)