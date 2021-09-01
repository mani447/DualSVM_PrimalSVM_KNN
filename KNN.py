#netid:MRM190005
import pandas as pd
import numpy as np
from scipy import stats
from sklearn import preprocessing

def load_data(file_path):
    with open(file_path, 'rb') as input_file:
        data = pd.read_csv(input_file, sep=',',header=None)
        target_data = pd.DataFrame(data[0], columns=[0])
        features_data = data.drop(columns = [0],axis=1)
    return target_data,features_data.to_numpy()

def data_normalization(data):
    mean = np.mean(data, axis=0)
    standard_deviation = np.std(data, axis=0)
    data = (data-mean) / standard_deviation
    return data

def accuracy(predictions,actual):
    TrueNegative=0
    TruePositive=0
    FalseNegative=0
    FalsePositive=0
    accuracy=0
    for i in range(len(predictions)):
        if int(actual[i]) == int(predictions[i]):
            if int(predictions[i]) == 1:
                TrueNegative = TrueNegative+1
            else:
                TruePositive  = TruePositive+1
        else:        
            if int(predictions[i])==1:
                FalseNegative = FalseNegative+1
            else:
                FalsePositive = FalsePositive+1
    accuracy=float((TrueNegative+TruePositive)/len(actual))*100
    return accuracy

def KNN(k,features_train,features_test, target_train):
    predictions = np.zeros((features_test.shape[0],1))
    for i in range(0,features_test.shape[0]):
        difference = np.subtract(features_train,features_test[i,:]) 
        distance = np.linalg.norm(difference,ord=2,axis=1)
        datapoints_index = np.argsort(distance)[0:k]
        nearest_points = target_train.iloc[datapoints_index]
        predictions[i] = stats.mode(nearest_points).mode[0]
    return predictions

target_traindata,features_traindata = load_data("park_train.data")
norm_features_traindata = data_normalization(features_traindata)
target_validdata,features_validdata = load_data("park_validation.data")
norm_features_validdata = data_normalization(features_validdata)
target_testdata,features_testdata = load_data("park_test.data")
norm_features_testdata = data_normalization(features_testdata)
actual = target_testdata.to_numpy()
k_values = [1,5,11,15,21]
accuracyList = dict()
for k in k_values:
    predictions = KNN(k,norm_features_traindata,norm_features_testdata, target_traindata)
    accuracyList[k]=accuracy(predictions,actual)

print(accuracyList)