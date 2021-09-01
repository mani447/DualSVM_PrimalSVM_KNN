import cvxopt
import numpy as np
import pandas as pd

def load_data(file_path):
    with open(file_path, 'rb') as input_file:
        df = pd.read_csv(input_file, sep=',', header=None)
    input_data = np.array(df[df.columns[1:]])
    temp = np.ones((input_data.shape[0], input_data.shape[1] + 1))
    temp[:, :-1] = input_data
    input_data = temp
    output_data = df[df.columns[0]]
    temp = []
    for x in output_data:
        if x == 0:
            temp.append(-1)
        else:
            temp.append(1)
    output_data = np.transpose(np.array(temp))
    return input_data, output_data
	
def get_accuracy(input_data, output_data, w_vec):
    estimated_output = np.matmul(w_vec, np.transpose(input_data))
    margin_vector = np.transpose(np.multiply(estimated_output, np.transpose(output_data)))
    acc = 100 - ((len(np.where(margin_vector <= 0)[0])/len(input_data)) * 100)
    print(len(np.where(margin_vector <= 0)[0]))
    return acc

def primal_svm_slack(i_data, o_data, c):
    if i_data.shape[0] != o_data.shape[0]:
        raise ValueError("Input and Output data size Mismatch")
    dataset_size = i_data.shape[0]
    input_size = i_data.shape[1]
    c_matrix = np.zeros((dataset_size * 2, input_size + dataset_size))
    h = np.zeros((dataset_size * 2, 1))
    for j in range(0, dataset_size):
        c_matrix[j, 0:input_size] = -1 * i_data[j] * o_data[j]
        c_matrix[j, input_size + j] = -1
        h[j] = -1
    for j in range(dataset_size, 2 * dataset_size):
        c_matrix[j, input_size + j - dataset_size] = -1
    p = np.zeros((input_size + dataset_size, input_size + dataset_size))
    for j in range(0, input_size):
        p[j][j] = 1
    q = np.zeros((input_size + dataset_size, 1))
    q[input_size:] = c
    p = cvxopt.matrix(p, tc='d')
    q = cvxopt.matrix(q, tc='d')
    g = cvxopt.matrix(c_matrix, tc='d')
    h = cvxopt.matrix(h, tc='d')
    sol = cvxopt.solvers.qp(p, q, g, h)
    return list(sol['x'])
	
park_train_in, park_train_out = load_data(r"park_train.data")
park_valid_in, park_valid_out = load_data(r"park_validation.data")
park_test_in, park_test_out = load_data(r"park_test.data")
input_size = park_train_in.shape[1]
train_accuracy = []
valid_accuracy = []
test_accuracy = []
for i in range(0, 9):
    c = np.power(10, i)
    W = primal_svm_slack(park_train_in, park_train_out, c)
    train_accuracy.append(get_accuracy(park_train_in, park_train_out, W[0:input_size]))
    valid_accuracy.append(get_accuracy(park_valid_in, park_valid_out, W[0:input_size]))
    test_accuracy.append(get_accuracy(park_test_in, park_test_out, W[0:input_size]))

print(train_accuracy, valid_accuracy, test_accuracy)