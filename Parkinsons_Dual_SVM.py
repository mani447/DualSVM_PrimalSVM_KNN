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
    
def gaussian_kernel(x, y, sigma):
    z = x - y
    l2_norm = np.linalg.norm(z, ord=2)
    gaussian = np.exp((-1*l2_norm)/(2*(sigma)))
    return gaussian
    
def get_accuracy(input_data, output_data, train_input_data, train_output_data, lambdas, sigma):
    dataset_size = len(output_data)
    margin = np.zeros((dataset_size, 1))
    for k in range(0, dataset_size):
        for l in range(0, train_input_data.shape[0]):
            margin[k] = margin[k] + (train_output_data[l]*lambdas[l]*gaussian_kernel(input_data[k], train_input_data[l], sigma))
        margin[k] = output_data[k]*margin[k]
    acc = 100 - ((len(np.where(margin <= 0)[0])/len(input_data)) * 100)
    print(acc)
    return acc
	
def dual_svm_slack(i_data, o_data, c, sigma):
    if i_data.shape[0] != o_data.shape[0]:
        raise ValueError("Input and Output data size Mismatch")
    dataset_size = i_data.shape[0]
    input_size = i_data.shape[1]
    c_matrix = np.zeros((2*dataset_size, dataset_size))
    h = np.zeros((dataset_size * 2, 1))
    for j in range(0, dataset_size):
        c_matrix[j, j] = -1
        c_matrix[j+dataset_size, j] = 1
        h[j] = 0
        h[j+dataset_size] = c
    p = np.zeros((dataset_size, dataset_size))
    for i in range(0, dataset_size):
        for j in range(0, dataset_size):
            p[i][j] = o_data[i]*o_data[j]*gaussian_kernel(i_data[i], i_data[j], sigma)
    q = -1 * np.ones((dataset_size, 1))
    cvxopt.solvers.options['maxiters'] = 1000
    p = cvxopt.matrix(p, tc='d')
    q = cvxopt.matrix(q, tc='d')
    g = cvxopt.matrix(c_matrix, tc='d')
    h = cvxopt.matrix(h, tc='d')
    sol = cvxopt.solvers.qp(p, q, g, h)
    lambdas = sol['x']
    return list(lambdas)
	
	
park_train_in, park_train_out = load_data(r"park_train.data")
park_valid_in, park_valid_out = load_data(r"park_validation.data")
park_test_in, park_test_out = load_data(r"park_test.data")
input_size = park_train_in.shape[1]
accuracy_matrix = np.zeros((9, 5, 3))
for i in range(0, 9):
    for j in range(0, 5):
        c = np.power(10, i)
        sigma = 10 ** (j-1)
        lambda_values = dual_svm_slack(park_train_in, park_train_out, c, sigma)
        train_accuracy = get_accuracy(park_train_in, park_train_out, park_train_in, park_train_out, lambda_values, sigma)
        valid_accuracy = get_accuracy(park_valid_in, park_valid_out, park_train_in, park_train_out, lambda_values, sigma)
        test_accuracy = get_accuracy(park_test_in, park_test_out, park_train_in, park_train_out, lambda_values, sigma)
        accuracy_matrix[i, j, :] = [train_accuracy, valid_accuracy, test_accuracy]

print(accuracy_matrix)