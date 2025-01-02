import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
import random

# from keras.datasets import mnist

# (trainX, trainy), (testX, testy) = mnist.load_data()

np.random.seed(1)

data = np.load('mnist.npz')

trainX = data["x_train"]
trainy = data["y_train"]
testX = data["x_test"]
testy = data["y_test"]

new_train = np.isin(trainy, [0, 1])
new_test = np.isin(testy, [0, 1])

X_train, Y_train = trainX[new_train], trainy[new_train] # Making the new dataset out of just classes 0 and 1.
X_test, Y_test = testX[new_test], testy[new_test]

Y_train = np.where(Y_train == 0, -1, Y_train)
Y_test = np.where(Y_test == 0, -1, Y_test) # Replacing 0s with -1s

first_indices = np.where(Y_train == -1)[0]
second_indices = np.where(Y_train == 1)[0]
random_first_indices = np.random.choice(first_indices, size=1000, replace=False)
random_second_indices = np.random.choice(second_indices, size=1000, replace=False)
indices = np.concatenate((random_first_indices, random_second_indices))
indices.sort()
rem_indices = np.delete(np.arange(Y_train.size), indices)

X_val, Y_val = X_train[indices], Y_train[indices]
X_train, Y_train = X_train[rem_indices], Y_train[rem_indices] # Arranging our new Train Set and Val

#------------------------------------------------------------------------
# Computing PCA Matrix

main_len = len(X_train)

data_matrix = []

# Generating data matrix
for i in range(main_len):
    data_matrix.append(X_train[i].flatten().tolist())

data_matrix = np.matrix(data_matrix).T
data_mean = np.mean(data_matrix, axis = 1)
cent_data_matrix = data_matrix - data_mean  # Centralized data

cov_matrix = np.matmul(cent_data_matrix, cent_data_matrix.T) / (main_len - 1) # Covariance matrix of centralized data

eig_1 = np.linalg.eig(cov_matrix)
eigenvalues = eig_1[0].real
eigenvectors = eig_1[1].real

indices = eigenvalues.argsort()[::-1]   # Finding the indices corresponding to the eigenvalues in descending order
U_matrix = eigenvectors[:,indices]  # Sorting the eigenvectors based on descending order of their eigenvalues

U_matrix_new = U_matrix[:,:5]
Y_Matrix = np.matmul(U_matrix_new.T, cent_data_matrix)

#------------------------------------------------------------------------
# Doing Pre-Computation For Efficiency

mis_class_list = [] # Indices of the misclassified points for every split
decision = [] # Stores decision

# This precomputes the mis-classified points in every split
def preCompute(data, labels, real_indices):
    temp_list = []
    dec_list = [[] for i in range(1000)]
    low = data[0]
    high = data[main_len - 1]

    interval = (high - low) / 1001
    midpoint = low + interval

    for i in range(1000):
        dec_list[i].append(midpoint)

        indices = np.where(data <= midpoint)[0]
        mode_label, mode_count = np.unique(labels[indices], return_counts=True)
        mode_label = mode_label[mode_count == np.max(mode_count)] # Finds the mode
        if(len(mode_label) > 1):
            mode_label = random.choice(mode_label)
        dec_list[i].append(mode_label)
        indices = indices[np.where(labels[indices] != mode_label)[0]]
        left_indices = real_indices[indices] # Indices in original array corresponding to mismatches

        indices = np.where(data > midpoint)[0]
        mode_label, mode_count = np.unique(labels[indices], return_counts=True)
        mode_label = mode_label[mode_count == np.max(mode_count)]
        if(len(mode_label) > 1):
            mode_label = random.choice(mode_label)
        dec_list[i].append(mode_label)
        indices = indices[np.where(labels[indices] != mode_label)[0]]
        right_indices = real_indices[indices]

        final_indices = np.concatenate((left_indices, right_indices))
        temp_list.append(final_indices)

        midpoint = midpoint + interval

    mis_class_list.append(temp_list)
    decision.append(dec_list)

def performInitialization(data, labels):
    for i in range(5):
        indices = np.argsort(np.ravel(data[i, :]))
        data2 = data[:, indices]
        labels2 = labels[indices]

        preCompute(np.ravel(data2[i, :]), labels2, indices)

performInitialization(Y_Matrix, Y_train)

#------------------------------------------------------------------------
# Computing The Tree

weights = [1/main_len for i in range(main_len)] # Setting Initial Weights

def computeWMError(weights):
    min_sum = 0
    dim = 0
    div = 0
    for i in mis_class_list[0][0]:
        min_sum += weights[i]
    min_sum = min_sum / sum(weights)

    for i in range(5):
        for j in range(1000):
            curr_sum = 0
            for k in mis_class_list[i][j]:
                curr_sum += weights[k]
            curr_sum = curr_sum / sum(weights)
            if curr_sum < min_sum:
                min_sum = curr_sum
                dim = i
                div = j
            elif curr_sum == min_sum:
                choice = random.choice([-1,1])
                if choice == -1:
                    min_sum = curr_sum
                    dim = i
                    div = j

    return dim, div, min_sum


# Computing PCA Matrix of Val Set First

val_len = len(Y_val)

data_matrix = []

# Generating data matrix
for i in range(val_len):
    data_matrix.append(X_val[i].flatten().tolist())

data_matrix = np.matrix(data_matrix).T
cent_data_matrix = data_matrix - data_mean  # Centralized data
Y_Matrix_Val = np.matmul(U_matrix_new.T, cent_data_matrix)


# Computing PCA Matrix of Test Set

test_len = len(Y_test)

data_matrix = []

# Generating data matrix
for i in range(test_len):
    data_matrix.append(X_test[i].flatten().tolist())

data_matrix = np.matrix(data_matrix).T
cent_data_matrix = data_matrix - data_mean  # Centralized data
Y_Matrix_Test = np.matmul(U_matrix_new.T, cent_data_matrix)


def computeTree():
    curr_max_acc = 0
    curr_iter = 0
    acc_values = []

    curr_pred = [0 for i in range(val_len)]
    curr_data = []

    for i in range(300):
        dim, div, loss = computeWMError(weights)
        alpha = np.log((1 - loss) / loss)
        curr_data.append([dim, div, alpha])
        
        for j in mis_class_list[dim][div]:
            weights[j] = weights[j] * np.exp(alpha)

        count = 0

        for j in range(val_len):
            if Y_Matrix_Val.item(dim, j) <= decision[dim][div][0]:
                curr_pred[j] += alpha * decision[dim][div][1]
                if(curr_pred[j] == 0):
                    val = random.choice([-1,1])
                else:
                    val = np.sign(curr_pred[j])
            else:
                curr_pred[j] += alpha * decision[dim][div][2]
                if(curr_pred[j] == 0):
                    val = random.choice([-1,1])
                else:
                    val = np.sign(curr_pred[j])

            if val == Y_val[j]:
                count +=1

        accuracy = count / val_len
        if accuracy >= curr_max_acc:
            curr_max_acc = accuracy
            curr_iter = i

        actual_acc = round((accuracy) * 100, 2)
        acc_values.append(actual_acc)

        print(f"Accuracy till {i + 1} Stumps -> {actual_acc}%")

    num_trees = list(range(1, 301))

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(num_trees, acc_values, linestyle='-')
    ax.set_xlabel('Number of Stumps')
    ax.set_ylabel('Accuracy on Val Set')
    ax.set_title('Accuracy on Val Set vs. Number of Stumps')
    ax.set_xticks(num_trees[24::25])
    y_ticks = np.linspace(90, 100, 21)
    ax.set_yticks(y_ticks)
    ax.set_ylim(90, 100)

    plt.show()

    print(f"\nBest Accuracy -> {round(curr_max_acc * 100, 2)}% on Stump {curr_iter + 1}\n")


    count = 0
    curr_pred = [0 for i in range(test_len)]

    for i in range(curr_iter + 1):
        for j in range(test_len):
            if Y_Matrix_Test.item(curr_data[i][0], j) <= decision[curr_data[i][0]][curr_data[i][1]][0]:
                curr_pred[j] += curr_data[i][2] * decision[curr_data[i][0]][curr_data[i][1]][1]
            else:
                curr_pred[j] += curr_data[i][2] * decision[curr_data[i][0]][curr_data[i][1]][2]
    
    for j in range(test_len):
        if(curr_pred[j] == 0):
            val = random.choice([-1,1])
        else:
            val = np.sign(curr_pred[j])

        if val == Y_test[j]:
            count +=1

    accuracy = count / test_len
    actual_acc = round((accuracy) * 100, 2)

    print(f"\nAccuracy for Test Set -> {actual_acc}%")

computeTree()