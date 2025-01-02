import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
import random

# from keras.datasets import mnist

# (trainX, trainy), (testX, testy) = mnist.load_data()

np.random.seed(80)

data = np.load('mnist.npz')

trainX = data["x_train"]
trainy = data["y_train"]
testX = data["x_test"]
testy = data["y_test"]

new_train = np.isin(trainy, [0, 1])
new_test = np.isin(testy, [0, 1])

X_train, Y_train = trainX[new_train], trainy[new_train] # Making the new dataset out of just classes 0 and 1.
X_test, Y_test = testX[new_test], testy[new_test]

first_indices = np.where(Y_train == 0)[0]
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

sorted_class_list = [] # Indices of the points to the left and right of point for every split
midpoint_list = [] # Stores midpoints

# This precomputes the mis-classified points in every split
def preCompute(data, real_indices):
    temp_list = []
    mid_temp_list = []
    low = data[0]
    high = data[main_len - 1]

    interval = (high - low) / 1001
    midpoint = low + interval

    for i in range(1000):
        mid_temp_list.append(midpoint)

        indices = np.where(data <= midpoint)[0]
        left_indices = real_indices[indices] # Indices in original array corresponding to left half of this point

        indices = np.where(data > midpoint)[0]
        right_indices = real_indices[indices] # Indices in original array corresponding to right half of this point

        final_indices = [left_indices, right_indices]

        temp_list.append(final_indices)

        midpoint = midpoint + interval

    sorted_class_list.append(temp_list)
    midpoint_list.append(mid_temp_list)

def performInitialization(data):
    for i in range(5):
        indices = np.argsort(np.ravel(data[i, :]))
        data2 = data[:, indices]

        preCompute(np.ravel(data2[i, :]), indices)

performInitialization(Y_Matrix)

#------------------------------------------------------------------------
# Performing PCA on Val And Test

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

#------------------------------------------------------------------------
# Computing the Tree

def ComputeTree():
    labels = Y_train
    labels = labels.astype(np.int32)

    labels_storage = []
    
    flag = 1

    decision_list = []
    mse_list = []
    residual_list = []
    min_mse = 0
    best_iter = 0

    label_mean = np.mean(labels)
    val_pred = [label_mean for i in range(val_len)]
    for i in range(main_len):
        temp = labels[i] - label_mean
        residual_list.append(temp)
        if temp == 0:
            labels[i] = random.choice([-1,1])
        else:
            labels[i] = np.sign(temp)
    
    min_ssr = 0
    min_dim = 0
    min_div = 0
    dec_left = 0
    dec_right = 0
    midpoint = 0

    for i in range(300):
        if flag == 1:
            flag2 = 0
            for j in labels_storage:
                if np.array_equal(j[0], labels):
                    min_dim = j[1]
                    midpoint = j[2]
                    dec_left = j[3]
                    dec_right = j[4]
                    flag2 = 1
                    break
            if flag2 == 0:
                min_ssr = 0
                min_dim = 0
                min_div = 0
                dec_left = 0
                dec_right = 0
                midpoint = 0

                for j in range(5):
                    for k in range(1000):
                        mean_left = np.mean(labels[sorted_class_list[j][k][0]])
                        mean_right = np.mean(labels[sorted_class_list[j][k][1]])
                        ssr = 0

                        for a in sorted_class_list[j][k][0]:
                            ssr += (labels[a] - mean_left) ** 2

                        for a in sorted_class_list[j][k][1]:
                            ssr += (labels[a] - mean_right) ** 2

                        if j == 0 and k == 0:
                            min_ssr = ssr
                        else:
                            if ssr < min_ssr:
                                min_ssr = ssr
                                min_dim = j
                                min_div = k
                                dec_left = mean_left
                                dec_right = mean_right
                                midpoint = midpoint_list[j][k]
                            elif ssr == min_ssr:
                                choice = random.choice([0,1])
                                if choice == 0:
                                    min_dim = j
                                    min_div = k
                                    dec_left = mean_left
                                    dec_right = mean_right
                                    midpoint = midpoint_list[j][k]

                labels_storage.append([labels.copy(),min_dim, midpoint, dec_left, dec_right])

        decision_list.append([min_dim, midpoint, dec_left, dec_right])
        flag = -1

        for j in range(main_len):
            if Y_Matrix.item((min_dim, j)) <= midpoint:
                residual_list[j] = residual_list[j] - 0.01 * dec_left
                if residual_list[j] == 0:
                    temp = random.choice([-1,1])
                    if labels[j] != temp:
                        flag = 1
                    labels[j] = temp
                else:
                    if np.sign(residual_list[j]) != labels[j]:
                        flag = 1
                    labels[j] = np.sign(residual_list[j])
            else:
                residual_list[j] = residual_list[j] - 0.01 * dec_right
                if residual_list[j] == 0:
                    temp = random.choice([-1,1])
                    if labels[j] != temp:
                        flag = 1
                    labels[j] = temp
                else:
                    if np.sign(residual_list[j]) != labels[j]:
                        flag = 1
                    labels[j] = np.sign(residual_list[j])

        error = 0

        for j in range(val_len):
            if Y_Matrix_Val.item((min_dim, j)) <= midpoint:
                val_pred[j] += 0.01 * dec_left
            else:
                val_pred[j] += 0.01 * dec_right
            
            error += (val_pred[j] - Y_val[j]) ** 2

        mse = error / val_len
        if i == 0:
            min_mse = mse
        else:
            if mse <= min_mse:
                min_mse = mse
                best_iter = i

        print(f"MSE On Stump {i + 1} -> {mse}")
        mse_list.append(mse)

    print(f"\nSmallest MSE for Stump {best_iter + 1} -> {min_mse}\n")



    num_trees = list(range(1, 301))

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(num_trees, mse_list, linestyle='-')
    ax.set_xlabel('Number of Stumps')
    ax.set_ylabel('MSE on Val Set')
    ax.set_title('MSE on Val Set vs. Number of Stumps')
    ax.set_xticks(num_trees[24::25])
    # y_ticks = np.linspace(99, 100, 21)
    # ax.set_yticks(y_ticks)
    # ax.set_ylim(99, 100)

    plt.show()



    test_pred = [label_mean for i in range(test_len)]

    for i in range(best_iter + 1):
        for j in range(test_len):
            if Y_Matrix_Test.item((decision_list[i][0], j)) <= decision_list[i][1]:
                test_pred[j] += 0.01 * decision_list[i][2]
            else:
                test_pred[j] += 0.01 * decision_list[i][3]

    error = 0
    
    for j in range(test_len):
        error += (test_pred[j] - Y_test[j]) ** 2

    mse = error / test_len

    print(f"\nMSE for Test Set -> {mse}")

ComputeTree()