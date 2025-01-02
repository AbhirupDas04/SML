import numpy as np
from matplotlib import pyplot
from scipy import ndimage
import random

random.seed(19) # To ensure consistent results

data = np.load('mnist.npz')

trainX = data["x_train"]
trainy = data["y_train"]
testX = data["x_test"]
testy = data["y_test"]

new_train = np.isin(trainy, [0, 1, 2])
new_test = np.isin(testy, [0, 1, 2])

X_train, Y_train = trainX[new_train], trainy[new_train] # Making the new dataset out of just classes 0, 1 and 2.
X_test, Y_test = testX[new_test], testy[new_test]
main_len = len(X_train)

#------------------------------------------------------------------------
# Computing PCA Matrix

data_matrix = []
data_positions = [[] for i in range(3)]
data_pos2 = []

# Generating data matrix
for i in range(main_len):
    data_matrix.append(X_train[i].flatten().tolist())
    data_positions[Y_train[i]].append(i)
    data_pos2.append(Y_train[i])

data_matrix = np.matrix(data_matrix).T
data_mean = np.mean(data_matrix, axis = 1)
cent_data_matrix = data_matrix - data_mean  # Centralized data

cov_matrix = np.matmul(cent_data_matrix, cent_data_matrix.T) / 999 # Covariance matrix of centralized data

eig_1 = np.linalg.eig(cov_matrix)
eigenvalues = eig_1[0].real
eigenvectors = eig_1[1].real

indices = eigenvalues.argsort()[::-1]   # Finding the indices corresponding to the eigenvalues in descending order
U_matrix = eigenvectors[:,indices]  # Sorting the eigenvectors based on descending order of their eigenvalues

U_matrix_new = U_matrix[:,:10]
Y_Matrix = np.matmul(U_matrix_new.T, cent_data_matrix)

#------------------------------------------------------------------------
# Computing Gini Index for First Cut

def Gini(len, dim, data, labels, mean):
    result = [[0,0,0],[0,0,0]]
    for i in range(len):
        if data.item((dim, i)) > mean[dim].item((0,0)):
            if labels[i] == 0:
                result[0][0] += 1
            elif labels[i] == 1:
                result[0][1] += 1
            else:
                result[0][2] += 1
        else:
            if labels[i] == 0:
                result[1][0] += 1
            elif labels[i] == 1:
                result[1][1] += 1
            else:
                result[1][2] += 1

    p_00 = result[0][0] / sum(result[0])
    p_10 = result[0][1] / sum(result[0])
    p_20 = result[0][2] / sum(result[0])

    sum_0 = p_00 * (1 - p_00) + p_10 * (1 - p_10) + p_20 * (1 - p_20)

    p_01 = result[1][0] / sum(result[1])
    p_11 = result[1][1] / sum(result[1])
    p_21 = result[1][2] / sum(result[1])

    sum_1 = p_01 * (1 - p_01) + p_11 * (1 - p_11) + p_21 * (1 - p_21)

    gini_index = (sum(result[0])/len) * sum_0 + (sum(result[1])/len) * sum_1

    return gini_index

#------------------------------------------------------------------------
# Computing Gini Index for Second Cut

def Gini2(len, dim, data, labels, first_cut, first_cut_mean, mean_matrix, first_cut_choice):
    result = [[0,0,0],[0,0,0]]

    if dim == first_cut:
        mid = first_cut_mean
    else:
        mid = mean_matrix[dim].item((0,0))

    if first_cut_choice == -1:
        for i in range(len):
            if data.item((dim, i)) < mid and data.item((first_cut,i)) < mean_matrix[first_cut].item((0,0)):
                if labels[i] == 0:
                    result[0][0] += 1
                elif labels[i] == 1:
                    result[0][1] += 1
                else:
                    result[0][2] += 1

            elif data.item((first_cut,i)) < mean_matrix[first_cut].item((0,0)):
                if labels[i] == 0:
                    result[1][0] += 1
                elif labels[i] == 1:
                    result[1][1] += 1
                else:
                    result[1][2] += 1

    else:
        for i in range(len):
            if data.item((dim, i)) < mid and data.item((first_cut,i)) > mean_matrix[first_cut].item((0,0)):
                if labels[i] == 0:
                    result[0][0] += 1
                elif labels[i] == 1:
                    result[0][1] += 1
                else:
                    result[0][2] += 1

            elif data.item((first_cut,i)) > mean_matrix[first_cut].item((0,0)):
                if labels[i] == 0:
                    result[1][0] += 1
                elif labels[i] == 1:
                    result[1][1] += 1
                else:
                    result[1][2] += 1

    p_00 = result[0][0] / (result[0][0] + result[0][1] + result[0][2])
    p_10 = result[0][1] / (result[0][0] + result[0][1] + result[0][2])
    p_20 = result[0][2] / (result[0][0] + result[0][1] + result[0][2])

    sum_0 = p_00 * (1 - p_00) + p_10 * (1 - p_10) + p_20 * (1 - p_20)

    p_01 = result[1][0] / (result[1][0] + result[1][1] + result[1][2])
    p_11 = result[1][1] / (result[1][0] + result[1][1] + result[1][2])
    p_21 = result[1][2] / (result[1][0] + result[1][1] + result[1][2])

    sum_1 = p_01 * (1 - p_01) + p_11 * (1 - p_11) + p_21 * (1 - p_21)

    new_len = result[1][0] + result[1][1] + result[1][2] + result[0][0] + result[0][1] + result[0][2]

    gini_index = ((result[0][0] + result[0][1] + result[0][2])/new_len) * sum_0 + ((result[1][0] + result[1][1] + result[1][2])/new_len) * sum_1

    return gini_index

#-----------------------------------------------------------------------------------------------------------------
# Finally use the above 2 to find the predicted classes for our regions that we shall obtain by the optimal cuts.

def compute(data_matrix, labels):
    mean_arr = np.mean(data_matrix, axis = 1) # Mean of the data matrix
    minGini = 0
    curr_index = 0

    for i in range(10):
        gini_index = Gini(main_len, i, data_matrix, labels, mean_arr) # Computing Gini Index for first cut
        if i==0:
            minGini = gini_index
        else:
            if gini_index < minGini:
                minGini = gini_index
                curr_index = i

    first_cut = curr_index # Cut with minimum Gini Index

    choices = [-1, 1] # -1 denotes going left of this cut, and 1 denotes going right of this cut.
    choice = random.choice(choices)

    sum = 0
    count = 0
    if choice == -1:
        for i in range(main_len):
            if(data_matrix.item((first_cut,i)) < 0): # Computing where the midpoint of the newly created sub-region lies
                sum+= data_matrix.item((0,i))
                count+=1
    else:
        for i in range(main_len):
            if(data_matrix.item((first_cut,i)) > 0):
                sum+= data_matrix.item((0,i))
                count+=1

    new_mean_first_cut = sum / count

    minGini = 0
    curr_index = 0
    for i in range(10):
        gini_index = Gini2(main_len, i, data_matrix, labels, first_cut, new_mean_first_cut, mean_arr, choice) # Computing Gini index for 2nd cut
        if i==0:
            minGini = gini_index
        else:
            if gini_index < minGini:
                minGini = gini_index
                curr_index = i

    second_cut = curr_index # Optimal 2nd cut

    #------------------------------------------------------------------------
    # Counting the no. of train samples in each region to make the classifier
        
    regionSplit = [[0,0,0], [0,0,0],[0,0,0]]

    if choice == -1:
        if first_cut != second_cut:
            for i in range(main_len):
                if data_matrix.item((first_cut, i)) >= mean_arr[first_cut].item((0,0)):
                    regionSplit[0][labels[i]] += 1
                elif data_matrix.item((second_cut, i)) < mean_arr[second_cut].item((0,0)):
                    regionSplit[1][labels[i]] += 1
                else:
                    regionSplit[2][labels[i]] += 1
        else:
            for i in range(main_len):
                if data_matrix.item((first_cut, i)) >= mean_arr[first_cut].item((0,0)):
                    regionSplit[0][labels[i]] += 1
                elif data_matrix.item((first_cut, i)) < new_mean_first_cut:
                    regionSplit[1][labels[i]] += 1
                else:
                    regionSplit[2][labels[i]] += 1

    else:
        if first_cut != second_cut:
            for i in range(main_len):
                if data_matrix.item((first_cut, i)) <= mean_arr[first_cut].item((0,0)):
                    regionSplit[0][labels[i]] += 1
                elif data_matrix.item((second_cut, i)) < mean_arr[second_cut].item((0,0)):
                    regionSplit[1][labels[i]] += 1
                else:
                    regionSplit[2][labels[i]] += 1
        else:
            for i in range(main_len):
                if data_matrix.item((first_cut, i)) <= mean_arr[first_cut].item((0,0)):
                    regionSplit[0][labels[i]] += 1
                elif data_matrix.item((first_cut, i)) < new_mean_first_cut:
                    regionSplit[1][labels[i]] += 1
                else:
                    regionSplit[2][labels[i]] += 1

    regionClass = []
    max = 0
    index = 0
    curr_index = 0
    for i in regionSplit: # Calculates the mode of each of the regions to find out which class a test point would belong to if it had those dimensions
        for j in i:
            if j > max:
                max = j
                curr_index = index
            index+=1
        regionClass.append(curr_index)
        index = 0
        curr_index = 0
        max = 0

    return first_cut, second_cut, new_mean_first_cut, regionClass, choice

first_cut, second_cut, new_mean_first_cut, regionClass, choice = compute(Y_Matrix, Y_train)
mean_arr = np.mean(data_matrix, axis = 1)


if choice == -1:
    if first_cut != second_cut:
        print(f"Splits:-\n\nFirst Cut ->    at {round(mean_arr[first_cut].item((0,0)), 2)} for dim {first_cut}\nSecond Cut ->    at {round(mean_arr[second_cut].item((0,0)), 2)} for dim {second_cut}\n")
        print(f"Three Regions:-\n\n1) >= {round(mean_arr[first_cut].item((0,0)), 2)} for dim {first_cut}\n2) < {round(mean_arr[first_cut].item((0,0)), 2)} for dim {first_cut} and < {round(mean_arr[second_cut].item((0,0)), 2)} for dim {second_cut}\n3) < {round(mean_arr[first_cut].item((0,0)), 2)} for dim {first_cut} and >= {round(mean_arr[second_cut].item((0,0)), 2)} for dim {second_cut}")
    else:
        print(f"Splits:-\n\nFirst Cut ->    at {round(mean_arr[first_cut].item((0,0)), 2)} for dim {first_cut}\nSecond Cut ->    at {round(new_mean_first_cut, 2)} for dim {first_cut}\n")
        print(f"Three Regions:-\n\n1) >= {round(mean_arr[first_cut].item((0,0)), 2)} for dim {first_cut}\n2) < {round(new_mean_first_cut, 2)} for dim {first_cut}\n3) < {round(mean_arr[first_cut].item((0,0)), 2)} and >= {round(new_mean_first_cut, 2)} for dim {first_cut}")

else:
    if first_cut != second_cut:
        print(f"Splits:-\n\nFirst Cut ->    at {round(mean_arr[first_cut].item((0,0)), 2)} for dim {first_cut}\nSecond Cut ->    at {round(mean_arr[second_cut].item((0,0)), 2)} for dim {second_cut}\n")
        print(f"Three Regions:-\n\n1) <= {round(mean_arr[first_cut].item((0,0)), 2)} for dim {first_cut}\n2) > {round(mean_arr[first_cut].item((0,0)), 2)} for dim {first_cut} and < {round(mean_arr[second_cut].item((0,0)), 2)} for dim {second_cut}\n3) > {round(mean_arr[first_cut].item((0,0)), 2)} for dim {first_cut} and >= {round(mean_arr[second_cut].item((0,0)), 2)} for dim {second_cut}")
    else:
        print(f"Splits:-\n\nFirst Cut ->    at {round(mean_arr[first_cut].item((0,0)), 2)} for dim {first_cut}\nSecond Cut ->    at {round(new_mean_first_cut, 2)} for dim {first_cut}\n")
        print(f"Three Regions:-\n\n1) <= {round(mean_arr[first_cut].item((0,0)), 2)} for dim {first_cut}\n2) > {round(mean_arr[first_cut].item((0,0)), 2)} and < {round(new_mean_first_cut, 2)} for dim {first_cut}\n3) >= {round(new_mean_first_cut, 2)} for dim {first_cut}")

#------------------------------------------------------------------------
# Calculating Accuracy of Test Set
    
new_len = len(X_test)

# Computing PCA Matrix of Test Set First

data_matrix = []
data_positions = [[] for i in range(3)]
data_pos2 = []

# Generating data matrix
for i in range(new_len):
    data_matrix.append(X_test[i].flatten().tolist())
    data_positions[Y_test[i]].append(i)
    data_pos2.append(Y_test[i])

data_matrix = np.matrix(data_matrix).T
cent_data_matrix = data_matrix - data_mean  # Centralized data
Y_Matrix_Test = np.matmul(U_matrix_new.T, cent_data_matrix)


def findClass(Matrix, sample_no, first_cut, second_cut, regionClass, new_mean_first_cut, choice, mean_arr):
    if choice == -1:
        if first_cut != second_cut:
            if Matrix.item((first_cut, sample_no)) >= mean_arr[first_cut].item((0,0)):
                val = regionClass[0]
            elif Matrix.item((second_cut, sample_no)) < mean_arr[second_cut].item((0,0)):
                val = regionClass[1]
            else:
                val = regionClass[2]
        else:
            if Matrix.item((first_cut, sample_no)) >= mean_arr[first_cut].item((0,0)):
                val = regionClass[0]
            elif Matrix.item((first_cut, sample_no)) < new_mean_first_cut:
                val = regionClass[1]
            else:
                val = regionClass[2]

    else:
        if first_cut != second_cut:
            if Matrix.item((first_cut, sample_no)) <= mean_arr[first_cut].item((0,0)):
                val = regionClass[0]
            elif Matrix.item((second_cut, sample_no)) < mean_arr[second_cut].item((0,0)):
                val = regionClass[1]
            else:
                val = regionClass[2]
        else:
            if Matrix.item((first_cut, sample_no)) <= mean_arr[first_cut].item((0,0)):
                val = regionClass[0]
            elif Matrix.item((first_cut, sample_no)) < new_mean_first_cut:
                val = regionClass[1]
            else:
                val = regionClass[2]

    return val

actual = [0 for i in range(3)]
predicted = [0 for i in range(3)]
count = 0
val = 0

for i in range(new_len):
    val = findClass(Y_Matrix_Test, i, first_cut, second_cut, regionClass, new_mean_first_cut, choice, mean_arr)

    actual[Y_test[i]] += 1

    if val == Y_test[i]:
        predicted[val] += 1
        count +=1

accuracy_values = []
print(f"\nAccuracy -> {round((count / new_len) * 100, 2)}%\n")
for i in range(3):
    print(f"Class {i} -> {round((predicted[i] / actual[i]) * 100, 2)}%")
    accuracy_values.append(round((predicted[i] / actual[i]) * 100, 2))

class_labels = list(range(3))

pyplot.bar(class_labels, accuracy_values, color='green')
pyplot.xlabel('Class')
pyplot.ylabel('Accuracy')
pyplot.title('Accuracy for Each Class')
pyplot.xticks(class_labels)
pyplot.ylim(0, 100)

pyplot.show()   # Plotting the Accuracy

#------------------------------------------------------------------------
# Creating 5 datasets for Bagging

data = (Y_Matrix.T).tolist()

D1 = []
D1_labels = []

D2 = []
D2_labels = []

D3 = []
D3_labels = []

D4 = []
D4_labels = []

D5 = []
D5_labels = []

for i in range(main_len):
    num = random.randint(0, main_len - 1)
    D1.append(data[num])
    D1_labels.append(Y_train[num])

    num = random.randint(0, main_len - 1)
    D2.append(data[num])
    D2_labels.append(Y_train[num])

    num = random.randint(0, main_len - 1)
    D3.append(data[num])
    D3_labels.append(Y_train[num])

    num = random.randint(0, main_len - 1)
    D4.append(data[num])
    D4_labels.append(Y_train[num])

    num = random.randint(0, main_len - 1)
    D5.append(data[num])
    D5_labels.append(Y_train[num])

D1 = np.matrix(D1).T
D2 = np.matrix(D2).T
D3 = np.matrix(D3).T
D4 = np.matrix(D4).T
D5 = np.matrix(D5).T

#------------------------------------------------------------------------
# Generating trees for each of the new datasets

first_cut_list = []
second_cut_list = []
first_mean_list = []
regionClass_list = []
choice_list = []

f_cut, s_cut, f_mean, reglist, choice = compute(D1, D1_labels)
first_cut_list.append(f_cut)
second_cut_list.append(s_cut)
first_mean_list.append(f_mean)
regionClass_list.append(reglist)
choice_list.append(choice)

f_cut, s_cut, f_mean, reglist, choice = compute(D2, D2_labels)
first_cut_list.append(f_cut)
second_cut_list.append(s_cut)
first_mean_list.append(f_mean)
regionClass_list.append(reglist)
choice_list.append(choice)

f_cut, s_cut, f_mean, reglist, choice = compute(D3, D3_labels)
first_cut_list.append(f_cut)
second_cut_list.append(s_cut)
first_mean_list.append(f_mean)
regionClass_list.append(reglist)
choice_list.append(choice)

f_cut, s_cut, f_mean, reglist, choice = compute(D4, D4_labels)
first_cut_list.append(f_cut)
second_cut_list.append(s_cut)
first_mean_list.append(f_mean)
regionClass_list.append(reglist)
choice_list.append(choice)

f_cut, s_cut, f_mean, reglist, choice = compute(D5, D5_labels)
first_cut_list.append(f_cut)
second_cut_list.append(s_cut)
first_mean_list.append(f_mean)
regionClass_list.append(reglist)
choice_list.append(choice)

mean_arr = []
mean_arr.append(np.mean(D1, axis=1))
mean_arr.append(np.mean(D2, axis=1))
mean_arr.append(np.mean(D3, axis=1))
mean_arr.append(np.mean(D4, axis=1))
mean_arr.append(np.mean(D5, axis=1))

#------------------------------------------------------------------------
# Computing Accuracy

actual = [0 for i in range(3)]
predicted = [0 for i in range(3)]
count = 0
val = 0

for i in range(new_len):
    valList = []
    for j in range(5):
        valList.append(findClass(Y_Matrix_Test, i, first_cut_list[j], second_cut_list[j], regionClass_list[j], first_mean_list[j], choice_list[j], mean_arr[j]))

    modeList = [0,0,0]
    for j in valList:
        modeList[j] += 1

    index = 0
    curr_max = 0
    for j in range(3):
        if modeList[j] > curr_max:
            index = j
            curr_max = modeList[j]
        elif modeList[j] == curr_max:
            index = random.choice([j , index])
            curr_max = modeList[index]

    val = index # This is the mode

    actual[Y_test[i]] += 1

    if val == Y_test[i]:
        predicted[val] += 1
        count +=1

accuracy_values = []
print(f"\nAccuracy with Bagging -> {round((count / new_len) * 100, 2)}%\n")
for i in range(3):
    print(f"Class {i} -> {round((predicted[i] / actual[i]) * 100, 2)}%")
    accuracy_values.append(round((predicted[i] / actual[i]) * 100, 2))

class_labels = list(range(3))

pyplot.bar(class_labels, accuracy_values, color='red')
pyplot.xlabel('Class')
pyplot.ylabel('Accuracy')
pyplot.title('Accuracy for Each Class with Bagging')
pyplot.xticks(class_labels)
pyplot.ylim(0, 100)

pyplot.show()   # Plotting the Accuracy