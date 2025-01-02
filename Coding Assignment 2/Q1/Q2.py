from numpy import *
from matplotlib import pyplot

data = load('mnist.npz')

trainX = data["x_train"]
trainy = data["y_train"]
testX = data["x_test"]
testy = data["y_test"]

data_matrix = []
data_positions = [[] for i in range(10)]
data_pos2 = []
count_matrix = [0 for i in range(10)]
count = 0

# Generating data matrix
for i in range(60000):
    if(count == 1000):
        break
    if(count_matrix[trainy[i]] == 100):
        continue
    else:
        data_matrix.append(trainX[i].flatten().tolist())
        data_positions[trainy[i]].append(i)
        data_pos2.append(trainy[i])
        count_matrix[trainy[i]] += 1
        count+=1

data_matrix = matrix(data_matrix).T
data_mean = mean(data_matrix, axis = 1)
cent_data_matrix = data_matrix - data_mean  # Centralized data

cov_matrix = matmul(cent_data_matrix, cent_data_matrix.T) / 999 # Covariance matrix of centralized data

eig_1 = linalg.eig(cov_matrix)
eigenvalues = eig_1[0].real
eigenvectors = eig_1[1].real

indices = eigenvalues.argsort()[::-1]   # Finding the indices corresponding to the eigenvalues in descending order
U_matrix = eigenvectors[:,indices]  # Sorting the eigenvectors based on descending order of their eigenvalues

Y_Matrix = matmul(U_matrix.T, cent_data_matrix)
X_Recon = matmul(U_matrix, Y_Matrix)    # Reconstructing the X matrix

def compute_MSE(Original, Recon):
    sum = 0
    Original = Original.tolist()
    Recon = Recon.tolist()
    for i in range(784):
        for j in range(1000):
            sum += (Original[i][j] - Recon[i][j])**2
    return sum

X_Recon_2 = X_Recon + data_mean

print(f"MSE for centralized X -> {compute_MSE(cent_data_matrix, X_Recon)}")
print(f"MSE for original X -> {compute_MSE(data_matrix, X_Recon_2)}")

U_arr = []
Y_arr = []

for i in [5,10,20]:
    U_matrix_new = U_matrix[:,:i]
    U_arr.append(U_matrix_new)
    Y_Matrix = matmul(U_matrix_new.T, cent_data_matrix)
    Y_arr.append(Y_Matrix)

    UY_mat = matmul(U_matrix_new, Y_Matrix)
    UY_mat = UY_mat + data_mean

    new_data_matrix = []
    for j in range(1000):
        new_data_matrix.append(UY_mat[:,j:j+1].reshape(28,28))  # Arranging the data into their respective classes

    # Plotting 5 samples from each class
    for j in range(10):
        for k in range(5):
            pyplot.subplot(10,5, 5*j + k + 1)
            pyplot.imshow(new_data_matrix[data_positions[j][k]], cmap=pyplot.get_cmap('gray'))
            pyplot.xticks([])
            pyplot.yticks([])

    pyplot.suptitle(f"Visualizations of 5 samples from each Class When p = {i}")
    pyplot.show()

def compute_mean(data_matrix):
    result_matrix = []
    for i in range(10):
        arr = mean(data_matrix[i], axis = 1)
        result_matrix.append(arr)
        
    return result_matrix
            
def compute_variance(data_matrix):
    result_matrix = []
    for i in range(10):
        arr = cov(data_matrix[i].T, rowvar = False, ddof = 1)
        result_matrix.append(arr)

    return result_matrix

def compute_discriminant(sample, category, spec_arr):
    main_arr = spec_arr[category]

    term_1 = main_arr[0]
    term_2 = -0.5 * matmul(matmul(sample.T,main_arr[1]), sample).item((0,0))
    term_3 = matmul(main_arr[2], sample).item((0,0))
    term_4 = main_arr[3]
    term_5 = main_arr[4]

    return term_1 + term_2 + term_3 + term_4 + term_5

def compute_class(sample, spec_arr):
    arr = []
    for i in range(10):
        arr.append(compute_discriminant(sample, i, spec_arr))
    return argmax(arr)

new_testX = []

for i in range(10000):
    new_testX.append(matrix(testX[i]).flatten().tolist()[0])

new_testX = matrix(new_testX).T
new_testX = new_testX - mean(new_testX, axis = 1)   # Centralizing the Test Set

# Doing QDA for p = 5, 10 and 20
for a in [5, 10, 20]:
    if(a == 5):
        var = 0
    elif a == 10:
        var = 1
    else:
        var = 2

    new_mat_qda = [[] for i in range(10)]
    for i in range(1000):
        new_mat_qda[data_pos2[i]].append(Y_arr[var][:,i:i+1].T.tolist()[0])

    for i in range(10):
        new_mat_qda[i] = matrix(new_mat_qda[i]).T

    new_mean = compute_mean(new_mat_qda)
    new_var = compute_variance(new_mat_qda)

    spec_arr = [[] for i in range(10)]

    # Computing terms needed for discriminant analysis
    for i in range(10):
        spec_arr[i].append(-0.5 * log(linalg.det(new_var[i])))
        inv = linalg.inv(new_var[i])
        spec_arr[i].append(inv)
        spec_arr[i].append(matmul(new_mean[i].T, inv))
        spec_arr[i].append(-0.5 * matmul(matmul(new_mean[i].T, inv), new_mean[i]).item((0,0)))
        spec_arr[i].append(log(len(data_positions[i]) / 1000))

    new_Y = matmul(U_arr[var].T, new_testX)

    actual = [0 for i in range(10)]
    expected = [0 for i in range(10)]
    count = 0
    for i in range(10000):
        data_vector = new_Y[:, i:i+1]

        val = compute_class(data_vector, spec_arr)
        expected[val] += 1

        if val == testy[i]:
            actual[val] += 1
            count += 1

    print(f"\nFinal Accuracy when p = {a} -> {count / 100}%")

    accuracy_values = []
    for i in range(10):
        accuracy = round((actual[i] / expected[i]) * 100, 2)
        accuracy_values.append(accuracy)
        print(f"Sample {i} -> {accuracy}%")

    class_labels = list(range(10))

    # Plotting accuracies of each class
    pyplot.bar(class_labels, accuracy_values, color='blue')
    pyplot.xlabel('Class')
    pyplot.ylabel('Accuracy')
    pyplot.title(f'Accuracy for Each Class When p = {a}')
    pyplot.xticks(class_labels)
    pyplot.ylim(0, 100)

    pyplot.show()