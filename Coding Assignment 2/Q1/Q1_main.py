from numpy import *
from matplotlib import pyplot
from scipy import ndimage

data = load('mnist.npz')

trainX = data["x_train"]
trainy = data["y_train"]
testX = data["x_test"]
testy = data["y_test"]

data_matrix = [[] for i in range(10)]
data_positions = [[] for i in range(10)]

for i in range(60000):
    temp = ndimage.rotate(trainX[i],random.uniform(-10,10), reshape = False)    # Rotating the sample to make the covariance matrices of classes not singular.
    temp = (temp.flatten() / 255) + 0.0001 * random.uniform(0, 1, 784)  # Normalizing the data and adding uniform noise
    temp = temp * 255 # Scaling the data back up
    data_matrix[trainy[i]].append(temp.tolist())
    # data_matrix[trainy[i]].append(trainX[i].flatten().tolist())
    data_positions[trainy[i]].append(i)

for i in range(10):
    data_matrix[i] = matrix(data_matrix[i]).T

for j in range(10):
    for i in range(5):
        pyplot.subplot(10,5, 5*j + i + 1)
        pyplot.imshow(trainX[data_positions[j][i]], cmap=pyplot.get_cmap('gray'))   # Displaying the data
        pyplot.xticks([])
        pyplot.yticks([])

pyplot.suptitle("Visualizations of 5 samples Chosen from each Class")
pyplot.show()
            
# ----------------------------------------------------------------------------------------------------

# This function computes the mean of all the 10 samples and returns them all in an array
def compute_mean(data_matrix):
    result_matrix = []
    for i in range(10):
        arr = mean(data_matrix[i], axis = 1)
        result_matrix.append(arr)
        
    return result_matrix

mean_main = compute_mean(data_matrix)   # Computing the means


# This function computes the covariance matrices of all the 10 samples and returns them all in an array
def compute_variance(data_matrix):
    result_matrix = []

    for i in range(10):
        temp = data_matrix[i] - mean_main[i]
        temp_2 = temp.T
        result = matmul(temp, temp_2)
        result = result / (shape(data_matrix[i])[1] - 1)
        result_matrix.append(result)

    return result_matrix

var = compute_variance(data_matrix) # Computing the covariance matrices
spec_arr = [[] for i in range(10)]  # This array stores all the components required to calculate the discriminant for QDA

for i in range(10):
    var[i] = add(var[i], identity(784) / 1000000)
    det = linalg.slogdet(var[i])
    spec_arr[i].append(-0.5 * det[0] * det[1])
    inv = linalg.pinv(var[i])
    spec_arr[i].append(inv)
    spec_arr[i].append(matmul(mean_main[i].T, inv))
    spec_arr[i].append(-0.5 * matmul(matmul(mean_main[i].T, inv), mean_main[i]).item((0,0)))
    spec_arr[i].append(log(len(data_positions[i]) / 60000))

# This function takes any sample and calculates the discriminant for any category of our choice
def compute_discriminant(sample, category):
    main_arr = spec_arr[category]

    term_1 = main_arr[0]
    term_2 = -0.5 * matmul(matmul(sample.T,main_arr[1]), sample).item((0,0))
    term_3 = matmul(main_arr[2], sample).item((0,0))
    term_4 = main_arr[3]
    term_5 = main_arr[4]

    return term_1 + term_2 + term_3 + term_4 + term_5   # The final discriminant

# We compute all 10 discriminants and then find the class that returns the greatest discriminant
def compute_class(sample):
    arr = []
    for i in range(10):
        arr.append(compute_discriminant(sample, i))
    return argmax(arr)

actual = [0 for i in range(10)]
expected = [0 for i in range(10)]
data_vector = [[0] for j in range(784)]
count = 0
for i in range(10000):
    # print(i)
    data_vector = matrix(testX[i]).flatten().T

    val = compute_class(data_vector)
    expected[val] += 1

    if val == testy[i]:
        actual[val] += 1
        count += 1

print(f"\nOverall Accuracy -> {count / 100}%")

accuracy_values = []
for i in range(10):
    accuracy = round((actual[i] / expected[i]) * 100, 2)    # Computing the Accuracy
    accuracy_values.append(accuracy)
    print(f"Sample {i} -> {accuracy}%")

class_labels = list(range(10))

pyplot.bar(class_labels, accuracy_values, color='blue')
pyplot.xlabel('Class')
pyplot.ylabel('Accuracy')
pyplot.title('Accuracy for Each Class')
pyplot.xticks(class_labels)
pyplot.ylim(0, 100)

pyplot.show()   # Plotting the Accuracy