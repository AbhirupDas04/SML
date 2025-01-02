import numpy as np

arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# Specify the row index and axis=1 for row-wise mean
mean_of_row_1 = np.mean(arr, axis=1)[0]  # Access mean for row 1

print(mean_of_row_1)  # Output: 5.0