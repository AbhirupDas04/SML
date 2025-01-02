# def unbiased_covariance_matrix(data):
#     # Calculate the covariance matrix using numpy's cov function
#     cov_matrix = cov(data, rowvar=False, ddof= 1)

#     return cov_matrix.tolist()

# test = unbiased_covariance_matrix(data_matrix[0].T)

# for i in range(784):
#     for j in range(784):
#         if(test[i][j] != var_1[i][j]):
#             print(var_1[i][j], test[i][j])

#print(compute_variance(data_matrix)[0])