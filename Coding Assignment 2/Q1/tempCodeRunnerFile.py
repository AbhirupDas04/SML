var_1 = compute_variance(data_matrix_1)
# # var_1 = add(var_1,identity(784) / 10)
# # print(linalg.det(var_1))
# var_2 = compute_variance(data_matrix_2)
# var_3 = compute_variance(data_matrix_3)
# var_4 = compute_variance(data_matrix_4)
# var_5 = compute_variance(data_matrix_5)
# var_6 = compute_variance(data_matrix_6)
# var_7 = compute_variance(data_matrix_7)
# var_8 = compute_variance(data_matrix_8)
# var_9 = compute_variance(data_matrix_9)
# var_10 = compute_variance(data_matrix_10)

# if(linalg.det(var_1) == 0):
#     var_1 = add(var_1, identity(784) / 100000)
#     inv_1 = linalg.pinv(var_1)
# else:
#     inv_1 = linalg.inv(var_1)
# det_1 = linalg.det(var_1)
# spec_1 = matmul(mean_1.T, inv_1)
# term_3_1 = -0.5 * matmul(matmul(mean_1.T, inv_1), mean_1).item((0,0))

# if(linalg.det(var_2) == 0):
#     var_2 = add(var_2, identity(784) / 100000)
#     inv_2 = linalg.pinv(var_2)
# else:
#     inv_2 = linalg.inv(var_2)
# det_2 = linalg.det(var_2)
# spec_2 = matmul(mean_2.T, inv_2)
# term_3_2 = -0.5 * matmul(matmul(mean_2.T, inv_2), mean_2).item((0,0))

# if(linalg.det(var_3) == 0):
#     var_3 = add(var_3, identity(784) / 100000)
#     inv_3 = linalg.pinv(var_3)
# else:
#     inv_3 = linalg.inv(var_3)
# det_3 = linalg.det(var_3)
# spec_3 = matmul(mean_3.T, inv_3)
# term_3_3 = -0.5 * matmul(matmul(mean_3.T, inv_3), mean_3).item((0,0))

# if(linalg.det(var_4) == 0):
#     var_4 = add(var_4, identity(784) / 100000)
#     inv_4 = linalg.pinv(var_4)
# else:
#     inv_4 = linalg.inv(var_4)
# det_4 = linalg.det(var_4)
# spec_4 = matmul(mean_4.T, inv_4)
# term_3_4 = -0.5 * matmul(matmul(mean_4.T, inv_4), mean_4).item((0,0))

# if(linalg.det(var_5) == 0):
#     var_5 = add(var_5, identity(784) / 100000)
#     inv_5 = linalg.pinv(var_5)
# else:
#     inv_5 = linalg.inv(var_5)
# det_5 = linalg.det(var_5)
# spec_5 = matmul(mean_5.T, inv_5)
# term_3_5 = -0.5 * matmul(matmul(mean_5.T, inv_5), mean_5).item((0,0))

# if(linalg.det(var_6) == 0):
#     var_6 = add(var_6, identity(784) / 100000)
#     inv_6 = linalg.pinv(var_6)
# else:
#     inv_6 = linalg.inv(var_6)
# det_6 = linalg.det(var_6)
# spec_6 = matmul(mean_6.T, inv_6)
# term_3_6 = -0.5 * matmul(matmul(mean_6.T, inv_6), mean_6).item((0,0))

# if(linalg.det(var_7) == 0):
#     var_7 = add(var_7, identity(784) / 100000)
#     inv_7 = linalg.pinv(var_7)
# else:
#     inv_7 = linalg.inv(var_7)
# det_7 = linalg.det(var_7)
# spec_7 = matmul(mean_7.T, inv_7)
# term_3_7 = -0.5 * matmul(matmul(mean_7.T, inv_7), mean_7).item(0,0)

# if(linalg.det(var_8) == 0):
#     var_8 = add(var_8, identity(784) / 100000)
#     inv_8 = linalg.pinv(var_8)
# else:
#     inv_8 = linalg.inv(var_8)
# det_8 = linalg.det(var_8)
# spec_8 = matmul(mean_8.T, inv_8)
# term_3_8 = -0.5 * matmul(matmul(mean_8.T, inv_8), mean_8).item((0,0))

# if(linalg.det(var_9) == 0):
#     var_9 = add(var_9, identity(784) / 100000)
#     inv_9 = linalg.pinv(var_9)
# else:
#     inv_9 = linalg.inv(var_9)
# det_9 = linalg.det(var_9)
# spec_9 = matmul(mean_9.T, inv_9)
# term_3_9 = -0.5 * matmul(matmul(mean_9.T, inv_9), mean_9).item((0,0))

# if(linalg.det(var_10) == 0):
#     var_10 = add(var_10, identity(784) / 100000)
#     inv_10 = linalg.pinv(var_10)
# else:
#     inv_10 = linalg.inv(var_10)
# det_10 = linalg.det(var_10)
# spec_10 = matmul(mean_10.T, inv_10)
# term_3_10 = -0.5 * matmul(matmul(mean_10.T, inv_10), mean_10).item((0,0))


# def compute_discriminant(sample, mean, covariance_matrix, inverse, determinant, spec, term_4):
#     term_1 = -0.5 * log(determinant)
#     term_2 = -0.5 * matmul(matmul(sample.T,inverse), sample).item((0,0))
#     term_3 = matmul(spec, sample).item((0,0))

#     return term_2 + term_3 + term_4 # Not taking the term ln(P(w_i)) as we assume equi-priors.


# # def unbiased_covariance_matrix(data):
# #     # Calculate the covariance matrix using numpy's cov function
# #     cov_matrix = cov(data, rowvar=False, ddof= 1)

# #     return cov_matrix.tolist()

# def compute_class(sample):
#     arr = []
#     arr.append(compute_discriminant(sample, mean_1, var_1, inv_1, det_1, spec_1, term_3_1))
#     arr.append(compute_discriminant(sample, mean_2, var_2, inv_2, det_2, spec_2, term_3_2))
#     arr.append(compute_discriminant(sample, mean_3, var_3, inv_3, det_3, spec_3, term_3_3))
#     arr.append(compute_discriminant(sample, mean_4, var_4, inv_4, det_4, spec_4, term_3_4))
#     arr.append(compute_discriminant(sample, mean_5, var_5, inv_5, det_5, spec_5, term_3_5))
#     arr.append(compute_discriminant(sample, mean_6, var_6, inv_6, det_6, spec_6, term_3_6))
#     arr.append(compute_discriminant(sample, mean_7, var_7, inv_7, det_7, spec_7, term_3_7))
#     arr.append(compute_discriminant(sample, mean_8, var_8, inv_8, det_8, spec_8, term_3_8))
#     arr.append(compute_discriminant(sample, mean_9, var_9, inv_9, det_9, spec_9, term_3_9))
#     arr.append(compute_discriminant(sample, mean_10, var_10, inv_10, det_10, spec_10, term_3_10))
#     return argmax(arr)


# # test = unbiased_covariance_matrix(array(data_matrix_1).T)

# # for i in range(784):
# #     for j in range(784):
# #         if(test[i][j] != var_1[i][j]):
# #             print(test[i][j], var_1[i][j])

# # data_vector = [[0] for i in range(784)]
# # for i in range(28):
# #     for j in range(28):
# #         data_vector[28*i + j][0] = testX[0][i][j]

# #print(compute_discriminant(matrix(data_vector), mean_1, var_1))
# # print("hey")
# # print(compute_class(matrix(data_vector)))
# # print("die")
# # print(testy[0])

# # my_ans = []
# # # test2 = [0 for i in range(10)]
# # # test3 = [0 for i in range(10)]
# # test4 = [0 for i in range(10)]
# # test5 = [0 for i in range(10)]
# # data_vector = [[0] for j in range(784)]
# # count = 0
# # for i in range(10000):
# #     print(i)
# #     data_vector = matrix(testX[i]).flatten().T

# #     val = compute_class(data_vector)
# #     my_ans.append(val)
# #     test5[val] += 1
# #     if val == testy[i]:
# #         test4[val] += 1
# #         count += 1
#     # else:

#         # test2[val] += 1
#         # test3[testy[i]] += 1

# # print(count / 100)

# # print(test2)
# # print(test3)


# #print(compute_discriminant(matrix(data_vector), mean_1, var_1))
# # print("hey")
# # print(compute_class(matrix(data_vector)))
# # print("die")
# # print(testy[0])