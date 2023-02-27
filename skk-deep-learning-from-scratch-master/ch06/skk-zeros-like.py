import numpy
# 0으로 채우되 data type도 반영한다.
# 실수면 소수점이 들어간다.

i_list=[[1,2,3],[4,5,6]]
zero_array=numpy.zeros_like(i_list)
print("zero_array:")
print(zero_array)

#[[0 0 0]
# [0 0 0]]

f_list=[[1,2,3.],[4,5,6]]
zero_array=numpy.zeros_like(f_list)
print("zero_array_float data type:")
print(zero_array)

#[[0. 0. 0.]
# [0. 0. 0.]]