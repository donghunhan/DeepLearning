import numpy as np
#rand 0~1 사이의 임의 값을 균일한 분포로 생성
#np.random.rand(*x.shape)로 *뒤에 shape이 오면 shape 모양대로 난수를 생성한다.
#여기서는 2행 3열

i_list=[[1,2,3],[4,5,6]]
x=np.array(i_list)
mask1=np.random.rand(*x.shape)
print('mask1:', mask1)
#mask1: [[0.78358271 0.66649543 0.27129703]
# [0.76488506 0.21866732 0.07485752]]

mask2=np.random.rand(*x.shape) > 0.5
print('mask2:', mask2)

# mask2: [[False  True False]
# [False False  True]]
x=x*mask2
print('x', x)

# x [[0 2 0]
# [0 0 6]]