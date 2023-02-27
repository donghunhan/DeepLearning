# coding: utf-8
import matplotlib.pyplot as plt
#from matplotlib.image import imread
import matplotlib.image as im

#img = imread('../dataset/cactus.png') # 이미지 읽어오기
img=im.imread('../dataset/cactus.png')
plt.imshow(img)

plt.show()
