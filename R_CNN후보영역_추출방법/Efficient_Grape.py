import skimage.segmentation
from matplotlib import pyplot as plt
import cv2

img = cv2.imread('parrot.png')
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
res1 = skimage.segmentation.felzenszwalb(img_rgb, scale=100, min_size = 20) # scale = K의값, min_size = 최소 픽셀사이즈
res2 = skimage.segmentation.felzenszwalb(img_rgb, scale=1000,min_size = 20)

fig = plt.figure(figsize=(12, 5))
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)
ax1.imshow(res1); ax1.set_xlabel("k=100")
ax2.imshow(res2); ax2.set_xlabel("k=1000")
#fig.suptitle("Graph based image segmentation")
#plt.tight_layout()
plt.show()