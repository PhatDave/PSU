import matplotlib.pyplot as plt
import matplotlib.image as mpimg

img = mpimg.imread('tiger.png')

for i in range(len(img)):
	for j in range(len(img[i])):
		img[i, j] = [img[i, j, 0] + 0.2, img[i, j, 1] + 0.2, img[i, j, 2] + 0.2]
		for k in range(img[i, j]):
			if img[i, j, k] > 1.0:
				img[i, j, k] = 1.0

imgplot = plt.imshow(img)
plt.show()
