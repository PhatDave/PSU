import matplotlib.pyplot as plt
import matplotlib.image as mpimg
img = mpimg.imread('tiger.png')

print(type(img))
for i in range(len(img)):
	for j in range(len(i)):
		print(img[i][j])

imgplot = plt.imshow(img)
plt.show()