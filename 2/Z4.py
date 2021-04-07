import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2


class bogecMali:
	factor = 0.5
	imgOriginal = cv2.imread("tiger.png")
	imgCopy = imgOriginal.copy().tolist()

	def __init__(self):
		pass

	def brighten(self):
		img = self.imgOriginal.copy()

		for i in range(len(self.imgOriginal)):
			for j in range(len(self.imgOriginal[i])):
				img[i, j] = [img[i, j, 0] + self.factor * 10,
				             img[i, j, 1] + self.factor * 10,
				             img[i, j, 2] + self.factor * 10]

		plt.imshow(img)
		plt.show()

	def rotate(self):
		# imgRotate = cv2.rotate(imgRotate, 0)
		img = []
		for j in range(self.imgCopy[0].__len__()):
			temp = []
			for i in range(self.imgCopy.__len__() - 1, 0, -1):
				temp.append(self.imgCopy[i][j])
			img.append(temp)

		plt.imshow(np.array(img))
		plt.show()

	def flip(self):
		img = []
		for i in range(self.imgCopy.__len__()):
			temp = []
			for j in range(self.imgCopy[i].__len__() - 1, 0, -1):
				temp.append(self.imgCopy[i][j])
			img.append(temp)

		plt.imshow(img)
		plt.show()

	def smolify(self):
		scale = 10
		img = []
		for i in range(0, int(self.imgCopy.__len__()), scale):
			temp = []
			for j in range(0, int(self.imgCopy[i].__len__()), scale):
				temp.append(self.imgCopy[i][j])
			img.append(temp)

		plt.imshow(img)
		plt.show()

	def fugly(self):
		imgFugly = self.imgOriginal.copy()
		for i in range(int(len(imgFugly))):
			for j in range(int(len(imgFugly[i]) / 4), int(len(imgFugly[i])), 1):
				imgFugly[i, j] = [0, 0, 0]
		plt.imshow(imgFugly)
		plt.show()


first = bogecMali()
first.brighten()
first.rotate()
first.flip()
first.smolify()
first.fugly()
