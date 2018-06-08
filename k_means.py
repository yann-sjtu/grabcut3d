# K-means
import numpy as np
import random
# from matplotlib import pyplot as plt

def get_size(A):
	return list(A.shape)[:2]

class kmeans:
	def __init__(self, A, n = 2, max_iter = 10):
		'''n: The amount of centers'''
		self.A = A.flatten()
		self.n = n
		self.amount = A.size
		self.types = np.zeros(self.amount, dtype = np.uint)
		self.max_iter = max_iter
		self._init_centers()

	def _init_centers(self):
		interval = 256//(self.n*2)
		self.centers = [interval * (2*k+1) for k in range(self.n)]
		# center_indexs = []
		# for i in range(self.n):
		# 	while True:
		# 		rindex = random.randint(0, self.amount-1)
		# 		if rindex not in center_indexs:
		# 			center_indexs.append(rindex)
		# 			break
		# self.centers = self.A[center_indexs]

	def determine_types(self):
		self.types = np.asarray([np.asarray([(self.A[i] - self.centers[j]) ** 2 for j in range(self.n)]).argmin(0) for i in range(self.amount)])

	def refresh_centers(self):

		cluster_length = []
		for i in range(self.n):
			index = np.where(self.types == i)
			length = len(index[0])
			cluster_length.append(length)
		cluster_length = np.asarray(cluster_length)
			
		# if some cluster appeared to be empty then:
		#   1. find the biggest cluster
		#   2. find the farthest k points from the center point in the biggest cluster
		#   3. exclude the farthest k points from the biggest cluster and form a new k-point cluster.(where k is equal to 1/3 of the length of biggest cluster)
		for i in range(self.n):
			if cluster_length[i] == 0:
				max_length = cluster_length.max()
				k = cluster_length.argmax(0)
				p = np.where(self.types == k)
				pixels = self.A[p] #像素点个数最多的一个聚类
				index = np.asarray([(r - self.centers[k])**2 for r in pixels]).argsort()[-(max_length//3):]
				index = p[0][index]
				self.types[index] = i

		for i in range(self.n):
			index = np.where(self.types == i)
			self.centers[i] = self.A[index].sum(axis=0)/len(index[0])
	
			



	def run(self):
		for i in range(self.max_iter):
			self.determine_types()
			self.refresh_centers()

	def plot(self):
		data_x = []
		data_y = []
		data_z = []
		for i in range(self.n):
			index = np.where(self.types == i)
			data_x.extend(self.A[index][:,0].tolist())
			data_y.extend(self.A[index][:,1].tolist())
			data_z.extend([i/self.n for j in range(len(index[0]))])
		sc = plt.scatter(data_x, data_y, c=data_z, vmin=0, vmax=1, s=35, alpha=0.8)
		plt.colorbar(sc)
		plt.show()

	def output(self):
		# self.types = np.asarray(self.types).reshape(self.shape[:-1])
		# print(self.types.shape)
		# return self.types
		self.data_by_comp = []
		for ci in range(self.n):
			index = np.where(self.types == ci)
			self.data_by_comp.append(self.A[index])
		return self.data_by_comp



if __name__ == '__main__':
	A = np.random.random([1000, 3, 2])
	k = kmeans(A, n = 20, max_iter = 10)
	k.run()
	k.plot()
	r = k.output()
	print(r.shape)
