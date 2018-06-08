import cv2
import SimpleITK as sitk
import numpy as np 
import os
import time
import math
from k_means import kmeans
from gcgraph import GCGraph

#read dicom images
def loadFile(filename):
    ds = sitk.ReadImage(filename)
    img_array = sitk.GetArrayFromImage(ds)
    return img_array

#空回调函数
def nothing(x):
    pass

#Gaussian Mixture Model
class GMM:
	'''The GMM: Gaussian Mixture Model algorithm'''
	'''Each point in the image belongs to a GMM, and because each pixel owns
		three channels: RGB, so each component owns three means, 9 covs and a weight.'''
	
	def __init__(self, k = 5):
		'''k is the number of components of GMM'''
		self.k = k
		self.weights = np.asarray([0. for i in range(k)], dtype = 'float32') # Weight of each component
		self.means = np.asarray([0. for i in range(k)], dtype = 'float32') # Means of each component
		self.vars = np.asarray([0. for i in range(k)], dtype = 'float32') # vars of each component

		self.pixel_counts = np.asarray([0 for i in range(k)]) # Count of pixels in each components
		self.pixel_total_count = 0 # The total number of pixels in the GMM
		
		# The following parameter is assistant parameters for counting pixels and calc. params.
		self._sums = np.asarray([0. for i in range(k)])

	def _prob_pixel_component(self, pixel, ci):
		'''Calculate the probability of each pixel belonging to the ci_th component of GMM'''
		'''Using the formula of multivariate normal distribution'''
		return 1/np.sqrt(self.vars[ci]) * np.exp(-0.5*math.pow(pixel-self.means[ci], 2)/self.vars[ci]) # gaussian distribution formula

	def prob_pixel_GMM(self, pixel):	
		'''Calculate the probability of each pixel belonging to this GMM, which is the sum of 
			the prob. of the pixel belonging to each component * the weight of the component'''
		'''Also the first term of Gibbs Energy(negative;)'''
		return sum([self._prob_pixel_component(pixel, ci) * self.weights[ci] for ci in range(self.k)])

	def most_likely_pixel_component(self, pixel):
		'''Calculate the most likely component that the pixel belongs to'''
		prob = np.asarray([self._prob_pixel_component(pixel, ci) for ci in range(self.k)])
		return prob.argmax(0)
	
	def learning(self, components):
		for ci in range(self.k):
			print('length of conponent[',ci,']:', len(components[ci]))
			self.means[ci] = components[ci].mean()
			self.vars[ci] = components[ci].var()
			if self.vars[ci] < 0.1:
				self.vars[ci] += 1
			self.pixel_counts[ci] = components[ci].size

		self.pixel_total_count = self.pixel_counts.sum()
		for ci in range(self.k):
			self.weights[ci] = self.pixel_counts[ci]/self.pixel_total_count
	

class GCClient:
	'''The engine of grabcut'''
	def __init__(self, img, k):
		self.k = k # The number of components in each GMM model 
		self.img3d = np.asarray(img, dtype = np.float32)
		self.img3d_2 = self.img3d.copy()
		self.depth, self.rows, self.cols = img.shape
		print('img.shape:', img.shape)
		self.gamma = 50
		self.lam = 28*self.gamma
		self.beta = 0

		self._BLACK = 0       # sure BG
		self._GRAY1 = 80      # PR BG
		self._GRAY2 = 160     # PR FG
		self._WHITE = 255     # sure FG
		self._GC_BGD = 0
		self._GC_FGD = 1
		self._GC_PR_BGD = 2
		self._GC_PR_FGD = 3
		self._DRAW_BG = {'color':self._BLACK, 'val':self._GC_BGD}
		self._DRAW_FG = {'color':self._WHITE, 'val':self._GC_FGD}
		self._DRAW_PR_BG = {'color':self._GRAY1, 'val':self._GC_PR_BGD}
		self._DRAW_PR_FG = {'color':self._GRAY2, 'val':self._GC_PR_FGD}

		# setting up flags
		self._rect1 = [0, 0, 1, 1, -1]  #[x1,x2, y1,y2, z1] 矩形1左上角的坐标以及宽、高、深度坐标
		self._rect2 = [0, 0, 1, 1, -1]  #[x1,x2, y1,y2, z2] 矩形2左上角的坐标以及宽、高、深度坐标
		self._cube = [0, 0, 1, 1, 0, 1] #[x1,x2, y1,y2, z1,z2]前景存在的立方体的坐标
		self._drawing = False          # flag for drawing curves
		self._rectangle1 = False       # flag for drawing rectangle1
		self._rectangle2 = False       # flag for drawing rectangle2
		self._rect1_over = False       # flag to check if rect drawn
		self._rect2_over = False       # flag to check if rect drawn
		self._rect_or_mask = 0         # flag for selecting rect or mask mode
		
		self._thickness = 2            # brush thickness
		self._DRAW_VAL = None  #color of brush
		self._mask = np.zeros([self.depth, self.rows, self.cols], dtype = np.uint8) # Init the mask
		self._mask[:, :, :] = self._GC_BGD
		self._mask3d = self._mask.astype('float32')

		self.calc_nearby_weight()

	def calc_nearby_weight(self):
		'''STEP 1:'''
		'''Calculate Beta -- The Exp Term of Smooth Parameter in Gibbs Energy'''
		'''beta = 1/(2*average(sqrt(||pixel[i] - pixel[j]||)))'''
		'''Beta is used to adjust the difference of two nearby pixels in high or low contrast rate'''
		'''STEP 2:'''
		'''Calculate the weight of the edge of each pixel with its nearby pixel, as each pixel is regarded
		as a vertex of the graph'''
		'''The weight of each direction is saved in a image the same size of the original image'''
		'''weight = gamma * 1/distance<m,n> * exp(-beta*(diff*diff))'''

		#diff1 = self.img3d[:-1,:-1,:-1] - self.img3d[1:,1:,1:]
		diff2 = self.img3d[:-1,:-1,:] - self.img3d[1:,1:,:]
		#diff3 = self.img3d[:-1,:-1,1:] - self.img3d[1:,1:,:-1]
		diff4 = self.img3d[:-1,:,:-1] - self.img3d[1:,:,1:]
		diff5 = self.img3d[:-1,:,:] - self.img3d[1:,:,:]
		diff6 = self.img3d[:-1,:,1:] - self.img3d[1:,:,:-1]
		#diff7 = self.img3d[:-1,1:,:-1] - self.img3d[1:,:-1,1:]
		diff8 = self.img3d[:-1,1:,:] - self.img3d[1:,:-1,:]
		#diff9 = self.img3d[:-1,1:,1:] - self.img3d[1:,:-1,:-1]
		diff10 = self.img3d[:,:-1,:-1] - self.img3d[:,1:,1:]
		diff11 = self.img3d[:,:-1,:] - self.img3d[:,1:,:]
		diff12 = self.img3d[:,:-1,1:] - self.img3d[:,1:,:-1]
		diff13 = self.img3d[:,:,:-1] - self.img3d[:,:,1:]

		beta = (diff2*diff2).sum() + (diff4*diff4).sum() + (diff5*diff5).sum() \
		 + (diff6*diff6).sum() + (diff8*diff8).sum() + (diff10*diff10).sum() \
		 + (diff11*diff11).sum() + (diff12*diff12).sum() + (diff13*diff13).sum()
		self.beta = (9*self.rows*self.cols*self.depth - 5*self.rows*self.cols - 5*self.rows*self.depth - 5*self.cols*self.depth + 2*self.rows + 2*self.cols + 2*self.depth)/(2*beta)
		print('self.beta:', self.beta)

		# Use the formula to calculate the weight
		self.weight2 = self.gamma * 0.707 * np.exp(-self.beta*(diff2*diff2))
		self.weight4 = self.gamma * 0.707 * np.exp(-self.beta*(diff4*diff4))
		self.weight5 = self.gamma * np.exp(-self.beta*(diff5*diff5))
		self.weight6 = self.gamma * 0.707 * np.exp(-self.beta*(diff6*diff6))
		self.weight8 = self.gamma * 0.707 * np.exp(-self.beta*(diff8*diff8))
		self.weight10 = self.gamma * 0.707 * np.exp(-self.beta*(diff10*diff10))
		self.weight11 = self.gamma * np.exp(-self.beta*(diff11*diff11))
		self.weight12 = self.gamma * 0.707 * np.exp(-self.beta*(diff12*diff12))
		self.weight13 = self.gamma * np.exp(-self.beta*(diff13*diff13))

	'''The following function is derived from the sample of opencv sources'''
	def init_mask(self, event, x, y, flags, param):
		'''Init the mask with interactive movements'''
		'''Notice: the elements in the mask should be within the follows:
			"GC_BGD":The pixel belongs to background;
			"GC_FGD":The pixel belongs to foreground;
			"GC_PR_BGD":The pixel MAY belongs to background;
			"GC_PR_FGD":The pixel MAY belongs to foreground;'''

		# Draw Rectangle1
		if self._rect1_over == False: 
			if event == cv2.EVENT_LBUTTONDOWN:
				self._rectangle1 = True
				self._ix,self._iy = x,y

			elif event == cv2.EVENT_LBUTTONUP:
				if self._rectangle1 == True:
					self._rect1_over = True
					self._rectangle1 == False
					cv2.rectangle(self.img3d_2[index1],(self._ix,self._iy),(x,y),self._WHITE, self._thickness)
					self._rect1 = [min(self._ix,x), max(self._ix,x), min(self._iy,y), max(self._iy,y), index1]
		
		elif self._rect2_over== False: 	
			if event == cv2.EVENT_LBUTTONDOWN:
				if index1 == self._rect1[4]:
					print('Please change the depth to draw anather rectangle!')
				else:
					self._rectangle2 = True
					self._ix,self._iy = x,y

			elif event == cv2.EVENT_LBUTTONUP:
				if self._rectangle2 == True:
					self._rect2_over = True
					cv2.rectangle(self.img3d_2[index1],(self._ix,self._iy),(x,y),self._WHITE, self._thickness)
					self._rect2 = [min(self._ix,x), max(self._ix,x), min(self._iy,y), max(self._iy,y), index1]
					self._cube = [min(self._rect1[0],self._rect2[0]), max(self._rect1[1],self._rect2[1]), min(self._rect1[2],self._rect2[2]), max(self._rect1[3],self._rect2[3]), min(self._rect1[4],self._rect2[4]), max(self._rect1[4],self._rect2[4])]
					print('self._cube:', self._cube)
					self._mask[self._cube[4]:self._cube[5]+1, self._cube[2]:self._cube[3], self._cube[0]:self._cube[1]] = self._GC_PR_FGD
					self._mask3d = self._mask.astype('float32')

		# Notice : The x and y axis in CV2 are inversed to those in numpy.
		elif self._DRAW_VAL:
			if event == cv2.EVENT_LBUTTONDOWN:
				self._drawing = True
				cv2.circle(self.img3d_2[index1], (x, y), self._thickness, self._DRAW_VAL['color'], -1)
				cv2.circle(self._mask3d[index1], (x, y), self._thickness, self._DRAW_VAL['val'], -1)

			elif event == cv2.EVENT_MOUSEMOVE:
				if self._drawing == True:
					cv2.circle(self.img3d_2[index1], (x, y), self._thickness, self._DRAW_VAL['color'], -1)
					cv2.circle(self._mask3d[index1], (x, y), self._thickness, self._DRAW_VAL['val'], -1)

			elif event == cv2.EVENT_LBUTTONUP:
				if self._drawing == True:
					self._drawing = False
					cv2.circle(self.img3d_2[index1], (x, y), self._thickness, self._DRAW_VAL['color'], -1)
					cv2.circle(self._mask3d[index1], (x, y), self._thickness, self._DRAW_VAL['val'], -1)

	def init_with_kmeans(self):
		'''Initialise the BGDGMM and FGDGMM, which are respectively background-model and foreground-model,
			using kmeans algorithm'''
		print('init with k-means processing...')
		self._mask = self._mask3d.astype('uint8')
		max_iter = 3 # Max-iteration count for Kmeans
		'''In the following two indexings, the np.logical_or is needed in place of or'''
		self._bgd = np.where(np.logical_or(self._mask == self._GC_BGD, self._mask == self._GC_PR_BGD)) # Find the places where pixels in the mask MAY belong to BGD.
		self._fgd = np.where(np.logical_or(self._mask == self._GC_FGD, self._mask == self._GC_PR_FGD)) # Find the places where pixels in the mask MAY belong to FGD.
		self._BGDpixels = self.img3d[self._bgd]
		self._FGDpixels = self.img3d[self._fgd]
		KMB = kmeans(self._BGDpixels, n = self.k, max_iter = max_iter) # The Background Model by kmeans
		KMF = kmeans(self._FGDpixels, n = self.k, max_iter = max_iter) # The Foreground Model by kmeans
		KMB.run()
		KMF.run()
		self._BGD_by_components = KMB.output()
		self._FGD_by_components = KMF.output()
		self.BGD_GMM = GMM() # The GMM Model for BGD
		self.FGD_GMM = GMM() # The GMM Model for FGD
		self.BGD_GMM.learning(self._BGD_by_components)
		self.FGD_GMM.learning(self._FGD_by_components)
		print('BGD_GMM:', '\nweights:\n', list(self.BGD_GMM.weights), '\nmeans:\n', list(self.BGD_GMM.means), '\nvars:\n', list(self.BGD_GMM.vars), '\n')
		print('FGD_GMM:', '\nweights:\n', list(self.FGD_GMM.weights), '\nmeans:\n', list(self.FGD_GMM.means), '\nvars:\n', list(self.FGD_GMM.vars), '\n')
	
	'''The first step of the iteration in the paper: Assign components of GMMs to pixels,
		(the kn in the paper), which is saved in self.components_index'''
	def assign_GMM_components(self):
		print('Refreshing GMM components...')
		self._mask = self._mask3d.astype('uint8')
		self.components_index = np.asarray([self.BGD_GMM.most_likely_pixel_component(pixel) if (mask == self._GC_BGD) or (mask == self._GC_PR_BGD) else self.FGD_GMM.most_likely_pixel_component(pixel) for mat1,mat2 in zip(self.img3d,self._mask) for row1,row2 in zip(mat1,mat2) for pixel,mask in zip(row1,row2)], dtype='uint8').reshape(self.depth, self.rows, self.cols)

	'''The second step in the iteration: Learn the parameters from GMM models'''
	def learn_GMM_parameters(self):
		'''Calculate the parameters of each component of GMM'''
		print('Learning GMM parameters...')
		self._BGD_by_components = []
		self._FGD_by_components = []

		for ci in range(self.k):
			# The places where the pixel belongs to the ci_th model and background model.
			bgd_ci = np.where(np.logical_and(self.components_index == ci, np.logical_or(self._mask == self._GC_BGD, self._mask == self._GC_PR_BGD)))
			self._BGD_by_components.append(self.img3d[bgd_ci])
			# The places where the pixel belongs to the ci_th model and foreground model.
			fgd_ci = np.where(np.logical_and(self.components_index == ci, np.logical_or(self._mask == self._GC_FGD, self._mask == self._GC_PR_FGD)))
			self._FGD_by_components.append(self.img3d[fgd_ci])

		self.BGD_GMM.learning(self._BGD_by_components)
		self.FGD_GMM.learning(self._FGD_by_components)
		print('BGD_GMM:', '\nweights:\n', list(self.BGD_GMM.weights), '\nmeans:\n', list(self.BGD_GMM.means), '\nvars:\n', list(self.BGD_GMM.vars), '\n')
		print('FGD_GMM:', '\nweights:\n', list(self.FGD_GMM.weights), '\nmeans:\n', list(self.FGD_GMM.means), '\nvars:\n', list(self.FGD_GMM.vars), '\n')

	def _construct_gcgraph(self, lam, z, y, x):
		'''Set t-weights: Calculate the weight of each vertex with Sink node and Source node'''
		vertex_index = self.graph.add_vertex() # add-node and return its index
		color = self.img3d[z, y, x]
		if self._mask[z, y, x] == self._GC_PR_BGD or self._mask[z, y, x] == self._GC_PR_FGD:
			# For each vertex, calculate the first term of G.E. as it be the BGD or FGD, and set them respectively as weight to t/s.
			fromSource = -np.log(self.BGD_GMM.prob_pixel_GMM(color))
			toSink = -np.log(self.FGD_GMM.prob_pixel_GMM(color))

		elif self._mask[z, y, x] == self._GC_BGD:
			# For the vertexs that are Background pixels, t-weight with Source = 0, with Sink = lam
			fromSource = 0
			toSink = lam
		else:
			fromSource = lam
			toSink = 0
		self.graph.add_term_weights(vertex_index, fromSource, toSink)

		'''Set n-weights and n-link, Calculate the weights between two neighbour vertexs, which is also the second term in Gibbs Energy(the smooth term)'''
		if z > 0:
			w = self.weight5[z-1, y, x]
			self.graph.add_edges(vertex_index, vertex_index-(self.rows*self.cols), w, w)
			if y > 0:
				w = self.weight2[z-1, y-1, x]
				self.graph.add_edges(vertex_index, vertex_index-(self.rows*self.cols+self.cols), w, w)
			if x > 0:
				w = self.weight4[z-1, y, x-1]
				self.graph.add_edges(vertex_index, vertex_index-(self.rows*self.cols+1), w, w)
			if x < self.cols-1:
				w = self.weight6[z-1, y, x]
				self.graph.add_edges(vertex_index, vertex_index-(self.rows*self.cols-1), w, w)
			if y < self.rows-1:
				w = self.weight8[z-1, y, x]
				self.graph.add_edges(vertex_index, vertex_index-(self.rows*self.cols-self.cols), w, w)
		if y > 0:
			w = self.weight11[z, y-1, x]
			self.graph.add_edges(vertex_index, vertex_index-self.cols, w, w)
			if x > 0:
				w = self.weight10[z, y-1, x-1]
				self.graph.add_edges(vertex_index, vertex_index-(self.cols+1), w, w)
			if x < self.cols-1:
				w = self.weight12[z, y-1, x]
				self.graph.add_edges(vertex_index, vertex_index-(self.cols-1), w, w)
		if x > 0:
			w = self.weight13[z, y, x-1]
			self.graph.add_edges(vertex_index, vertex_index-1, w, w)

	def construct_gcgraph(self, lam):
		'''Construct a GCGraph with the Gibbs Energy'''
		'''The vertexs of the graph are the pixels, and the edges are constructed by two parts,
		the first part of which are the edges that connect each vertex with Sink Point(the background) and the Source Point(the foreground),
		and the weight of which is the first term in Gibbs Energy;
		the second part of the edges are those that connect each vertex with its neighbourhoods,
		and the weight of which is the second term in Gibbs Energy.'''
		print('Constructing grabcut graph...')
		vertex_count = self.cols*self.rows*self.depth
		edge_count = 2*(9*self.rows*self.cols*self.depth-5*(self.rows*self.cols+self.rows*self.depth+self.cols*self.depth)+2*(self.rows+self.cols+self.depth))   #有向图的边数，每两个顶点之间有正反两条边
		self.graph = GCGraph(vertex_count, edge_count)
		[self._construct_gcgraph(lam, z, y, x) for z in range(self.depth) for y in range(self.rows) for x in range(self.cols)]		

	def estimate_segmentation(self):
		print('Estimating segmentation...')
		a =  self.graph.max_flow()
		self._mask = np.asarray([self._GC_PR_FGD if self.graph.insource_segment(self.rows*self.cols*z + self.cols*y + x) else self._GC_PR_BGD for z in range(self.depth) for y in range(self.rows) for x in range(self.cols)]).reshape(self.depth, self.rows, self.cols)

	def iter(self, n):
		for i in range(n):
			self.assign_GMM_components()
			self.learn_GMM_parameters()
			self.construct_gcgraph(self.lam)
			self.estimate_segmentation()

	def run(self):
		self.init_with_kmeans()
		self.construct_gcgraph(self.lam)
		self.estimate_segmentation()

		
if __name__ == '__main__':
	dir_name = '/Users/yann/Downloads/28204'
	file_prefix = '600440935_20160318_1_1_P0160612101150_9_'
	frames = 0

	files = os.listdir(dir_name)
	series = []
	#check the amount of the slices
	for file in files:
		if file.find(file_prefix) != -1:
			frames += 1
			series.append(file)
			if frames == 1:
				img = loadFile(os.path.join(dir_name, file))
				print('img.shape:', img.shape)
				f, h, w = img.shape
	#Initialize an array to store the slices
	img_array = np.zeros([frames, h, w], dtype=img.dtype)
	#sort the file name
	series.sort(key = lambda x:int(x[-6:-4]) if x[-7] == '_' else int(x[-7:-4]))


	#read all slices
	for index, file in enumerate(series):
		img_array[index, :, :] = loadFile(os.path.join(dir_name, file))
		
	img_array_uint8 = ((img_array - img_array.min()) * 1.0 / (img_array.max()-img_array.min()) * 255).astype('uint8')
	img_array_uint8 = img_array_uint8[85:120,:,:] 


	#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
	# img_array_uint8 = np.zeros([15, 135, 240], dtype = 'uint8')
	# temp = cv2.imread('/Users/yann/毕业设计/mycode/grabcut3d/test/bull.jpg', 0)
	# temp = temp[::2,::2]
	# print('temp:', temp.shape)
	# img_array_uint8[:,:,:] = temp
	# frames, h, w = img_array_uint8.shape 

	start = time.time()
	GC = GCClient(img_array_uint8, k = 5)
	print('time cost:',time.time() - start)

	cv2.namedWindow('DicomReader')
	cv2.namedWindow('DicomOutput')
	cv2.setMouseCallback('DicomReader',GC.init_mask)
	#cv2.moveWindow('DicomReader', 1, 100)
	#cv2.moveWindow('DicomOutput', 1, img.shape[1])
	cv2.createTrackbar('slices','DicomReader',0,img_array_uint8.shape[0]-1,nothing)
	#cv2.createTrackbar('slices','DicomOutput',0,len(series)-1,nothing)
	tips = '''
********************************************************************************
Instructions: \n
Draw two rectangles in two different slices around the object using left mouse button
Press N to segment
For finer touchups, mark foreground and background after pressing any key of 0~3 and again press 'N'
0: Sure background
1: Sure foreground
2: Probable background
3: Probable foreground
********************************************************************************'''
	print(tips)
	output = np.zeros([frames, h, w], dtype='uint8')
	index1 = 0
	while(1): 
		cv2.imshow('DicomReader', GC.img3d_2[index1].astype('uint8'))
		cv2.imshow('DicomOutput', output[index1])
		k=cv2.waitKey(1) & 0xFF 

		if k==27:
			break

		elif k == ord('n'):
			if GC._rect2_over == 0:
				print('Please draw two ractangles first!')
			else:
				print('Segmenting!')
				if GC._rect_or_mask == 0:
					GC.run()
					GC._rect_or_mask = 1
					print('Segmenting over')
					print('mask.shape:', GC._mask.shape, 'max:', GC._mask.max())
					index = np.where(GC._mask==3)
					print('num:', len(index[0]))
				elif GC._rect_or_mask == 1:
					GC.iter(1)
					print('Segmenting over')

		elif k == ord('0'):
			print('Mark background regions with left mouse button \n')
			GC._DRAW_VAL = GC._DRAW_BG

		elif k == ord('1'):
			print('Mark foreground regions with left mouse button \n')
			GC._DRAW_VAL = GC._DRAW_FG

		elif k == ord('2'):
			print('Mark prob. background regions with left mouse button \n')
			GC._DRAW_VAL = GC._DRAW_PR_BG

		elif k == ord('3'):
			print('Mark prob. foreground regions with left mouse button \n')
			GC._DRAW_VAL = GC._DRAW_PR_FG

		elif k == ord('s'):
			np.save('gc_img.npy', output)
			print('Save image successfully!')

		elif k == ord('r'):
			GC.__init__(img_array_uint8, k = 5)
			print('Reset successfully!')

		index1=cv2.getTrackbarPos('slices','DicomReader')

		FGD = np.where((GC._mask == 1) + (GC._mask == 3), 1, 0).astype('uint8')
		output = np.multiply(GC.img3d.astype('uint8'), FGD)

	cv2.destroyAllWindows()