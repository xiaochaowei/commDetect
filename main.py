import community
import networkx as nx
import matplotlib.pyplot as plt 
import numpy as np
import math
import base

def Similarity(A,B):
	# print "sim"
	tmp = np.flipud(B)
	tmp_conv = np.convolve(A, tmp)
	min_len = min(len(A), len(B))
	max_len = max(len(A), len(B))
	#normalization
	sim_value = 0
	for i in range(0, len(tmp_conv)):
		if tmp_conv[i] == 0:
			continue
		if i < min_len:
			# print tmp_conv[i]
			value = float(tmp_conv[i]) / float(math.sqrt(np.dot(A[0:i+1] , A[0:i+1])) * math.sqrt(np.dot(B[len(B) - (i+1):], B[len(B) - (i+1):])))
			# print tmp_conv[i]
			# print  float((np.linalg.norm(A[0:i+1]) * np.linalg.norm(B[len(B) - (i+1):])))
			# print A[0:i+1], B[len(B) - (i+1):]
			# print tmp_conv[i]
		elif i >= max_len:
			value = float(tmp_conv[i]) / float(math.sqrt(np.dot(A[((i+1) - max_len):], A[((i+1) - max_len):])) *math.sqrt( np.dot(B[0:(2* max_len - (i+1))], B[0:(2 * max_len - (i+1))])))
			# print A[((i+1) - max_len):], B[0:(2* max_len - (i+1))]
		# print tmp_conv[i]
		
		if sim_value < value and ( i > 0.1 * max_len and i < 0.9 * max_len*2 ):
			sim_value = value
	return  sim_value


def similarityTest():
	a = np.array([1,2,3,4,5, 6,7,8,9,10,11])
	b = np.array([1,2,3,4,5,6,7,8,9,10,11])
	Similarity(a,b)

def ComuputeSimilarity(A,B):
	#correlation
	# print "similarity"
	return 2 * np.dot(A,B) / (np.dot(A,A)+ np.dot(B,B))
	# (row, col) = matrix.shape 
	# S = np.zeros(row, row)
	# for r_idx in range(0, row):
	# 	r_norm = np.dot(matrix[r_idx, :], matrix[r_idx,:])
	# 	for c_idx in range(0, row):
	# 		c_norm = np.dot(matrix[c_idx,:], matrix[c_idx, :])
	# 		S[r_idx, c_idx] = np.dot(matrix[r_idx,:], matrix[c_idx,:]) * 2 / (c_norm + r_norm)

	# return S
	#cosein
	#sim = no.corrceof(matrix, matrix.T)

def genGraph(matrix, G):
	[n_nodes,dim] = matrix.shape
	G.add_nodes_from(range(0, n_nodes))
	for i in range(0, n_nodes):
		# print i
		for j in range(i, n_nodes):
			G.add_edge(i,j,weight = Similarity(matrix[i,:], matrix[j,:]))


def computeWeightToFile(filename, data):
	[n_nodes, dim] = data.shape
	result = np.zeros((n_nodes, n_nodes))
	for i in range(0, n_nodes):
		print i, n_nodes 
		for j in range(i,n_nodes):
			weight = Similarity(data[i,:], data[j,:])
			result[i,j] = weight
			result[j,i] = weight
	sio.savemat(filename, {"weight": result})

def communityDetect(G):
	partition = community.best_partition(G, None, "weight", 1)
	return partition

def virutalization(partition, G):
	size = float(len(set(partition.values())))
	print "size", size
	count = 0
	cm = plt.cm.get_cmap('jet')
	pos = nx.spring_layout(G)
	# nx.draw_networkx_nodes(G, pos, )
	# for comm in set(partition.values()):
	# 	count += 1
	# 	list_nodes = [nodes for nodes in partition.keys() if partition[nodes] == comm]
		
	sc = nx.draw_networkx_nodes(G, pos, node_size = 20, node_color = partition.values(), vmin = 0, vmax = size, cmap = cm )
	# nx.draw_networkx_edges(G, pos, alpha = 0.5)
	plt.colorbar(sc)




def genGraphFromFile(filename):
	G = nx.Graph()
	data = sio.loadmat(filename)
	weight = data['weight']
	print "load success"
	# weight = np.loadtxt(filename, delimiter = ',')
	[n_nodes, n_nodes] =weight.shape
	for i in range(0, n_nodes):
		for j in range(i+1, n_nodes):
			G.add_edge(i,j,weight = weight[i,j])
	return G
#	partition = readResult('data/partition0')
#	virutalization(partition, G)
#	plt.savefig("result1.png")
def preprocess():
	data = sio.loadmat('f_data/phishing_2013_filter.mat')
	phish_data = data['phish']
	prefix_data = data['networks']
#	computeWeightToFile('f_data/weight.mat', phish_data)
	G = genGraphFromFile('f_data/weight.mat')
#	print 'load file success'
	# S = ComuputeSimilarity(phish_data)
	#G = nx.Graph()
	#genGraph(phish_data, G)
	# nx.write_gml(G, 'data/graph')
	# nx.draw(G)
	# partition = communityDetect(G)
	# partition = readResult("data/partition1")
	dendo = community.generate_dendrogram(G)
	# print len(dendo)
#	print 'partition sucess', len(dendo)
#	filename = "f_data/partition"
	for level in range(len(dendo)):
		partition = community.partition_at_level(dendo,level)
		print 'size', len(set(partition.values()))
		saveResult(filename + str(level), partition)
		# print "partition at level", level,\
			# "is ", community.partition_at_level(dendo, level)
	# saveResult("partition2.txt", partition)
	# partition = readResult("data/partition1")
	# computerDisforEachComm(partition, prefix_data, 'data/distribution_result1.txt')
	# virutalization(partition, G)
	# plt.show()	
# test_visual()
preprocess()
# genGraphFromFile('data/weight.txt')
# computeHopMatrix()

