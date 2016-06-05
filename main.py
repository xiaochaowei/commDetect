import community
import networkx as nx
import matplotlib.pyplot as plt 
import numpy as np

def Similarity(A,B):
	# print "sim"
	tmp = np.flipud(B)
	tmp_conv = np.convolve(A, tmp)
	min_len = min(len(A), len(B))
	max_len = max(len(A), len(B))
	#normalization
	sim_value = -1
	for i in range(0, len(tmp_conv)):
		if i < min_len:
			tmp_conv[i] = tmp_conv[i] / (i+1)
		elif i > max_len and i < min_len:
			tmp_conv[i] = tmp_conv[i] / min_len
		else:
			tmp_conv[i] = tmp_conv[i] / ( len(B) - (i+1 - len(A)) )
		if sim_value < tmp_conv[i]:
			sim_value = tmp_conv[i]
	return  sim_value

def similarityTest():
	a = np.array([1,2,3,4,5])
	b = np.array([1,2,3,4,5])
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
# def testDraw():
# 	G = nx.Graph()
# 	G.add_nodes_from(range(0,10))
# 	G.add_edge(0,1,weight = 100)
# 	G.add_edge(0,2,weight = 10)
# 	G.add_edge(0,3,weight = 30)
# 	G.add_edge(1,7,weight = 3)
# 	nx.draw(G)
# 	plt.show()
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

def test_visual():
	G = nx.Graph()
	G.add_nodes_from(range(10))
	cm = plt.cm.get_cmap('RdYlBu')
	pos = nx.spring_layout(G)
	nodes1 = range(0,2)
	nodes2 = range(2,4)
	nodes3 = range(4,6)
	nodes4 = range(6,8)
	nx.draw_networkx_nodes(G, pos, nodes1, node_size = 20, node_color = str(0.0), cmap = plt.cm.jet)
	nx.draw_networkx_nodes(G, pos, nodes2, node_size = 20, node_color = str(0.5), cmap = plt.cm.jet)
	nx.draw_networkx_nodes(G, pos, nodes3, node_size = 20, node_color = str(1.0), cmap = plt.cm.jet)
	
	# nx.draw_networkx_nodes(G, pos, nodes4, node_size = 20, node_color = str(1), vmin = 0, vmax = 4,cmap = cm)
	plt.show()

import scipy.io as sio
def saveResult(filename, partition):
	fid = open(filename,'w')
	for key in partition.keys():
		value = partition[key]
		tmp_str = str(key) + "," + str(value) + "\n"
		fid.write(tmp_str)
	fid.close()

def readResult(filename):
	fid = open(filename,'r')
	data = fid.read()
	data = data.split("\n")
	partition = {}
	for tmp in data:
		if tmp == "":
			break
		k_v = tmp.split(",")
		print k_v
		partition[k_v[0]] = int(k_v[1])
	fid.close()
	return partition

import csv
def readBGP(filename = 'bpg.csv'):
	data = open(filename, 'r').read()
	rows = data.split('\n')
	prefix2as = {}
	rows_len = len(rows)
	for i in range(0, rows_len):
		row = rows[i].split(',')
		prefix = row[0]
		as_num = row[1]
		if prefix2as.has_key(prefix):
			print "something wrong"
			prefix2as[prefix] = as_num
		else:
			prefix2as[prefix] = as_num

	return prefix2as

def readAs_type(filename = 'as_type.csv'):
	with open(filename,'r') as f:
		data = f.read()
	rows = data.split('\n')
	rows_len = len(rows)
	prefix2astype = {}
	for i in range(0, rows_len):
		row = rows[i].split(',')
		prefix = row[0]
		country = row[1]
		as_type = row[2]
		if prefix2astype.has_key(prefix):
			print "something wrong"
		prefix2astype[prefix] = [country, as_type]
	return prefix2astype

def readNeighborAndGenGraph(filename = 'as_neightbors.csv'):
	with open(filename, 'r') as f:
		data = f.read()
	rows = data.split("\n")
	as2as = {}
	rows_len = len(rows)
	G = nx.Graph()
	for i in range(0, rows_len):
		row = rows[i]
		as1 = row[0]
		as2 = row[1]
		G.add_edge(as1, as2)
	return G

import MySQLdb
conn = MySQLdb.connect(user = 'root', passwd = '19920930', db = 'rbl_data')
cursor = conn.cursor()
QUERY_AS_NUM  = """SELECT * FROM bgp WHERE prefix = {prefix} ;"""
QUERY_AS_TYPE = """SELECT * FROM as_type WHERE prefix = {prefix} ; """ 
def computerDisforEachComm(partition, filename):
	fid = open(filename, 'w')
	for i in partition.keys():
		prefix = prefix_data[i]
		comm = partition[i]
		comment_sql = QUERY_AS_NUM.format(prefix = prefix)
		cursor.execute(comment_sql)
		rows = cursor.fetchall()
		if len(rows) == 1:
			as_num = rows[0][1]
		else:
			print "error"
		comment_sql = QUERY_AS_TYPE.format(prefix = prefix)
		cursor.execute(comment_sql)
		rows = cursor.fetchall()
		if len(rows) == 1:
			country = rows[0][1]
			as_type = rows[0][2]
		else:
			print "wrong"
		#insert into database 
		tmp = prefix + "," + as_num + "," + country + "," + as_type+"\n"
		fid.write(tmp)
		fid.close()


def preprocess():
	data = sio.loadmat('phishing_2013.mat')
	phish_data = data['phish']
	prefix_data = data['networks']
	# S = ComuputeSimilarity(phish_data)
	G = nx.Graph()
	genGraph(phish_data, G)
	# nx.draw(G)
	# partition = communityDetect(G)
	dendo = community.generate_dendrogram(G)
	# print dendo
	filename = "partition"
	for level in range(len(dendo)):
		save(filename + str(level), community.partition_at_level(dendo, level))
		# print "partition at level", level,\
			# "is ", community.partition_at_level(dendo, level)
	# saveResult("partition2.txt", partition)
	# partition = readResult("partition2.txt")
	# virutalization(partition, G)
	# plt.show()	
# test_visual()
preprocess()
# similarityTest()