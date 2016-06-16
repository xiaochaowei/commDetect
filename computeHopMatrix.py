import community
import networkx as nx
import scipy.io as sio
import numpy as np
def readNeighborAndGenGraph(filename = 'as_neightbors.csv'):
	with open(filename, 'r') as f:
		data = f.read()
	rows = data.split("\n")
	as2as = {}
	rows_len = len(rows)
	G = nx.Graph()
	for i in range(0, rows_len):
		# print i, rows_len
		if rows[i] == "":
			break
		row = rows[i].split(",")
		as1 = row[0]
		as2 = row[1]
		G.add_edge(as1, as2)
	return G

def computeHopMatrix():
	filename = "tmp/hopmatrix.mat"
	# comment_sql = "SELECT as_num from distri ;"
	prefix2asList = []
	# cursor.execute(comment_sql)
	# rows = cursor.fetchall()
	fid = open('data/distribution_result.txt','r')
	data = fid.read()
	rows = data.split("\n")
	for row in rows:
		if row =="":
			break
		tmp = row.split(",")
		prefix2asList.append(tmp[1])
	n_nodes = len(prefix2asList)
	print n_nodes
	matrix = np.zeros((n_nodes, n_nodes))
	G_hop = readNeighborAndGenGraph()
	for i in range(0, n_nodes):
		for j in range(i, n_nodes):
			if prefix2asList[j] == "-2" or prefix2asList[i] == "-2":
				hop = -1
			else:
				try:
					hop = nx.shortest_path(G_hop, source = prefix2asList[i], target = prefix2asList[j])
					hop = len(hop)
				except Exception as e:
					hop = -1
			matrix[i,j] = hop
			matrix[j,i] = hop
			print hop
	sio.savemat(filename, {"matrix":matrix})

computeHopMatrix()