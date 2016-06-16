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
		# print k_v
		partition[int(k_v[0])] = int(k_v[1])
	fid.close()
	return partition
