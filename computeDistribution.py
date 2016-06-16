import MySQLdb
from  base import readResult
import scipy.io as sio
conn = MySQLdb.connect(user = 'root', passwd = '19920930', db = 'rbl_data')
cursor = conn.cursor()
QUERY_AS_NUM  = """SELECT * FROM bgp WHERE prefix = "{prefix}" ;"""
QUERY_AS_TYPE = """SELECT * FROM as_type WHERE prefix = "{prefix}" ; """ 

def computerDisforEachComm(partition, prefix_data, filename):
	fid = open(filename, 'w')
	for i in partition.keys():
		as_num = -2
		country = ""
		as_type = -1
		print i 
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
		tmp = prefix + "," + str(as_num) + "," + country + "," + str(as_type) + "," + str(partition[i]) + "\n"
		fid.write(tmp)
	fid.close()

import sys
def run():
	partition = readResult(sys.argv[1])
	data = sio.loadmat(sys.argv[2])
	prefix_data = data['networks']
	computerDisforEachComm(partition, prefix_data, 'f_data/distribtution_result.txt')

run()