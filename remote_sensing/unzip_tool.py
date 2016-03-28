#title: unzip_tool.py
#author: tony chang
#date:  4/15/2015
#abstract: script to uncompress tar.gz files for the landsat5 scene collections
#additional comments: landsat file naming convention 
#(i.e. LT50370282004273PAC01: LT5: landsat5, 037028: path37 row28, 
# 2004273: year 2004 julien day 273, PAC01  ground station identifier [see http://landsat.usgs.gov/about_ground_stations.php])

import numpy as np
import matplotlib.pyplot as plt
import os as os
from os import listdir, makedirs
from os.path import isfile, join, exists
import tarfile
import logging

def createLogger(name = 'K:\\out.log'):
	#sets up a logger with a given filename and sets up formatter to print timestamp for each log.
	logger = logging.getLogger()
	hdlr = logging.FileHandler(name)
	formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s') 
	hdlr.setFormatter(formatter)
	logger.addHandler(hdlr) 
	logger.setLevel(logging.INFO)
	return()
	
if __name__ == "__main__":
	wd = r'K:\nasa_data\landsat5'
	#create log file
	lf = r'%s\unzip_tool.log' %(wd)
	createLogger(name = lf)
	p = np.arange(37,40)	#define range of path
	r = np.arange(28,31)	#define range of row
	counter_y = 0
	counter_n = 0
	for pi in p:
		for ri in r:
			#pi = p[0] #path index
			#ri = r[0] #row index
			#path = r"K:\nasa_data\landsat5\p%sr%s\bulk order 441875\l4-5 tm"% (pi,ri)
			path = r"%s\p%sr%s"% (wd,pi,ri)
			path_dirs = listdir(path) 
			order_name = [s for s in path_dirs if "Bulk Order" in s] #get the name of the bulk order directory 
			w_path = r"%s\%s\l4-5 tm" %(path, order_name[0])#assumes only one directory
			#search all the files
			f_names =  [f for f in listdir(w_path) if f.endswith('.gz')] #find only the .gz files
			for fi in f_names:
				#fi = f_names[0] # first iteration of .tar.gz
				tar_file = r'%s\%s'%(w_path,fi)
				#create a directory to house the data
				gz_dir = r'%s\%s'% (w_path, fi[:(fi.find('.tar.gz'))])
				if not exists(gz_dir):
					makedirs(gz_dir) #make a directory to store the uncompressed data
					tfile = tarfile.open(tar_file, 'r:gz')
					tfile.extractall(path = gz_dir)
					tfile.close()
					counter_y += 1
					logmess = '%s unzipped!' % (fi)
					print(logmess)
					logger.info(logmess)
				else: #skip to the next for loop iteration for the .tar.gz files
					logmess = '%s already unzipped, skipping...' %(fi)
					print(logmess)
					logger.info(logmess)
					counter_n += 1
					continue	
	logmess = 'Job Done: %s files unzipped, %s files skipped'%(counter_y, counter_n)
	print(logmess)
	logger.info(logmess)