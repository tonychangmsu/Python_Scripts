#title: LT_unzip_tool.py
#author: tony chang
#creation date:  2/04/2016
#modified date: 02/05/2016
#dates ran: 02/04/2016
#comments: incomplete unzipping due to download from the USGS Bulk Download Application 
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
	return(logger)

if __name__ == "__main__":
	#wd = r'K:\nasa_data\landsat5'
	wd = r'G:\NASA_remote_data\LT_downloads'
	#create log file
	lf = r'%s\unzip_tool.log' %(wd)
	logger = createLogger(name = lf)
	counter_y = 0
	counter_n = 0
	#pi = p[0] #path index
	#ri = r[0] #row index
	#path = r"K:\nasa_data\landsat5\p%sr%s\bulk order 441875\l4-5 tm"% (pi,ri)
	#path = r"%s\p%sr%s"% (wd,pi,ri)
	path = wd
	path_dirs = listdir(path) 
	order_name = [s for s in path_dirs if "Bulk Order" in s] #get the name of the bulk order directory 
	w_path = r"%s\%s\l4-5 tm" %(path, order_name[0])#assumes only one directory
	#search all the files
	f_names =  [f for f in listdir(w_path) if f.endswith('.gz')] #find only the .gz files
	for fi in f_names:
		#fi = f_names[0] # first iteration of .tar.gz
		tar_file = r'%s\%s'%(w_path,fi)
		#from here extract the path and row
		p = int(fi[4:6])
		r = int(fi[7:9])
		#set the path for extraction
		extract_wd = r"G:\NASA_remote_data\Landsat5\p%sr%s"%(p,r)
		order_name = [s for s in listdir(extract_wd) if "Bulk Order" in s][0] #get the name of the bulk order directory 
		extract_path = r"%s\%s\L4-5 TM"%(extract_wd, order_name)
		#create a directory to house the data
		gz_dir = r'%s\%s'% (extract_path, fi[:(fi.find('.tar.gz'))])
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