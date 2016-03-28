'''
Title: 		ESPA_website_downloader.py
Date: 		Created 2016.02.11

Author: 	Tony Chang
Abstract: 	Uses requests and BeautifulSoup libraries to get the urls of all the Landsat scenes to download
			from the http://espa.cr.usgs.gov/ordering/status/tony.chang@msu.montana.edu-02102016-172703/
			webpage, after requesting post processing
			###
			The above failed an I resorted to copying the html source and then parsing it with BeautifulSoup
			Once parsed, I retrieved the a href urls and downloaded everything with urllib.request
			
			In addition, after downloading the main unzips all the files and stores them in appropriate directory
'''

import requests
from bs4 import BeautifulSoup as bs
import urllib.request
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

'''
espa_url = 'http://espa.cr.usgs.gov/ordering/status/tony.chang@msu.montana.edu-02102016-172703/'
espa_login_url = 'https://espa.cr.usgs.gov/login/'
payload = {'id_username': 'tony.chang', 'id_password':'Bd060582#'}
r = requests.get(espa_url, params = payload)
'''

#code above does not quite work. So I will just parse the html
if __name__ == "__main__":
	html_file = 'G:\\NASA_remote_data\\earthexplorer\\espa_request_2016_02_11.html'
	f = open(html_file, 'r')
	soup = bs(f.read(), 'html.parser')
	#find all the a href links
	dl_links = soup.findAll('a', href = True)
	urls = [a['href'] for a in dl_links if '.tar.gz' in a['href']]

	#now we have the urls
	#loop through them and download what we need.
	workspace = r'G:\NASA_remote_data\LT5_LEDAPS_processed'
	for i in urls:
		savefile = '%s\\%s'%(workspace, i[74:])
		urllib.request.urlretrieve(i, savefile)
		print('downloaded from %s'%i)
		urllib.request.urlcleanup() #clean up cache

	#then unzip all these files to the appropriate place
	#make directories to store in appropriate place
	wd = r'G:\NASA_remote_data\LT5_LEDAPS_processed'
	p = [37,38,39]
	r = [28,29,30]
	for pi in p:
		for ri in r:
			d = r'%s\p%sr%s'%(wd, pi, ri)
			if not os.path.exists(d):
				os.makedirs(d)

#############################################		UNZIPPING		####################################################
	#create log file
	lf = r'%s\unzip_tool.log' %(wd)
	logger = createLogger(name = lf)
	counter_y = 0
	counter_n = 0
	w_path = wd
	path_dirs = os.listdir(path) 
	#search all the files
	f_names = [s for s in path_dirs if ".tar.gz" in s] #get the name of the bulk order directory 
	for fi in f_names:
		#fi = f_names[0] # first iteration of .tar.gz
		tar_file = r'%s\%s'%(w_path,fi)
		p = int(fi[4:6])
		r = int(fi[7:9])
		extract_wd = r"G:\NASA_remote_data\LT5_LEDAPS_processed\p%sr%s"%(p,r)
		gz_dir = r"%s\%s"%(extract_wd, fi[:16])
		#create a directory to house the data
		if not os.path.exists(gz_dir):
			os.makedirs(gz_dir) #make a directory to store the uncompressed data
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
	#end