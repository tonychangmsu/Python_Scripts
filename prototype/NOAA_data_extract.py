#python NOAA COOP station extract 
#script automatically downloads all the data from the NOAA FTP server from COOP station list 3200 GHCN and COOP

import numpy as np
from ftplib import FTP
import os

ftp = FTP('ftp3.ncdc.noaa.gov', 'anonymous', 'tony.chang@msu.montana.edu') #access the NCDC FTP database
ftp.cwd('pub/data/3200') #access the 3200 database
years = np.arange(1895, 2013)
months = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
for i in years:
	ftp.cwd(str(i))
	for j in months:
		filename = '3200' + j + str(i)
		fhandle = open(os.path.join('E:\\NCDC\\3200\\' + str(i), filename), 'wb')
		ftp.retrbinary('RETR ' + filename, fhandle.write)
		fhandle.close()
	ftp.cwd('..')
ftp.quit()