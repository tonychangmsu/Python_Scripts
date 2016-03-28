#Title: MODIS_download
#Author: Tony Chang
#Date: 1.20.2015
#Abstract: opens the data_url_script file from the http://reverb.echo.nasa.gov/reverb website for the GYE MOD09A1 
#			from 2000-01 to 2015-01
#

import os
import pandas as pd
import urllib.request

filename = 'K:\\NASA_data\\data_url_script_2015-01-20_135939.txt'
urllist = pd.read_csv(filename, header = None)
workspace = 'K:\\NASA_data\\MOD09A1'
for i in range(len(urllist)):
#index for the name from the url starts at 62
	date = urllist[0][i][62:72]
	name = urllist[0][i][73:]
	savefile = '%s\\%s\\%s'%(workspace,date,name)
	if not os.path.exists('%s\\%s'%(workspace,date)):
		os.makedirs('%s\\%s'%(workspace,date))
	urllib.request.urlretrieve(urllist[0][i], savefile)
	urllib.request.urlcleanup() #clean up cache
