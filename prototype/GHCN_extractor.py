#station data formatter script
import numpy as np
import csv

def stationreader():
	workspace = "D:\\chang\\climate_models\\station_data\\ghcn_daily\\"
	filename = "ghcnd-stations.txt"
	f = workspace + filename
	s =[]
	with open(f, newline = '') as f1:
		reader= csv.reader(f1)
		for row in reader:
			s.append(row)
	s = np.array(s)
	code = []
	lat = []
	lon = []
	ele = []
	st = []
	name = []
	gsnflag = []
	hcnflag =[]
	wmoid = []
	for i in range(len(s)):
		code.append(s[i][0][:11])
		lat.append(float(s[i][0][12:20]))
		lon.append(float(s[i][0][21:30]))
		ele.append(float(s[i][0][31:37]))
		st.append(s[i][0][38:40])
		name.append(s[i][0][41:71])
		gsnflag.append(s[i][0][72:75])
		hcnflag.append(s[i][0][76:79])
		wmoid.append(s[i][0][81:85])
	station = {'code' : np.array(code), 'lat' : np.array(lat), 'lon' : np.array(lon), 'ele':np.array(ele), 'st':np.array(st), 'name':np.array(name), 'gsn':np.array(gsnflag), 'hcn':np.array(hcnflag), 'wmo':np.array(wmoid)}
	l = np.array([code,lat,lon, ele, st, name, gsnflag, hcnflag, wmoid])
	return(station)

#GYE extent
#xmin = -112.438; xmax = -108.271; ymin = 42.262; ymax = 46.187
def stationfilter(station, xmin,xmax,ymin,ymax): #filters the station list by the bounding box extent	
	fstation = []
	dt = [('code', 'S11'),('lat', 'f8'),('lon','f8'),('ele','f8'),('st','S2'),('name', 'S30'),('gsn','S3'),('hcn','S3'),('wmo','f5')]
	for i in range(len(station['code'])):
			sy = station['lat'][i]
			sx = station['lon'][i]
			if ((sx>xmin) and (sx<xmax) and (sy>ymin) and (sy<ymax)):
				fstation.append([station['code'][i],station['lat'][i],station['lon'][i],station['ele'][i],station['st'][i],station['name'][i],station['gsn'][i],station['hcn'][i],station['wmo'][i]])
				#fstation.append(station['code'][i])
	fstation = np.array(fstation)
	return(fstation)

def datareader(stationid): #input station desired 
	workspace = "d:\\chang\\climate_models\\station_data\\ghcn_daily\\ghcnd_all\\ghcnd_all\\"
	data = []
	filename = stationid + '.dly'
	f = workspace + filename
	s =[]
	with open(f, newline = '') as f1:
		reader= csv.reader(f1)
		for row in reader:
			s.append(row)
	s = np.array(s)
	data.append(s)
	return(np.array(data))

def datagather(fstation):
	dataset = []
	for i in range(len(fstation)):
		dataset.append(datareader(fstation[i][0])[0])
	return(np.array(dataset))
	
def databyvar(dataset,var):
	#filters the dataset to contain only one var climate element
	"""
	var = 
	PRCP = Precipitation (tenths of mm) 
	SNOW = Snowfall (mm)
	SNWD = Snow depth (mm)
	TMAX = Maximum temperature (tenths of degrees C)
	TMIN = Minimum temperature (tenths of degrees C)
	"""
	fdataset = []
	for i in range(len(dataset)):
		stationdata = []
		count = 0
		for j in range(len(dataset[i])):
			if (dataset[i][j][0][17:21] == var):
				stationdata.append(dataset[i][j])
				count +=1
		if (count != 0):
			fdataset.append(np.array(stationdata))
	return(np.array(fdataset))

def datasummary(dataset, timescale):
	"""summarizes the data by month('m'), season('s'), or annually('a') at the individual station level
	(note dataset must be pre-filtered by the databyvar function)"""
	if (timescale == 'm'): #monthly case
		return(monsummary(dataset))
	elif (timescale == 's'): #seasonal case
		return(seasonsummary(dataset))
	elif (timescale == 'a'): #annual case
		return(annualsummary(dataset))

def regionalsummary(dataset,timescale):
	"""summarizes the data for the entire region by month('m'), season('s'), or annually('a') 
	(note dataset must be pre-filtered by the databyvar function)"""
	if (timescale == 'a'): #annual case
		adata = datasummary(dataset, timescale)
		r = np.arange(1895,2013) #take only the dataset that runs fron 1895 to 2012
		z = np.zeros((2,2013-1895))
		rannualsum = np.vstack((r, z)).T
		for i in range(len(adata)):
			for j in range(len(adata[i])):
				currentyear = float(adata[i][j][1])
				currentval = float(adata[i][j][2])
				if (np.isnan(currentval)):
					continue
				elif (currentyear>=1895 and currentyear<=2012): #if the value is not np.nan and within the year boundaries
					rannualsum[np.where(rannualsum == currentyear)[0]][0][1] += currentval
					rannualsum[np.where(rannualsum == currentyear)[0]][0][2] += 1
	rout = np.array([rannualsum[:,0], rannualsum[:,1]/rannualsum[:,2], rannualsum[:,2]]).T
	return(rout)
			
		
def monsummary(dataset): 
	"""returns the monthly summary of the filtered dataset"""
	datasum = []
	months = np.arange(1,13)
	for i in range(len(dataset)):
		stationdata = []
		for j in range(len(dataset[i])):
			k = 21
			label = dataset[i][j][0][:k]
			monmean = 0
			moncount = 0
			flagcount = 0
			nullcount = 0
			while (k < len(dataset[i][j][0])-1):
				dailyval = float(dataset[i][j][0][k:k+5])
				qflag = dataset[i][j][0][k+5+1]
				if (dailyval == -9999):
					nullcount += 1 #count the number of null values
				if (dailyval != -9999 and qflag == (' ' or 'N')): #check for null values or failed quality flags
					monmean += dailyval
					moncount += 1
				else:
					flagcount += 1 #count the missing value or flag for record
				k += 8 #go to next day iteration
			if (flagcount < 15): #if the flag and null sum count are less than 15
				monmean = monmean/moncount # calculate the average month value
				stationdata.append([label,monmean, flagcount, nullcount, flagcount-nullcount])
			else:
				monmean = np.nan #if the number of flagged values are greater than or equal to 15, then discard then set month average to np.nan
				stationdata.append([label,monmean, flagcount, nullcount, flagcount-nullcount])
		datasum.append(stationdata)
	return(np.array(datasum))

def seasonsummary(dataset):
	monsum = monsummary(dataset)
	seasonsum = []
	for i in range(len(monsum)):
		firstyear = int(monsum[i][0][0][11:15])
		lastyear = int(monsum[i][-1][0][11:15]) #find the year range of the dataset
		y = np.arange(firstyear, lastyear+1)
		s = np.zeros((8,(lastyear-firstyear+1))) #season array has addition 4 elements to count number of valid month inputs
		annualseason = np.vstack((y,s)).T #create a season array for each year 
		for j in range(len(monsum[i])):
			currentyear = int(monsum[i][j][0][11:15])
			currentmonth = int(monsum[i][j][0][15:17])
			currentval = monsum[i][j][1]
			if (currentmonth == 12 and currentyear != firstyear): #December of the previous year considered winter of current year
				annualseason[np.where(annualseason==currentyear-1)[0][0]][1] += currentval
				annualseason[np.where(annualseason==currentyear-1)[0][0]][5] += 1
			elif (currentmonth == (1 or 2)): #winter case
				annualseason[np.where(annualseason==currentyear)[0][0]][1] += currentval
				annualseason[np.where(annualseason==currentyear)[0][0]][5] += 1
			elif (currentmonth == (3 or 4 or 5)): #spring case
				annualseason[np.where(annualseason==currentyear)[0][0]][2] += currentval
				annualseason[np.where(annualseason==currentyear)[0][0]][6] += 1
			elif (currentmonth == (6 or 7 or 8)): #summer case
				annualseason[np.where(annualseason==currentyear)[0][0]][3] += currentval
				annualseason[np.where(annualseason==currentyear)[0][0]][7] += 1
			elif (currentmonth == (9 or 10 or 11)): #fall case
				annualseason[np.where(annualseason==currentyear)[0][0]][4] += currentval
				annualseason[np.where(annualseason==currentyear)[0][0]][8] += 1	
		seasonadd = []
		label = monsum[i][0][0][:11]+monsum[i][0][0][17:21]
		for k in range(len(annualseason)):
			win = annualseason[k][1]/annualseason[k][5]
			spr = annualseason[k][2]/annualseason[k][6]
			sum = annualseason[k][3]/annualseason[k][7]
			fal = annualseason[k][4]/annualseason[k][8]
			seasonadd.append([label,annualseason[k][0], win, spr, sum, fal])
		seasonsum.append(np.array(seasonadd))
	return(np.array(seasonsum)) #returns a seasonal summer array in the form ('station_vartype', 'year','winterval', 'springval', 'summerval', 'fallval')
								#nan will be report for values where a month is missing for that season

def annualsummary(dataset):
	monsum = monsummary(dataset)
	annualsum = []
	for i in range(len(monsum)):
		firstyear = int(monsum[i][0][0][11:15])
		lastyear = int(monsum[i][-1][0][11:15])
		y = np.arange(firstyear, lastyear +1)
		s = np.zeros((3, (lastyear-firstyear +1)))
		yearval = np.vstack((y,s)).T
		for j in range(len(monsum[i])):
			currentyear= int(monsum[i][j][0][11:15])
			currentval = monsum[i][j][1]
			if (np.isnan(currentval)):
				yearval[np.where(yearval==currentyear)[0][0]][3] += 1 #count how many null months there are
			else:
				yearval[np.where(yearval==currentyear)[0][0]][1] += currentval
				yearval[np.where(yearval==currentyear)[0][0]][2] += 1
		yearadd =[]
		label = monsum[i][0][0][:11]+monsum[i][0][0][17:21]
		for k in range(len(yearval)):
			if (yearval[k][0]!= 2013 or yearval[k][3]>3): #neglect 2013 as the year is not complete yet, or if more than 3 months are missing
				yearadd.append(([label,yearval[k][0], yearval[k][1]/yearval[k][2], yearval[k][3]]))
		annualsum.append(np.array(yearadd))
	return(np.array(annualsum)) #returns an annualsum array in the form ('station_vartype', 'year', 'val', 'number of null values') 
								#nan will be report for values where a year is missing for that season
#-------MAIN-------#
xmin = -112.438; xmax = -108.271; ymin = 42.262; ymax = 46.187
stationlist = stationreader()
fstation = stationfilter(stationlist, xmin, xmax, ymin, ymax)
dataset = datagather(fstation)
var = 'TMIN'
tmindata = databyvar(dataset,var)
atmin = datasummary(tmindata,'a')
