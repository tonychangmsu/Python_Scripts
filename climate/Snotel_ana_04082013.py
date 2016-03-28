# -*- coding: utf-8 -*--------------------------------------------------------
"""
Created on Wed Jan 30 10:12:42 2013

@author: tony.chang
Extract SNOtel data and process for averages

"""
import numpy as np
import scipy.stats.stats as stats
import matplotlib.pyplot as plt
#import scikits.timeseries as ts
#import scikits.timeseries.lib.plotlib as tpl
import datetime as dt
import matplotlib.dates as mdates
import pandas as pd
#Build class to store dataset
class SnotelSite(object):
    #initialize function to construct PRISMdata class
    def __init__(self, st, code, lat,lon, elev, name, data, monsum):
        self.st = st #state
        self.code = code #station code
        self.lat = lat #latitude
        self.lon = lon #longitude
        self.elev = elev #elevation in meters
        self.name = name #site name
        self.data = data #snotel data in the Snotel Data class 
        self.monsum = monsum #monthly summary of Snotel dataset

class SnotelData(object):
    #initialize function to construct PRISMdata class
    def __init__(self, day, month, year, pill, prec,tmax, tmin, tavg, prcp):
        self.day = day
        self.month = month
        self.year = year
        self.pill = pill #pillow which measures the snow water equivalent in inches
        self.prec = prec #Accumulated precipitation on the water year basis (10/10 -9/30) in inches
        self.tmax = tmax
        self.tmin = tmin
        self.tavg = tavg
        self.prcp = prcp

class SnotelMonthly(object):
    def __init__(self, year, month,SWE, melt,zerosnowdays,zerosnowdates):
        self.year = year
        self.month = month
        self.SWE = SWE
        self.melt = melt
        self.zerosnowdays = zerosnowdays
        self.zerosnowdates = zerosnowdates

'''
def MonthlyDataFormat(data, fname = 'Temp_mon.csv', write = 'n'):
	#creates a .csv file that stores extent snotel data for pandas format
	numstations = len(data)
	fdata = []
	indexd = []
	for i in range(numstations):
		numyears = len(data[i].monsum)
		for j in range(numyears):
			nummonths = len(data[i].monsum[j].month)
			y = data[i].monsum[j].year
			for k in range(nummonths):
				frow = []
				m = data[i].monsum[j].month[k]
				d= 1
				ft = dt.date(y,m,d)
				datestr = ft.strftime('%Y-%m-%d')
				inddate = np.datetime64(datestr)
				indexd.append(inddate)
				inddate = np.datetime64(datestr)
				indexd.append(inddate)
				#frow.append(datestr) #first copy the date
				frow.append(data[i].code) # copy code
				frow.append(data[i].name) # copy name
				frow.append(data[i].lat)
				frow.append(data[i].lon)
				frow.append(data[i].elev)
				frow.append(data[i].monsum[j].SWE[k])
				frow.append(data[i].monsum[j].melt[k])
				frow.append(data[i].monsum[j].zerosnowdays[k])
				frow.append(data[i].monsum[j].zerosnowdates[k])
				frow.append(data[i].data[j]['TMIN']) 
				frow.append(data[i].data[j]['TAVG']) 
				frow.append(data[i].data[j]['PRCP']) 
				frow.append(data[i].data[j]['PILL']) 
				frow.append(data[i].data[j]['PREC']) 
				
				frow = np.array(frow)
				fdata.append(frow)
	fdata = np.array(fdata)
	filepath = 'D:\\CHANG\\Climate_Models\\Station_Data\\SNOTEL\\Python_format\\' + fname
	#head = "date,code,name,lat,lon,elev,tmax,tmin,tavg,ppt,pill,cumppt"	
	labels = ["code","name","lat","lon","elev","tmax","tmin","tavg","ppt","pill","cumppt"]
	#format = "%s %s %s %f %f %f %f %f %f %f %f %f"
	#np.savetxt(filepath,fdata,delimiter=",", header = head, fmt = format)
	indexd = np.array(indexd)
	snotelframe = pd.DataFrame(fdata,index = indexd, columns =labels)
	snotelframe.index.name = 'date'
	if (write=='y'):
		snotelframe.to_csv(filepath,labels)
	return(fdata, snotelframe)
'''


	
#initialize-------------------------------------------------------------------

"""---------------------------------------------------------------------------
----------------------------UTILITIES-----------------------------------------
---------------------------------------------------------------------------"""
def DailyDataFormat(data,fname= 'Temp_day.csv',write = 'n'):
	#creates a .csv file that stores extent snotel data for pandas format
	numstations = len(data)
	fdata = []
	indexd = []
	for i in range(numstations):
		numdatapoints = len(data[i].data)
		for j in range(numdatapoints):
			frow = []
			m,d,y = Dateformat(data[i].data[j]['DATE'])
			ft = dt.date(y,m,d)
			datestr = ft.strftime('%Y-%m-%d')
			inddate = np.datetime64(datestr)
			indexd.append(inddate)
			#frow.append(datestr) #first copy the date
			frow.append(data[i].code) # copy code
			frow.append(data[i].name) # copy name
			frow.append(data[i].lat)
			frow.append(data[i].lon)
			frow.append(data[i].elev)
			frow.append(data[i].data[j]['TMAX']) 
			frow.append(data[i].data[j]['TMIN']) 
			frow.append(data[i].data[j]['TAVG']) 
			frow.append(data[i].data[j]['PRCP']) 
			frow.append(data[i].data[j]['PILL']) 
			frow.append(data[i].data[j]['PREC']) 
			if (data[i].data[j]['PILL']>0):
				frow.append(0) #note the zero snowdays
			else:
				frow.append(1)
			frow = np.array(frow)
			fdata.append(frow)
	fdata = np.array(fdata)
	filepath = 'D:\\CHANG\\Climate_Models\\Station_Data\\SNOTEL\\Python_format\\' + fname
	#head = "date,code,name,lat,lon,elev,tmax,tmin,tavg,ppt,pill,cumppt"	
	labels = ["code","name","lat","lon","elev","tmax","tmin","tavg","ppt","pill","cumppt", "zsd"]
	#format = "%s %s %s %f %f %f %f %f %f %f %f %f"
	#np.savetxt(filepath,fdata,delimiter=",", header = head, fmt = format)
	indexd = np.array(indexd)
	snotelframe = pd.DataFrame(fdata,index = indexd, columns =labels)
	if (write=='y'):
		snotelframe.to_csv(filepath,labels)
	return(fdata, snotelframe)

def All_monsummary(data,Beginyear, Endyear,min_elev = 0,max_elev = 9999,write='n'): #specify the minimum elevation of data points
	swe_m = np.zeros(((Endyear-Beginyear)+1,12)) #declare matrices to hold variable values for length of years by mon
	melt_m = np.zeros(((Endyear-Beginyear)+1,12))
	zsn_m = np.zeros(((Endyear-Beginyear)+1,12))
	s_count = np.zeros((3,(Endyear-Beginyear)+1,12))
	for i in range(len(data)):
		if (data[i].elev>=min_elev and data[i].elev<= max_elev): #check if within the elevation range
			for j in range(len(data[i].monsum)):
				if (data[i].monsum[j].year >=Beginyear and data[i].monsum[j].year <= Endyear): #check if within the year range
					yr = data[i].monsum[j].year-Beginyear
					for k in range(len(data[i].monsum[j].month)):
						mon = data[i].monsum[j].month[k]-1
						if (~np.isnan(data[i].monsum[j].SWE[k])):
							swe_m[yr][mon] += data[i].monsum[j].SWE[k]
							s_count[0][yr][mon] +=1 #add a station sample count for that data point
						if (~np.isnan(data[i].monsum[j].melt[k])):
							melt_m[yr][mon] += data[i].monsum[j].melt[k]
							s_count[1][yr][mon] +=1 #add a station sample count for that data point
						if (~np.isnan(data[i].monsum[j].melt[k])):
							zsn_m[yr][mon] += data[i].monsum[j].zerosnowdays[k]
							s_count[2][yr][mon] +=1 #add a station sample count for that data point
	for i in range(len(s_count)):
		s_count[i][np.where(s_count[i]==0)] = np.nan #find where there are no station counts
	#find the means for all the summaries
	swe_mean = np.array(swe_m/s_count[0])
	melt_mean = np.array(melt_m/s_count[1])
	zsn_mean = np.array(zsn_m/s_count[2])
	sc = s_count[0]
	datearray = np.arange(np.datetime64(str(Beginyear)+'-01','M'), np.datetime64(str(Endyear)+'-12','M')+1)
	#find the means for the prec, tmin, tmax, tavg,pill.... #section added 4/8/2013
	#==============================================================================#
	tmax_m = np.zeros(((Endyear-Beginyear)+1,12))
	tmin_m = np.zeros(((Endyear-Beginyear)+1,12))
	tavg_m = np.zeros(((Endyear-Beginyear)+1,12))
	ppt_m = np.zeros(((Endyear-Beginyear)+1,12))
	pill_m = np.zeros(((Endyear-Beginyear)+1,12))
	cump_m = np.zeros(((Endyear-Beginyear)+1,12))
	s_count = np.zeros((6,(Endyear-Beginyear)+1,12)) #make a separate counter for each considering that some are nan
	for i in range(len(data)):
		if (data[i].elev >= min_elev and data[i].elev<=max_elev):
			for j in range(len(data[i].data)): #loop through the entire dataset
				cdate = Dateformat(data[i].data[j]['DATE'])	#take in the date 
				if (cdate[2] >=Beginyear and cdate[2] <= Endyear): #check if within the year range
					yr = cdate[2]-Beginyear
					mon = cdate[0]-1
					if (~np.isnan(data[i].data[j]['TMAX'])):
						tmax_m[yr][mon] += data[i].data[j]['TMAX']
						s_count[0][yr][mon] +=1 #add a station count for that data point
					if (~np.isnan(data[i].data[j]['TMIN'])):
						tmin_m[yr][mon] += data[i].data[j]['TMIN']
						s_count[1][yr][mon] +=1 #add a station count for that data point
					if (~np.isnan(data[i].data[j]['TAVG'])):
						tavg_m[yr][mon] += data[i].data[j]['TAVG']
						s_count[2][yr][mon] +=1 #add a station count for that data point
					if (~np.isnan(data[i].data[j]['PRCP'])):
						ppt_m[yr][mon] += data[i].data[j]['PRCP']
						s_count[3][yr][mon] +=1 #add a station count for that data point
					if (~np.isnan(data[i].data[j]['PILL'])):
						pill_m[yr][mon] += data[i].data[j]['PILL']
						s_count[4][yr][mon] +=1 #add a station count for that data point
					if (~np.isnan(data[i].data[j]['PREC'])):
						cump_m[yr][mon] += data[i].data[j]['PREC']
						s_count[5][yr][mon] +=1 #add a station count for that data point
	for i in range(len(s_count)):
		s_count[i][np.where(s_count[i]==0)] = np.nan #find where there are no station counts
	#find the means for all the summaries
	tmax_mean = np.array(tmax_m/s_count[0])
	tmin_mean = np.array(tmin_m/s_count[1])
	tavg_mean = np.array(tavg_m/s_count[2])
	ppt_mean = np.array(ppt_m/s_count[3])	
	pill_mean = np.array(pill_m/s_count[4])
	cump_mean = np.array(cump_m/s_count[5])
	mdict = {'swe':swe_mean,'melt':melt_mean,'zsn':zsn_mean, 'tmax':tmax_mean, 'tmin':tmin_mean, 'tavg':tavg_mean, 'ppt':ppt_mean, 'pill':pill_mean,'cump':cump_mean}
	#create a pandas date frame for csv output and figure creation
	dt = pd.date_range(str(Beginyear) +"-1", str(Endyear+1) +"-1", freq= "M") #date array
	swe_mean = np.reshape(swe_mean, 12*(Endyear-Beginyear+1))
	melt_mean = np.reshape(melt_mean, 12*(Endyear-Beginyear+1))
	zsn_mean = np.reshape(zsn_mean, 12*(Endyear-Beginyear+1))
	tmax_mean = np.reshape(tmax_mean, 12*(Endyear-Beginyear+1))
	tmin_mean = np.reshape(tmin_mean, 12*(Endyear-Beginyear+1))
	tavg_mean = np.reshape(tavg_mean, 12*(Endyear-Beginyear+1))
	pill_mean = np.reshape(pill_mean, 12*(Endyear-Beginyear+1))
	ppt_mean = np.reshape(ppt_mean, 12*(Endyear-Beginyear+1))
	cump_mean = np.reshape(cump_mean, 12*(Endyear-Beginyear+1))
	sc = np.reshape(sc, 12*(Endyear-Beginyear+1))
	labels = ['swe', 'melt', 'zsn', 'tmax', 'tmin', 'tavg', 'pill', 'ppt', 'cump', 'num_stations']
	msummary = np.array((swe_mean,melt_mean,zsn_mean, tmax_mean, tmin_mean, tavg_mean, pill_mean, ppt_mean,cump_mean,sc))
	msframe = pd.DataFrame(msummary.T, index = dt, columns = labels)
	if (write=='y'):
		fname = 'monthly_snotel.csv'
		filepath ='D:\\CHANG\\Climate_Models\\Station_Data\\SNOTEL\\Python_format\\' + fname
		msframe.to_csv(filepath)
	return(mdict, msframe, dt)
	
def Snoteldataextract(data): #formats the snotel data into a class 
    sdata = []    
    data.fill_value = np.nan
    data= data.filled()
    for i in range(len(data)):
        date = data[i][0]
        day = int((date-(date%10000))/10000)
        year =int(((date%1000)%100))
        month = int(((date%1000)-year)/100)
        if (year <=13): #add 1900 or 2000 depending on the last two digits. Since it is currently 2013, any values =< 13 must be from the 2000 millenium
            year = year + 2000
        else:
            year = year + 1900
        sdataarray = SnotelData(day, month, year, data[i][1],data[i][2],data[i][3],data[i][4],data[i][5],data[i][6])
        sdata.append(sdataarray)
    return(sdata)

def Datafilecreator(snotelsite):
    datereq = 10199
    statereq = 'MT'
    inforeq = []
    for i in range(len(snotelsite)):
        if (snotelsite[i].st == statereq):
            for j in range(len(snotelsite[i].data)):
                if (snotelsite[i].data[j][0] == datereq):
                    infoarray = [snotelsite[i].name, snotelsite[i].code,snotelsite[i].lat,snotelsite[i].lon,snotelsite[i].elev,snotelsite[i].data[j][1], snotelsite[i].data[j][4]]
                    inforeq.append(infoarray)
    return(inforeq)

def Datawrite(data):
    f = open('D:\\CHANG\\Python_Scripts\\Output\\output.dat', 'w')
    header = 'SITE_NAME CODE LAT LONG ELEV SWE TMIN\n'
    f.write(header)
    for i in range(len(data)):
        for j in range(len(data[i])):            
            o = np.array(data[i])            
            f.write(o[j])
            f.write(' ')
        f.write('\n')
    f.close()
    return()

def SnotelIn(extent): #returns all snotel data within the area extent array specified    
    maxy = extent[3]
    miny = extent[1]
    maxx = extent[2]
    minx = extent[0]
    filename = "D:\\CHANG\\Climate_models\\Station_data\\Snotel\\nrcs_snotel_list01312013.txt"
    dt = [('ST_ABBR', 'S2'), ('ST', 'f8'), ('CTY', 'f8'), ('Type', 'S4'), ('HUC', 'f8'), ('Station_Code', 'S6'), ('Lat', 'f8'), ('Long', 'f8'), ('Elev_M', 'f8'), ('Elev_FT', 'f8'),('SiteName', 'S30')]
    stationlist = np.genfromtxt(filename,dtype=dt, names=True)

    """import SNOTEL data"""
    d = [('DATE', 'f8'), ('PILL', 'f8'), ('PREC', 'f8'),('TMAX', 'f8'),('TMIN', 'f8'),('TAVG', 'f8'),('PRCP', 'f8')]
    m = (" "," "," "," "," "," "," ")
    f = (np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan)
        
    dataarray = []
    snotelsite = []
    for i in range(len(stationlist)):
        sy = stationlist[i]['Lat']
        sx = stationlist[i]['Long']
        if ((sx>minx) and (sx<maxx) and (sy>miny) and (sy<maxy)):
            datafile = "D:\\CHANG\Climate_models\\station_data\\snotel\\data\\" + stationlist[i]['ST_ABBR'].decode('utf-8') + "\\" + stationlist[i]['Station_Code'].decode('utf-8') +"_all.txt" #code syntax change for python 3.3
            try:    
                data = np.genfromtxt(datafile, delimiter ='\t',dtype=d, skip_header=1, missing_values = m, filling_values = f,usemask=True )    
                dataarray.append(data)
                #sdata = Snoteldataextract(data)
                data.fill_value = np.nan
                data= data.filled()
                monsum = MonSummary(data)
                stationarray = SnotelSite(stationlist[i]['ST_ABBR'].decode('utf-8'),stationlist[i]['Station_Code'].decode('utf-8'),stationlist[i]['Lat'], stationlist[i]['Long'], stationlist[i]['Elev_M'], stationlist[i]['SiteName'].decode('utf-8'),data,monsum)
                snotelsite.append(stationarray)
            except IOError:
                continue
    return(snotelsite)
    
def Dateformat(date): #takes in the 5 to 6 digit date and returns month, day, and year
    month = int((date-(date%10000))/10000)
    year =int(date%100)
    day = int(((date%10000)-year)/100)
    if (year<=13): #since the latest year is 2013 currently
        year = year + 2000
    else:
        year = year + 1900
    return(month,day,year)
    
def MonSummary(data):  #calculates the monthly accumulated snow, melt amount, and snow free days
    SWEarray = [] #12x1 array
    zerosnowdate = []
    zerosnowday =[]
    meltarray = []
    stationsummary = []
    montharray = []
    date = data['DATE']
    SWE = data['PILL']
    zerosnow = 0
    accum = 0
    melt = 0
    
    for i in range(1,len(date)):
        m,d,y = Dateformat(date[i])
        mp,dp,yp = Dateformat(date[i-1])
        if (y==yp):
            if (m==mp):
                dSWE = SWE[i]- SWE[i-1]
                if (SWE[i-1] == 0):
                    zerosnow = zerosnow + 1         #counts days without snow
                    zerosnowdate.append(date[i-1])
                if (dSWE >=0):
                    accum = accum + dSWE        #adds new snow counter
                else:
                    melt = melt + dSWE        #adds melt counter
            else:
                dSWE = SWE[i]- SWE[i-1]
                if (SWE[i-1] == 0):
                    zerosnow = zerosnow + 1         #counts days without snow
                    zerosnowdate.append(date[i-1])
                if (dSWE >=0):
                    accum = accum + dSWE        #adds new snow counter
                else:
                    melt = melt + dSWE        #adds melt counter
                SWEarray.append(accum)
                meltarray.append(melt)
                zerosnowday.append(zerosnow)
                montharray.append(mp)
                accum = 0
                melt = 0
                zerosnow = 0
            if (i== len(date)-1):
                SWEarray.append(accum)
                meltarray.append(melt)
                zerosnowday.append(zerosnow)
                montharray.append(mp)
        else:
            dSWE = SWE[i]- SWE[i-1]
            if (SWE[i-1] == 0):
                zerosnow = zerosnow + 1         #counts days without snow
                zerosnowdate.append(date[i-1])
            if (dSWE >=0):
                accum = accum + dSWE        #adds new snow counter
            else:
                melt = melt + dSWE        #adds melt counter
            SWEarray.append(accum)
            meltarray.append(melt)
            zerosnowday.append(zerosnow)
            montharray.append(mp)
            stationsummary.append(SnotelMonthly(yp,np.array(montharray),np.array(SWEarray),np.array(meltarray),np.array(zerosnowday),np.array(zerosnowdate)))
            SWEarray = [] #12x1 array
            zerosnowdate = []
            zerosnowday =[]
            meltarray = []
            montharray = []
            accum = 0
            melt = 0
            zerosnow = 0
        if (i== len(date)-1):
            stationsummary.append(SnotelMonthly(yp,np.array(montharray),np.array(SWEarray),np.array(meltarray),np.array(zerosnowday),np.array(zerosnowdate)))
    return(stationsummary)

"""---------------------------------------------------------------------------------
-------------------------------TEST FUNCTIONS---------------------------------------
---------------------------------------------------------------------------------"""

def Nosnowplot(data,beginyear,endyear):
    annualzerosnow = np.zeros(endyear+1-beginyear)   
    years= np.arange(beginyear, endyear+1)
    stationcount = []
    for k in range(len(annualzerosnow)):    
        count = 0
        for i in range(len(data)):
            for j in range(len(data[i].monsum)):
                if(data[i].monsum[j].year==years[k]):
                    count +=1
                    annualzerosnow[k]=sum(data[i].monsum[j].zerosnowdays)+annualzerosnow[k]
        annualzerosnow[k] =annualzerosnow[k]/count #returns the average number of zerosnowdays for all stations
        stationcount.append(count)
    plt.figure()
    plt.subplot(2,2,(1,2))
    plt.plot(years, annualzerosnow)
    plt.subplot(2,2,3)
    plt.ylabel('Number of stations')
    plt.xlabel('Year')
    plt.bar(years, stationcount)
    plt.show()
    return(annualzerosnow,np.array(stationcount), years)

def SWEmeltplot(data,beginyear,endyear):
    SWE = np.zeros(endyear+1-beginyear)   
    melt = np.zeros(endyear+1-beginyear)   
    years= np.arange(beginyear, endyear+1)
    stationcount = []
    for k in range(len(SWE)):    
        count = 0
        for i in range(len(data)):
            for j in range(len(data[i].monsum)):
                if(data[i].monsum[j].year==years[k]):
                    count +=1
                    SWE[k]=np.nansum(data[i].monsum[j].SWE)+SWE[k]
                    melt[k]=np.nansum(data[i].monsum[j].melt)+melt[k]
        SWE[k] =SWE[k]/count #returns the mean SWE for all stations
        melt[k] =melt[k]/count #returns the mean melt for all stations
        stationcount.append(count)
    plt.figure()
    plt.subplot(3,1,1)
    plt.plot(years, SWE)
    plt.title('SWE')
    plt.subplot(3,1,2)    
    plt.plot(years, melt)
    plt.title('melt')
    plt.subplot(3,1,3)
    plt.bar(years, stationcount)
    return(SWE,melt)
    
def ElevComp(data,elevdiv, beginyear, endyear):                    
    ''' It might be useful to consider 2743m +- 152m, as this is the definition of treeline
    by Despain 1990'''    
    egroup1 = []
    egroup2 = []
    for i in range(len(data)):
        if (data[i].elev<=elevdiv):
            egroup1.append(data[i])
        else:
            egroup2.append(data[i])
    Nosnowplot(egroup1, beginyear, endyear)
    plt.title("Stations below " + str(elevdiv))
    Nosnowplot(egroup2, beginyear, endyear)
    plt.title("Stations above " + str(elevdiv))
    return()

def ElevCompSWE(data,elevdiv, beginyear, endyear):                    
    ''' It might be useful to consider 2743m +- 152m, as this is the definition of treeline
    by Despain 1990'''    
    egroup1 = []
    egroup2 = []
    for i in range(len(data)):
        if (data[i].elev<=elevdiv):
            egroup1.append(data[i])
        else:
            egroup2.append(data[i])
    SWE1,melt1 = SWEmeltplot(egroup1, beginyear, endyear)
    plt.title("Stations below " + str(elevdiv))
    SWE2,melt2 = SWEmeltplot(egroup2, beginyear, endyear)
    plt.title("Stations above " + str(elevdiv))
    return(SWE1,SWE2,melt1,melt2)

def Plotzerosnowdays(data,beginyear,endyear): #function does not work with python 3....maybe problem with inputs
    date1 = dt.date(beginyear,1,1)
    date2 = dt.date(endyear+1,1,1)
    delta = dt.timedelta(days=1)    
    datearray = mdates.drange(date1,date2,delta) #create date array
    zerosnowvalues = np.zeros(len(datearray)) #store zero snow days
    for i in range(len(datearray)):
        for j in range(len(data)):
            m,d,y = Dateformat(data[j])
            datenum = mdates.date2num(dt.date(y,m,d))
            if (datearray[i] == datenum):
                zerosnowvalues[i] = 1  #count day as zero snow
    datearray = mdates.num2date(datearray)
    plt.plot(datearray,zerosnowvalues)
    plt.ylim(0,2)
    return()   

def Regionsummary(data,beginyear,endyear):
	npoints = 0
	nstations = 0
	for i in range(len(data)): #loop through all stations
		count = 0 #begin counter
		for j in range(len(data[i].data['DATE'])):
			m,d,y = Dateformat(data[i].data[j]['DATE'])
			if ((y >= beginyear) and (y<=endyear)):
				count +=1
		if (count > npoints):
			npoints = count
			longestsiteindex = i
	return(npoints,longestsiteindex)
			
	
	
"""---------------------------------------------------------------------------
----------------------------------MAIN----------------------------------------
---------------------------------------------------------------------------"""

AOA = [-112.436, 42.252, -108.263, 46.182]  #Area of interest (GYE)
AOA = [-116,44,-110,49]
datalist = SnotelIn(AOA)

