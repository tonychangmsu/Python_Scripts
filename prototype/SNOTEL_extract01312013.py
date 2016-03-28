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
        
#initialize-------------------------------------------------------------------

"""---------------------------------------------------------------------------
----------------------------UTILITIES-----------------------------------------
---------------------------------------------------------------------------"""
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
            datafile = "D:\\CHANG\Climate_models\\station_data\\snotel\\data\\" + stationlist[i]['ST_ABBR'] + "\\" + stationlist[i]['Station_Code'] +"_all.txt"
            try:    
                data = np.genfromtxt(datafile, delimiter ='\t',dtype=d, skip_header=1, missing_values = m, filling_values = f,usemask=True )    
                dataarray.append(data)
                #sdata = Snoteldataextract(data)
                data.fill_value = np.nan
                data= data.filled()
                monsum = MonSummary(data)
                stationarray = SnotelSite(stationlist[i]['ST_ABBR'],stationlist[i]['Station_Code'],stationlist[i]['Lat'], stationlist[i]['Long'], stationlist[i]['Elev_M'], stationlist[i]['SiteName'],data,monsum)
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
    #SWE = [.1,0,.5,0., 0., .1, .1, .15, .5, .75, .65, .9]
    #date = [100112,100212,100312,12913, 13013, 13113, 20113, 20213, 20313, 20413, 20513,102513]
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
    plt.bar(years, stationcount)
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
                    SWE[k]=sum(data[i].monsum[j].SWE)+SWE[k]
                    melt[k]=sum(data[i].monsum[j].melt)+melt[k]
        SWE[k] =SWE[k]/count #returns the average number of zerosnowdays for all stations
        melt[k] =melt[k]/count #returns the average number of zerosnowdays for all stations
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

def Plotzerosnowdays(data,beginyear,endyear):
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
        
"""---------------------------------------------------------------------------
----------------------------------MAIN----------------------------------------
---------------------------------------------------------------------------"""

AOA = [-112.436, 42.252, -108.263, 46.182]  #Area of interest (GYE)
data = SnotelIn(AOA)

