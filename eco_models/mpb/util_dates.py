'''
Created on Mar 11, 2010

@author: jared.oyler

##Modified by Tony Chang 03/21/2014 for python 3.x
'''
from datetime import datetime
from datetime import timedelta
from calendar import Calendar
import calendar
import numpy

A_DAY = timedelta(days=1)
A_WEEK = timedelta(days=7)
TWO_WEEKS = timedelta(days=14)

DATE="DATE"
YMD="YMD"
YEAR="YEAR"
MONTH="MONTH"
DAY="DAY"
YDAY="YDAY"

MTH_SRT_END_DATES = {1:(datetime(2003,1,1),datetime(2003,1,31)),
                     2:(datetime(2003,2,1),datetime(2003,2,28)),
                     3:(datetime(2003,3,1),datetime(2003,3,31)),
                     4:(datetime(2003,4,1),datetime(2003,4,30)),
                     5:(datetime(2003,5,1),datetime(2003,5,31)),
                     6:(datetime(2003,6,1),datetime(2003,6,30)),
                     7:(datetime(2003,7,1),datetime(2003,7,31)),
                     8:(datetime(2003,8,1),datetime(2003,8,31)),
                     9:(datetime(2003,9,1),datetime(2003,9,30)),
                     10:(datetime(2003,10,1),datetime(2003,10,31)),
                     11:(datetime(2003,11,1),datetime(2003,11,30)),
                     12:(datetime(2003,12,1),datetime(2003,12,31))}

def get_date_array_from_strings(strings,format="%Y-%m-%d"):
    
    return numpy.array([datetime.strptime(date,format) for date in strings])

def get_date_array(str_date,end_date):
    
    date=str_date
    dates=[]
    while date <= end_date:
        dates.append(date)
        date=date + A_DAY
    return numpy.array(dates)

def get_year_array(dates):
    years=numpy.zeros(dates.size,dtype=numpy.int32)
    date_nums = numpy.arange(dates.size)
    for x in date_nums:
        years[x] = dates[x].year
    return years

def get_year_day_array(dates):
    days=numpy.zeros(dates.size,dtype=numpy.int32)
    date_nums = numpy.arange(dates.size)
    for x in date_nums:
        days[x] = dates[x].timetuple().tm_yday
    return days

def get_month_day_array(dates):
    days=numpy.zeros(dates.size,dtype=numpy.int32)
    date_nums = numpy.arange(dates.size)
    for x in date_nums:
        days[x] = dates[x].day
    return days

def get_month_array(dates):
    months=numpy.zeros(dates.size,dtype=numpy.int32)
    date_nums = numpy.arange(dates.size)
    for x in date_nums:
        months[x] = dates[x].month
    return months

def get_ymd_array(dates):
    ymds=numpy.zeros(dates.size,dtype=numpy.int32)
    date_nums = numpy.arange(dates.size)
    for x in date_nums:
        ymds[x] = ymdL(dates[x])
    return ymds

def get_md_array(dates):
    mds=numpy.zeros(dates.size,dtype=numpy.int32)
    date_nums = numpy.arange(dates.size)
    for x in date_nums:
        mds[x] = mdL(dates[x])
    return mds

def get_mth_str_end_dates(mth,yr):
    num_days = calendar.monthrange(yr,mth)[1]
    str_date =  datetime(yr,mth,1)
    end_date = str_date + timedelta(days=int(num_days)-1)
    return str_date,end_date

def get_day_array(year,month):
    return numpy.arange(1,calendar.monthrange(year, month)[1]+1)

def dates_in_dates(dates,all_dates):
    ymds = get_ymd_array(dates)
    all_ymds = get_ymd_array(all_dates)
    
    return numpy.in1d(all_ymds, ymds, assume_unique=True)
    
def ymdL_to_date(ymd):
    return datetime.strptime(str(ymd),"%Y%m%d")

def mdL(date):
    return int(datetime.strftime(date,"%m%d"))
    
def ymdL(date):
    try:
        return int(datetime.strftime(date,"%Y%m%d"))
    except ValueError:
        return int("%d%02d%02d"%(date.year,date.month,date.day))

def get_days_metadata(srtDate=datetime(1948,1,1),endDate=datetime(2011,12,31)):

    dates = get_date_array(srtDate, endDate)
    days_metadata = numpy.recarray(dates.size,dtype=[(DATE,numpy.object_),(YEAR,numpy.int32),(MONTH,numpy.int32),(DAY,numpy.int32),(YDAY,numpy.int32),(YMD,numpy.int32)])
    days_metadata[DATE] = dates
    days_metadata[YEAR] = get_year_array(dates)
    days_metadata[MONTH] = get_month_array(dates)
    days_metadata[DAY] = get_month_day_array(dates)
    days_metadata[YDAY] = get_year_day_array(dates)
    days_metadata[YMD] = get_ymd_array(dates)
    return days_metadata

def get_days_metadata_dates(dates):

    days_metadata = numpy.recarray(dates.size,dtype=[(DATE,numpy.object_),(YEAR,numpy.int32),(MONTH,numpy.int32),(DAY,numpy.int32),(YDAY,numpy.int32),(YMD,numpy.int32)])
    days_metadata[DATE] = dates
    days_metadata[YEAR] = get_year_array(dates)
    days_metadata[MONTH] = get_month_array(dates)
    days_metadata[DAY] = get_month_day_array(dates)
    days_metadata[YDAY] = get_year_day_array(dates)
    days_metadata[YMD] = get_ymd_array(dates)
    return days_metadata 

def get_mth_metadata(str_yr,end_yr):
    
    dates = get_date_mth_array(str_yr, end_yr)
    mth_metadata = numpy.recarray(dates.size,dtype=[(DATE,numpy.object_),(YEAR,numpy.int32),(MONTH,numpy.int32),(YMD,numpy.int32)])
    mth_metadata[DATE] = dates
    mth_metadata[YEAR] = get_year_array(dates)
    mth_metadata[MONTH] = get_month_array(dates)
    mth_metadata[YMD] = get_ymd_array(dates)
    return mth_metadata

def get_date_yr_array(str_yr,end_yr):
    
    yrs = numpy.arange(str_yr,end_yr+1)
    
    dates=[]
    for yr in yrs:
            
        dates.append(datetime(yr,1,1))
    
    return numpy.array(dates)

def get_date_mth_array(str_yr,end_yr):
    
    yrs = numpy.arange(str_yr,end_yr+1)
    mths = numpy.arange(1,13)
    
    dates=[]
    for yr in yrs:
        
        for mth in mths:
            
            dates.append(datetime(yr,mth,1))
    
    return numpy.array(dates)    