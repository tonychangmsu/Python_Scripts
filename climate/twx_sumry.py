'''
Created on Jan 30, 2014
Utility methods for calculating statistical summaries of TopoWx data
@author: jared.oyler
'''

import numpy as np
from util_dates import YEAR,MONTH
import util_dates as utld
from scipy import stats
import osgeo.gdal as gdal
import osgeo.gdalconst as gdalconst
import osgeo.osr as osr
from netCDF4 import Dataset, num2date

MISSING = -9999
PROJ_EPSG_WGS84 = 4326

class TairAggregate():
    '''
    A class for aggregating daily data to monthly and annual
    '''
    
    def __init__(self, days):
        '''
        Constructor
        
        @param days: a structured array of date information for the time period of interest
        produced from  util_dates.get_days_metadata_* methods
        '''
    
        uYrs = np.unique(days[YEAR])
        
        self.yr_mths_masks = []
        
        for aYr in uYrs:
            
            for aMth in np.arange(1,13):
                
                self.yr_mths_masks.append(np.nonzero(np.logical_and(days[YEAR]==aYr,days[MONTH]==aMth))[0])
                
        self.days = days
        self.yrMths = utld.get_mth_metadata(uYrs[0],uYrs[-1])
        
        self.yr_masks = []
        
        for aYr in uYrs:
            
            self.yr_masks.append(np.nonzero(self.yrMths[YEAR]==aYr)[0])
        
        self.uYrs = uYrs
    
    
    def daily_to_mthly(self,tair):
        '''
        Aggregates daily data to monthly.
        
        @param tair: a numpy array or masked array. Can be of any shape, 
        but first axis must be the time dimension
        '''
        
        tair_mthly = np.ma.array([np.ma.mean(np.ma.take(tair, aMask, axis=0),axis=0,dtype=np.float) for aMask in self.yr_mths_masks])
        
        return tair_mthly
    
    def daily_to_ann(self,tair):
        '''
        Aggregates daily data to annual.
        
        @param tair: a numpy array or masked array. Can be of any shape, 
        but first axis must be the time dimension
        '''
        
        tair_mthly = self.daily_to_mthly(tair)
        tair_ann = self.mthly_to_ann(tair_mthly)
        
        return tair_ann
    
    def mthly_to_ann(self,tair_mthly):
        '''
        Aggregates daily data to monthly.
        
        @param tair: a numpy array or masked array. Can be of any shape, 
        but first axis must be the time dimension
        '''
        
        tair_ann = np.ma.masked_array([np.ma.mean(np.ma.take(tair_mthly, aMask, axis=0),axis=0,dtype=np.float) for aMask in self.yr_masks])
        
        return tair_ann

class TairTrend():
    '''
    A class for calculating trends
    '''
    
    def __init__(self, days, start_yr=1948, end_yr=2012):
        '''
        Constructor
        
        @param days: a structured array of date information for the time period of interest
        produced from  util_dates.get_days_metadata_* methods
        @param start_yr: the year to start the trend calculation (default: 1948)
        @param end_yr: the year to end the trend calculation (default: 2012)
        '''
        
        self.day_mask = np.nonzero(np.logical_and(days[YEAR] >= start_yr,days[YEAR] <= end_yr))[0]
        self.days = days[self.day_mask]
        self.tair_agg = TairAggregate(self.days)
    
    def get_ann_trend(self,tair):
        '''
        Calculate ann trend from daily data
        :param tair: a 3 dimensional numpy array (time,lat,lon)
        '''
        
        print("Aggregating daily data to annual means...")
        tair_ann = self.tair_agg.daily_to_ann(tair)
        tair_trend = np.zeros((tair_ann.shape[1],tair_ann.shape[2]))
        uYrs = np.unique(self.days[YEAR])
        
        print("Calculating trend for each grid cell...")
        for r in np.arange(tair_ann.shape[1]):
        
            for c in np.arange(tair_ann.shape[2]):
                
                if np.ma.is_masked(tair_ann[0,r,c]):
                    tair_trend[r,c] = MISSING
                else:
                    tair_trend[r,c] = stats.linregress(uYrs,tair_ann[:,r,c])[0]
    
        tair_trend = np.ma.masked_equal(tair_trend, MISSING)
        tair_trend.fill_value = MISSING
        
        return tair_trend

def twx_tile_to_gtiff(nc_ds,a,path_out):
    '''
    Outputs a summary statistic GeoTIFF for a TopoWx tile
    :param nc_ds: a TopoWX NetCDF dataset
    :param a: a 2d or 3d masked numpy array (time,lat,lon) containing the summary statistic
    :param path_out: the output path
    '''
        
    nrow = len(nc_ds.dimensions['lat'])
    ncol = len(nc_ds.dimensions['lon'])
            
    driver = gdal.GetDriverByName("GTiff")
    
    if len(a.shape) == 3:
        raster = driver.Create(path_out,ncol,nrow,a.shape[0],gdalconst.GDT_Float64) 
    else:
        raster = driver.Create(path_out,ncol,nrow,1,gdalconst.GDT_Float64) 
    
    '''
    Create GDAL geotransform list to define resolution and bounds
    GeoTransform[0] /* top left x */
    GeoTransform[1] /* w-e pixel resolution */
    GeoTransform[2] /* rotation, 0 if image is "north up" */
    GeoTransform[3] /* top left y */
    GeoTransform[4] /* rotation, 0 if image is "north up" */
    GeoTransform[5] /* n-s pixel resolution */
    '''
    geotransform = [None]*6
    #n-s pixel height/resolution needs to be negative.
    geotransform[5] = -np.abs(nc_ds.variables['lat'][0] - nc_ds.variables['lat'][1])   
    geotransform[1] = np.abs(nc_ds.variables['lon'][0] - nc_ds.variables['lon'][1])
    geotransform[2],geotransform[4] = (0.0,0.0)
    geotransform[0] = nc_ds.variables['lon'][0] - (geotransform[1]/2.0) 
    geotransform[3] = nc_ds.variables['lat'][0] + np.abs(geotransform[5]/2.0)

    raster.SetGeoTransform(geotransform)

    sr = osr.SpatialReference()
    sr.ImportFromEPSG(PROJ_EPSG_WGS84)
    raster.SetProjection(sr.ExportToWkt())
    
    ndataVal = a.fill_value
    a = np.ma.filled(a)
    
    if len(a.shape) == 3:
        
        for x in np.arange(a.shape[0]):
            
            band = raster.GetRasterBand(int(x+1))
            band.SetNoDataValue(float(ndataVal))
            band.WriteArray(a[x,:,:],0,0) 
            band.FlushCache()
    else:
        
        band = raster.GetRasterBand(1)
        band.SetNoDataValue(float(ndataVal))
        band.WriteArray(a,0,0) 
        band.FlushCache()
    
if __name__ == '__main__':
    
    #Example code for calculating annual trends for a TopoWx tile and outputting them as a GeoTiff
    nc_ds = Dataset('/stage/climate/test_tile_output/h05v02/h05v02_tmin.nc')
    days = utld.get_days_metadata_dates(num2date(nc_ds.variables['time'][:],
                                                 units=nc_ds.variables['time'].units))
    tair_trend = TairTrend(days,1948,2012)
    
    #Doing this in one shot will take ~10GB of memory
    #For less memory usage, process in chunks
    tair = nc_ds.variables['tmin'][:]
    tair_ann = tair_trend.get_ann_trend(tair)
    
    #Output results
    twx_tile_to_gtiff(nc_ds, tair_ann, '/stage/climate/test_tile_output/h05v02_tmin_trend.tiff')
    
