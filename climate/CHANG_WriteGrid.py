# -*- coding: utf-8 -*-
"""
Created on Wed Sep 12 12:40:26 2012

@author: Tony Chang
"""

def WriteGrid(Pdata,Sdata):
    from osgeo import osr    
    from osgeo import gdal
    Nx = Pdata[0].ncols
    Ny = Pdata[0].nrows
    nbands = 1
    cwidth = Pdata[0].csize
    Yul = Pdata[0].yul
    Xul = Pdata[0].xll
    writename = "C:\CHANG\Temp\Out2.tif"
    fileformat = "GTiff"
    driver = gdal.GetDriverByName(fileformat)
    ds = driver.Create(writename, Ny, Nx, nbands, gdal.GDT_Float32)
    geotransform = [Xul, cwidth, 0.0, Yul, 0.0, -cwidth]      
    ds.SetGeoTransform(geotransform)
    srs = osr.SpatialReference()
    srs.SetWellKnownGeogCS("WGS72")
    ds.SetProjection(srs.ExportToWkt())   
    ds.GetRasterBand(1).WriteArray(Sdata)
    ds = None    
    return ()
    
    #Below is attempt to import to GTiff, needs work, possibly version errors 
'''
from osgeo import osr
from osgeo import gdal, gdal_array
from osgeo.gdalconst import GDT_Float32

Ny, Nx = gMat.shape
cwidth = Pdata[0].csize
Yul = Pdata[0].yul
Xul = Pdata[0].xll

fileformat = "GTiff"
driver = gdal.GetDriverByName(fileformat)
geotransform = [Xul, cwidth,0.0, Yul, 0.0, -cwidth]
srs = osr.SpatialReference()
srs.ImportFromEPSG(4322) #4322 coordinate ref# for World Geodetic System 1972
#writename = "C:\CHANG\PRISM\PRISM_Analysis\us_" + var + "_" + str(BeginYear) + "_" + str(EndYear) + "_gradients.tif"
writename = "C:\CHANG\PRISM\PRISM_Analysis\a.tif"
outDs = driver.Create(writename, Nx, Ny, 1, gdal.GDT_Float32)
outDs.SetGeoTransform(geotransform)
outDs.SetProjection(srs.ExportToWkt())
outBand = outDs.GetRasterBand(1)
outBand.SetNoDataValue(-9999)
outBand.WriteArray(gMat, 0, 0)
#gdal_array.BandWriteArray(outBand,gMat)

'''
