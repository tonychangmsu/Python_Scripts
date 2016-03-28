# -*- coding: utf-8 -*-
"""
Created on Wed Jan 23 14:51:39 2013

@author: tony.chang
"""

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
#srs.ImportFromEPSG(4322) #4322 coordinate ref# for World Geodetic System 1972
#writename = "C:\CHANG\PRISM\PRISM_Analysis\us_" + var + "_" + str(BeginYear) + "_" + str(EndYear) + "_gradients.tif"
writename = "D:\\CHANG\\Python_Scripts\\Output\\test.tif"
outDs = driver.Create(writename, Nx, Ny, 1, gdal.GDT_Float32)
outDs.SetGeoTransform(geotransform)
srs.SetWellKnownGeogCS("WGS72")
outDs.SetProjection(srs.ExportToWkt())
outBand = outDs.GetRasterBand(1)
outBand.SetNoDataValue(-9999)
outBand.WriteArray(gMat, 0, 0)
outDs = None