# -*- coding: utf-8 -*-
"""
Created on Wed Sep 12 12:40:26 2012

@author: Tony Chang
"""

def WriteGrid(Pdata,Sdata,monthnum):
    from osgeo import osr    
    from osgeo import gdal
    Nx = Pdata[0].ncols
    Ny = Pdata[0].nrows
    nbands = 1
    cwidth = Pdata[0].csize
    Yul = Pdata[0].yul
    Xul = Pdata[0].xll #same left edge value
    sourcename = "C:\\CHANG\\Temp\\"
    monthlabel = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec","All"]
    writename = sourcename + monthlabel[monthnum-1] + "TminGrid.tif"
    fileformat = "GTiff"
    driver = gdal.GetDriverByName(fileformat)
    ds = driver.Create(writename, Nx, Ny, nbands, gdal.GDT_Float32)
# top left x, w-e pixel resolution, rotation, top left y, rotation, n-s pixel resolution
    geotransform = [Xul, cwidth, 0.0, Yul, 0.0, -cwidth]      
    ds.SetGeoTransform(geotransform)
    srs = osr.SpatialReference()
    srs.SetWellKnownGeogCS("WGS72")
    ds.SetProjection(srs.ExportToWkt())   
    ds.GetRasterBand(1).WriteArray(Sdata)
    ds = None    
    return ()

#---------Write ESRI raster file
def ESRIWriteGrid(gMat,Pdata, filename):
    import arcpy
    gncols = gMat.shape[1]
    gnrows = gMat.shape[0]
    gxllcorner = Pdata[0].xll
    gyllcorner = Pdata[0].yul - (gnrows*Pdata[0].csize)
    corner = arcpy.Point(gxllcorner, gyllcorner)
    gcellsize = Pdata[0].csize
    gNODATA = int(Pdata[0].NODATA)

    myRaster = arcpy.NumPyArrayToRaster(gMat, corner, gcellsize, gcellsize, gNODATA)
    projection = "GEOGCS['GCS_WGS_1972',DATUM['D_WGS_1972',SPHEROID['WGS_1972',6378135.0,298.26]],PRIMEM['Greenwich',0.0],UNIT['Degree',0.0174532925199433]]"
    arcpy.DefineProjection_management(myRaster,projection)
    myRaster.save("C:/CHANG/PRISM/PRISM_Analysis/" + filename)  #change file name here
    return ()
  
