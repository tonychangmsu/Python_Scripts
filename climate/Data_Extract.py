import glob, sys, os, decimal, numpy
from numpy import *

"""
   Initialize

"""

workspace = "C:\\CHANG\\PRISM\\tmax\\Uncompressed\\"

BeginYear = [1895]
EndYear = [2011]
desire_extent = [-112.436, 42.252, -108.263, 46.182]

BeginYear = int(BeginYear[0])       #Converting BeginYear to Integer
EndYear = int(EndYear[0]+ 1)        #Converting EndYear to Integer
yearList = arange(BeginYear, EndYear)
File = []     #Make a list of files that you want to store
returnbil = []    #Array to store read header information
datamatrix = []   #Array to store read data
header_string = ''
newheader = []    #Array to store write header information
tempbil = "t1.bil"
readfromheader = "True"
var = "tmax"
filenum = str(01)

#read orinial ascii PRISM file
originalgrid = workspace + "us_" + var + "_" + str(BeginYear) + ".0" + filenum

def read_grid(originalgrid, desire_extent):
   """
      Read grid function to read and clip ascii PRISM file according to a specified dimension and returns
       a new header and bil

      ARGUMENTS:
      originalgrid - name of the bil
      desire_extent - list containing the desired extent of the new bil in form [maxx, miny, minx, maxy].
                      Note taht western hemisphere longitudes are expressed as negative values.
      RETURNS:
      returnbil - 2 element list
      writtenbil[0] - header as a list
                writtenbil[0][0] - ncols
                writtenbil[0][1] - nrows
                writtenbil[0][2] - xllcorner
                writtenbil[0][3] - yllcorner
                writtenbil[0][4] - cellsize
                writtenbil[0][5] - NODATA_value
                writtenbil[1] - bil matrix as a list
   Written by: Tony Chang
   Adapted from: Frankes, Brent. I&M Division, NRPC, NPS.
   """
   returnbil = []
   datamatrix = []
   header_string = ''
   newheader = []
   tempbil = "t1.bil"
    
   readfile =  open (originalgrid,'r')
   a = readfile.readline()
   temp = a.split()
   ncols = int(temp[1]) #Define number of columns
   a = readfile.readline()
   temp = a.split()
   nrows = int(temp[1])  #Define number of rows
   a = readfile.readline()
   temp = a.split()
   xllcorner = float(temp[1])   #Define xll corner
   a = readfile.readline()
   temp = a.split()
   yllcorner = float(temp[1])   #Define yll corner
   a = readfile.readline()
   temp = a.split()
   cellsize  = float(temp[1])   #Define cellsize
   a = readfile.readline()
   temp = a.split()
   NODATA_value  = temp[1]      #Define NoData value
   readfile.close()

   begin_x = xllcorner
   begin_y = (nrows*cellsize) + yllcorner   

   y_pos = begin_y    #Initialize the first y position
   x_pos = begin_x    #Initialize the first x position

   minx = desire_extent[0]
   miny = desire_extent[1]
   maxx = desire_extent[2]
   maxy = desire_extent[3]

   newxllcorner = begin_x+(ncols*cellsize)     #Initialize the ending x
   newyllcorner = yllcorner     #Initialize the ending y

   readfile =  open (originalgrid,'r')
   nhead = 6                   #First 6 lines of the header to be removed
   for z in range(nhead):      #Strip out header
       a = readfile.readline()

   for z in range(1, nrows+1):    #loop through all rows
       line = readfile.readline()
       datarow = line.split()
       x_pos = begin_x  
       newrow=[]
       for element in datarow:
           if (x_pos <= minx and x_pos >= maxx and
               y_pos >= miny and y_pos <= maxy):
               newrow.append(element) #add element row to newrow array 
               if newxllcorner > x_pos:
                   newxllcorner = x_pos    #if outside bounds, define newxllcorner
               if newyllcorner < y_pos:
                   newyllcorner = y_pos
           x_pos = x_pos + cellsize      # move to next cell
       y_pos = y_pos - cellsize
       if len(newrow) != 0:
          datamatrix[z].append(newrow)    #add row to datamatrix
       
   newcols = len(datamatrix[0])    #define new column length for park extent
   newrows = len(datamatrix)       #define new row length for park extent
   newyllcorner = newyllcorner - len(datamatrix)* cellsize
   newheader = [newcols, newrows, newxllcorner, newyllcorner, cellsize, NODATA,value]
   finalmatrix = []
   for row in datamatrix:
       for element in row:
           finalmatrix.append(element)
   returnbil.append(newheader)
   returnbil.append(finalmatrix)

   return returnbil    #new built array of PRISM data

test = read_grid(originalgrid, desire_extent)
"""
def _write_grid(writefile, headerinfo, datamatrix):
    ncols = headerinfo[0] 
    nrows = headerinfo[1]
    xllcorner = headerinfo[2]
    yllcorner = headerinfo[3]
    cellsize = headerinfo[4]
    NODATA_value = headerinfo[5]

    newheader = 'ncols '+ str(ncols) + '\nnrows '+ str(nrows) + '\nxllcorner ' \
         + str(xllcorner) + '\nyllcorner ' + str(yllcorner) + '\ncellsize ' + \
         str(cellsize) + '\nNODATA_value ' + str(NODATA_value) + '\n'

    try:    
        file2write = open(writefile,"w")
        file2write.write(newheader)
        numberpoints = len(datamatrix)
        for x in range(1,numberpoints+1):
            file2write.write(str(datamatrix[x-1]))
            file2write.write(" ")
            if x % ncols == 0:
                  file2write.write('\n')
        file2write.close()
        returninfo = "True"

   except:
   print "Unable to write to ", file2write
   returninfo = "False"
   return returninfo
"""   
"""
for Year in yearList:
for file in tempFiles:
        allFiles.append(file)
        #Grabing the monthly files of interest
        monthLst = arange[13]
                   
                #Selecting files within monthly date range 
                
                file2unpack = PRISM_SelectDateRange(variable, files, range(Year,Year+1), monthLst)
                #Unpacking all monthly files within yearly date range
                file2Len = len(file2unpack) 
                filesunpacked = PRISM_Unzip(file2unpack)

                tempFiles = filesunpacked[0]
                for file in tempFiles:
                    allFiles.append(file)            
"""
"""
            #print Finding files for water year calculation
            if variable.lower() == "ppt" and averageType.lower() == "wateryear":
                 
                # Selecting all files within the date range (Oct-Dec years of interest for PPT)
                file2unpack = PRISM_SelectDateRange('PPT',files, range(Year-1,Year),[10,11,12])
                file2Len = len(file2unpack) 
                if file2Len != 3:       #Checking to see if the list of files to unpack equals 3 if not continue to next year
                    continue
                # Unpacking all files within the date range (Oct-Dec years of interest)
                filesunpacked = PRISM_Unzip(file2unpack)  

                # Selecting all files within the date range (Jan - Sept years of interest) 
                file2unpack = PRISM_SelectDateRange('PPT',files, range(Year,Year+1),[1,2,3,4,5,6,7,8,9])
                file2Len = len(file2unpack)
                if file2Len != 9:       #Checking to see if the list of files to unpack equals 9 if not continue to next year
                    continue
                # Unpacking all files within the date range (Jan-Sept years of interest)
                filesunpacked_b = PRISM_Unzip(file2unpack)

                # Combining file list for Oct-Dec & Jan-Sept
                filesunpacked = filesunpacked[0]+ filesunpacked_b[0]
                tempFiles = filesunpacked
                for file in tempFiles:
                    allFiles.append(file)

            #Grabing annual average files of interest (.14 files)  
            elif timeStep.lower()=="year":       
                # Selecting all files within the date range (Jan - Dec years of interest) for TMAX/TMIN variable 
                file2unpack = PRISM_SelectDateRange(variable, files, range(Year,Year+1),[14])
                #Unpacking all files within the date range (Jan - Dec years of interest) for tmax/tmin variable 
                file2Len = len(file2unpack) 
                
                filesunpacked = PRISM_Unzip(file2unpack)

                tempFiles = filesunpacked[0]
                for file in tempFiles:
                    allFiles.append(file)   

            #Grabing the monthly files of interest   
            else:       
                strDate = str(percentileDate)
                month = strDate[4:6]
                intMonth = int(month)
                monthLst = [intMonth]
                
                #Selecting files within monthly date range 
                
                file2unpack = PRISM_SelectDateRange(variable, files, range(Year,Year+1), monthLst)
                #Unpacking all monthly files within yearly date range
                file2Len = len(file2unpack) 
                filesunpacked = PRISM_Unzip(file2unpack)

                tempFiles = filesunpacked[0]
                for file in tempFiles:
                    allFiles.append(file)            

        return 1, allFiles, yearList
    except:
        print ("extractPRISM Not Working")
        return 0


def _write_grid(writefile, headerinfo, datamatrix):
    
    ############################################################
    # _write_grid.function
    #
    # ABSTRACT: writes a list and header to a proper bil file
    #   
    # ARGUMENTS: writefile - filename of the bil to create
    #    headerinfo - List of the header information.
    #        headerinfo[0] - ncols
    #        headerinfo[1] - nrows
    #        headerinfo[2] - xllcorner
    #        headerinfo[3] - yllcorner
    #        headerinfo[4] - cellsize
    #        headerinfo[5] - NODATA_value
    #    datamatrix - the list of the dat
    #
    # RETURNS: .bil file as defined by the specified header
    #
    # CREATED BY: Brent Frakes I&M Division, NRPC, NPS.
    ###########################################

    ncols = headerinfo[0] 
    nrows = headerinfo[1]
    xllcorner = headerinfo[2]
    yllcorner = headerinfo[3]
    cellsize = headerinfo[4]
    NODATA_value = headerinfo[5]


    newheader = 'ncols '+ str(ncols) + '\nnrows '+ str(nrows) + '\nxllcorner ' \
         + str(xllcorner) + '\nyllcorner ' + str(yllcorner) + '\ncellsize ' + \
         str(cellsize) + '\nNODATA_value ' + str(NODATA_value) + '\n'
    
    try:    
        file2write = open(writefile,"w")
        file2write.write(newheader)
        numberpoints = len(datamatrix)
        '''
        #NOTE:
        The following for loop has caused some problems.  I have had success
        using either combination at different times and am unsure as to why
        -Brent Frakes 20081106

        '''
        for x in range(1,numberpoints+1):
        #for x in range(0,numberpoints):
              file2write.write(str(datamatrix[x-1]))
              #file2write.write(str(datamatrix[x]))
              file2write.write(" ")
              #if x !=0 and x % ncols == 0:
              if x % ncols == 0:
                  file2write.write('\n')
        file2write.close()
        returninfo = "True"

    except:
        print "Unable to write to ", file2write
        returninfo = "False"
    return returninfo    


def toGrid(FinalFile, workspace):
        
    ###########################################################
    # toGrid.function
    #
    # ABSTRACT: Converts ascii to .img file
    #   
    # ARGUMENTS: 
    #
    # RETURNS: finalarray - Returns a list of percentiles as integers between 0 and 100 
    #
    # CREATED BY: Kirk Sherrill Geospatial Technician, I&M Division, NRPC, NPS.
    ###########################################


    import glob, sys, arcgisscripting, os

    # Create the Geoprocessor object
    gp = arcgisscripting.create(9.3)
    # Check out any necessary licenses
    gp.CheckOutExtension("spatial")
        
    oldName = FinalFile
    txtName = oldName[:-4] + ".txt"
    os.rename(oldName,txtName)
    split = oldName.split("\\")
    splitLen = len(split)
    nameFull = split[splitLen -1]
    name = nameFull[:-4] + ".img"
    outRaster = workspace + name
    # Process: ASCII to Raster...
    gp.SnapRaster = FinalFile
    gp.ASCIIToRaster_conversion(txtName, outRaster, "INTEGER")

    os.remove(txtName)          #removing the .txt Percentile File
    return outRaster
"""
