##############################################################################
# Python library to deal with BIL (Band Interleaved by Line) and BSQ (Band Sequential) files
#
# Author: Ben Taylor
#
# History:
# 12th Feb. 2009: (benj) Created
# 9th Jun. 2009: (benj) Added writeData function
# 20th Aug. 2009: (benj) Added append option to writeDataFile
# 3rd Dec. 2009: (benj) Added findHdrFile function
#
# Available functions:
# readxy: Wrapper function for readBil/readBsq that allows you to omit the number of bands in the file (works it out from the file size)
# readxb: Wrapper function for readBil/readBsq that allows you to omit the number of lines in the file (works it out from the file size)
# readyb: Wrapper function for readBil/readBsq that allows you to omit the number of pixels per line (works it out from the file size)
# readBil: Reads a BIL file and returns a list containing the data from the file
# readBilLine: Reads a line from an open BIL file and returns a list containing the data that was read
# readBsq: Reads a BSQ file and returns a list containing the data from the file
# readBsqBand: Reads a band from an open BSQ file and returns a list containing the data that was read
# writeData: Writes data to an output file straight from an input list
# writeDataFile: Writes a BIL or BSQ file from a 1D list containing data already in the right file order
# write3DData: Writes a BIL or BSQ file from a 3D list with 1st dimension band number, 2nd dimension line number and 3rd dimension pixel number (unfolds to a 1D array then calls writeDataFile)
# writeHdrFile: Writes an ENVI .hdr file to be associated with a BIL or BSQ file
# readHdrFile: Reads data from a given ENVI-style header file
# getEnviType: Gets the ENVI type code equivalent to a particular Python struct format string
# getStructType: Gets the Python struct format string equivalent to a particular ENVI type code
# findHdrFile: Tries to find an associated hdr file for the given filename
#
# You may use or alter this script as you wish, but no warranty of any kind is offered, nor is it guaranteed 
# not to cause security holes in an unsafe environment.
##############################################################################

import os
import os.path
import stat
import struct
import re
import sys

defformat = "h" # Default data format (2-byte signed short int)

# Function readxy
# Wrapper function for readBil/readBsq that allows you to omit the number of bands in the file (works it out from the file size)
# See readBil/readBsq description for arguments and return value
# filetype: "bil" or "bsq" appropriately
def readxy(filename, numlines, pixperline, dataformat=defformat, filetype="bil"):
    fileinfo = os.stat(filename)
    filesize = fileinfo[stat.ST_SIZE]
    
    # Check given format string is valid
    try:
        bytesperpix = struct.calcsize(dataformat)
    except:
        raise ValueError, "Supplied data format " + str(dataformat) + " is invalid"
    # end try
    
    numbands = ((filesize / float(numlines)) / float(pixperline)) / float(bytesperpix)
    
    # Should be an integer, if it's not then one of the given attributes is wrong or the file is corrupt
    if (numbands == int(numbands)):
        if (filetype == "bil"):
            return readBil(filename, int(numlines), int(pixperline), int(numbands), dataformat)
        else:
            if (filetype == "bsq"):
                return readBsq(filename, int(numlines), int(pixperline), int(numbands), dataformat)
            else:
                raise ValueError, "File type argument must be either 'bil' or 'bsq', got: " + filetype
            # end if
        # end if
    else:
        raise ValueError, "File size and supplied attributes do not match"
    # end if
# end function

# Function readxb
# Wrapper function for readBil/readBsq that allows you to omit the number of lines in the file (works it out from the file size)
# See readBil/readBsq description for arguments and return value
# filetype: "bil" or "bsq" appropriately
def readxb(filename, pixperline, numbands, dataformat=defformat, filetype="bil"):
    fileinfo = os.stat(filename)
    filesize = fileinfo[stat.ST_SIZE]
    
    # Check given format string is valid
    try:
        bytesperpix = struct.calcsize(dataformat)
    except:
        raise ValueError, "Supplied data format " + str(dataformat) + " is invalid"
    # end try
    
    numlines = ((filesize / float(numbands)) / float(pixperline)) / float(bytesperpix)
    
    # Should be an integer, if it's not then one of the given attributes is wrong or the file is corrupt
    if (numlines == int(numlines)):
        if (filetype == "bil"):
            return readBil(filename, int(numlines), int(pixperline), int(numbands), dataformat)
        else:
            if (filetype == "bsq"):
                return readBsq(filename, int(numlines), int(pixperline), int(numbands), dataformat)
            else:
                raise ValueError, "File type argument must be either 'bil' or 'bsq', got: " + filetype
            # end if
        # end if
    else:
        raise ValueError, "File size and supplied attributes do not match"
    # end if
# end function

# Function readyb
# Wrapper function for readBil/readBsq that allows you to omit the number of pixels per line (works it out from the file size)
# See readBil/readBsq description for arguments and return value
# filetype: "bil" or "bsq" appropriately
def readyb(filename, numlines, numbands, dataformat=defformat, filetype="bil"):
    fileinfo = os.stat(filename)
    filesize = fileinfo[stat.ST_SIZE]
    
    # Check given format string is valid
    try:
        bytesperpix = struct.calcsize(dataformat)
    except:
        raise ValueError, "Supplied data format " + str(dataformat) + " is invalid"
    # end try
    
    pixperline = ((filesize / float(numbands)) / float(numlines)) / float(bytesperpix)
    
    # Should be an integer, if it's not then one of the given attributes is wrong or the file is corrupt
    if (numlines == int(numlines)):
        if (filetype == "bil"):
            return readBil(filename, int(numlines), int(pixperline), int(numbands), dataformat)
        else:
            if (filetype == "bsq"):
                return readBsq(filename, int(numlines), int(pixperline), int(numbands), dataformat)
            else:
                raise ValueError, "File type argument must be either 'bil' or 'bsq', got: " + filetype
            # end if
        # end if
    else:
        raise ValueError, "File size and supplied attributes do not match"
    # end if
# end function

# Function readBil
# Reads a BIL file and returns a list containing the data from the file
#
# Arguments:
# filename: Name of file to read
# numlines: Number of lines of data in the file
# pixperline: Number of pixels on a line
# numbands: Number of bands in the file
# dataformat: Format string for data, as Python struct definition
#
# Returns: A list containing the data from filename formatted as a list of bands
#   containing a list of lines, each containing a list of pixel values
def readBil(filename, numlines, pixperline, numbands, dataformat=defformat):

    # Check file exists and is a file
    if (not os.path.isfile(filename)):
        raise ValueError, "Supplied filename " + str(filename) + " does not exist"
    # end if
    
    # Check given format string is valid
    try:
        bytesperpix = struct.calcsize(dataformat)
    except:
        raise ValueError, "Supplied data format " + str(dataformat) + " is invalid"
    # end try
    
    # Check file size matches with size attributes
    fileinfo = os.stat(filename)
    filesize = fileinfo[stat.ST_SIZE]
    checknum = (((filesize / float(numbands)) / float(numlines)) / float(bytesperpix)) / pixperline
    if (checknum != 1):
        raise ValueError, "File size and supplied attributes do not match"
    # end if
    
    # Open the file for reading in binary mode
    try:
        bilfile = open(filename, "rb")
    except:
        print "Failed to open BIL file " + filename
        raise
    # end try
    
    # Create a list of bands containing an empty list for each band
    bands = [[] for i in range(0, numbands)] 
    
    # BIL format so have to cycle through lines at top level rather than bands
    for linenum in range(0, numlines):
        for bandnum in range(0, numbands):
            
            if (linenum == 0):
                # For each band create an empty list of lines in the band, but only the first time
                bands[bandnum] = [[] for i in range(0, numlines)] 
            # end if
            
            for pixnum in range(0, pixperline):
            
                # Read one data item (pixel) from the data file. No error checking because we want this to fall over
                # if it fails.
                dataitem = bilfile.read(bytesperpix)
                
                # If we get a blank string then we hit EOF early, raise an error
                if (dataitem == ""):
                    raise EOFError, "Ran out of data to read before we should have"
                # end if
                
                # If everything worked, unpack the binary value and store it in the appropriate pixel value
                bands[bandnum][linenum].append(struct.unpack(dataformat, dataitem)[0])
            # end for
        # end for
    # end for
    bilfile.close()
    
    return bands
# end function

# Function readBilLine
# Reads a line of data from an open BIL file
#
# Arguments:
# bilfile: Open BIL file object
# pixperline: Number of pixels on a line
# numbands: Number of bands in the file
# dataformat: Format string for data, as Python struct definition
#
# Returns: A 2D list with the band number in the first dimension and the pixel number in the second, containing the data values
#   for the line that was read
def readBilLine(bilfile, pixperline, numbands, dataformat=defformat):
    line = []
    
    # Get the size in bytes for the given data format
    itemsize = struct.calcsize(dataformat)
    
    # For each pixel in each band, read a data item, unpack it and store it in the output list
    for bandnum in range(0, numbands):
        line.append([])
        for pixnum in range(0, pixperline):
            dataitem = bilfile.read(itemsize)
            
            if ((dataitem == "") or (len(dataitem) < itemsize)):
                raise EOFError, "Ran out of data to read before we should have"
            # end if
            
            line[bandnum].append(struct.unpack(dataformat, dataitem)[0])
        # end for
    # end for
    
    return line
# end function

# Function readBsq
# Reads a BSQ file and returns a list containing the data from the file
#
# Arguments:
# filename: Name of file to read
# numlines: Number of lines of data in the file
# pixperline: Number of pixels on a line
# numbands: Number of bands in the file
# dataformat: Format string for data, as Python struct definition
#
# Returns: A list containing the data from filename formatted as a list of bands
#   containing a list of lines, each containing a list of pixel values
def readBsq(filename, numlines, pixperline, numbands, dataformat=defformat):

    # Check file exists and is a file
    if (not os.path.isfile(filename)):
        raise ValueError, "Supplied filename " + str(filename) + " does not exist"
    # end if
    
    # Check given format string is valid
    try:
        bytesperpix = struct.calcsize(dataformat)
    except:
        raise ValueError, "Supplied data format " + str(dataformat) + " is invalid"
    # end try
    
    # Check file size matches with size attributes
    fileinfo = os.stat(filename)
    filesize = fileinfo[stat.ST_SIZE]
    checknum = (((filesize / float(numbands)) / float(numlines)) / float(bytesperpix)) / pixperline
    if (checknum != 1):
        raise ValueError, "File size and supplied attributes do not match"
    # end if
    
    # Open the file for reading in binary mode
    try:
        bsqfile = open(filename, "rb")
    except:
        print "Failed to open BSQ file " + filename
        raise
    # end try
    
    # Create a list of bands containing an empty list for each band
    bands = []
    
    # Read data for each band at a time
    for bandnum in range(0, numbands):
        bands.append([])
        
        for linenum in range(0, numlines):
            
            bands[bandnum].append([])
            
            for pixnum in range(0, pixperline):
            
                # Read one data item (pixel) from the data file. No error checking because we want this to fall over
                # if it fails.
                dataitem = bsqfile.read(bytesperpix)
                
                # If we get a blank string then we hit EOF early, raise an error
                if (dataitem == ""):
                    raise EOFError, "Ran out of data to read before we should have"
                # end if
                
                # If everything worked, unpack the binary value and store it in the appropriate pixel value
                bands[bandnum][linenum].append(struct.unpack(dataformat, dataitem)[0])
            # end for
        # end for
    # end for
    
    bsqfile.close()
    
    return bands
# end function

# Function readBsqBand
# Reads a band of data from an open BSQ file
#
# Arguments:
# bsqfile: Open BSQ file object
# pixperline: Number of pixels on a line
# numlines: Number of lines in the file
# dataformat: Format string for data, as Python struct definition
#
# Returns: A 2D list with the band number in the first dimension and the pixel number in the second, containing the data values
#   for the line that was read
def readBsqBand(bsqfile, pixperline, numlines, dataformat=defformat):
    band = []
    
    # Get the size in bytes for the given data format
    itemsize = struct.calcsize(dataformat)
    
    # For each pixel in each band, read a data item, unpack it and store it in the output list
    for linenum in range(0, numlines):
        band.append([])
        for pixnum in range(0, pixperline):
            dataitem = bsqfile.read(itemsize)
            
            if ((dataitem == "") or (len(dataitem) < itemsize)):
                raise EOFError, "Ran out of data to read before we should have"
            # end if
            
            band[linenum].append(struct.unpack(dataformat, dataitem)[0])
        # end for
    # end for
    
    return band
# end function

# Function writeData
# Writes data to an output file straight from an input list
#
# Arguments:
# data: List containing data to be written
# datafile: Open (binary) data file to write to
# dataformat: Format string for data, as Python struct definition
def writeData(data, datafile, dataformat=defformat):
    
    # Check given format string is valid
    try:
        bytesperpix = struct.calcsize(dataformat)
    except:
        raise ValueError, "Supplied data format " + str(dataformat) + " is invalid"
    # end try
    
    # Get the size in bytes for the given data format
    itemsize = struct.calcsize(dataformat)
    
    writestr = ""
    
    for i in range(0, len(data)):
        try:
            packeditem = struct.pack(dataformat, data[i])
        except:
            datafile.close()
            raise IOError, "Could not pack " + str(dataitem) + " into " + str(bytesperpix) + " bytes. Reason: " + str(sys.exc_info()[1])
        # end try
        
        if ((i % 1000 != 999) and (i != len(data) - 1)):
            writestr = writestr + packeditem
        else:
            writestr = writestr + packeditem
            try:
                datafile.write(writestr)
            except:
                datafile.close()
                raise IOError, "Failed to write to data file. Reason: " + str(sys.exc_info()[1])           
            # end try
            writestr = ""
        # end if
        
    # end if
    
    # Write data to file in order
    #for dataitem in data:
    #    try:
    #        packeditem = struct.pack(dataformat, dataitem)
    #    except:
    #        datafile.close()
    #        raise IOError, "Could not pack " + str(dataitem) + " into " + str(bytesperpix) + " bytes. Reason: " + str(sys.exc_info()[1])
    #    # end try
    #    
    #    if (len(writestr) < 1000):
    #        writestr = writestr + packeditem
    #    else:
    #        writestr = packeditem
    #    # end try
    #    
    #    try:
    #        datafile.write(packeditem)
    #    except:
    #        datafile.close()
    #        raise IOError, "Failed to write to data file. Reason: " + str(sys.exc_info()[1])           
    #    # end try
    ## end for
# end function

# Function writeDataFile
# Writes a data file (BIL or BSQ) from a 1D list containing data already in the right file order
# (ie data are written to the file in the order that they're in the list)
#
# Arguments:
# data: List containing data to write to the file
# filename: Name of file to be written to
# dataformat: Format string for data, as Python struct definition
# append: If True then appends the data to the end of the file rather than writing a new blank file. Default False
def writeDataFile(data, filename, dataformat=defformat, append=False):

    # Get correct format string to open the file with
    if (append):
        writeformat = "ab"
    else:
        writeformat = "wb"
    # end if
    
    # Open the data file for writing in binary mode
    try:
        datafile = open(filename, writeformat)
    except:
        print "Could not open data file " + str(filename) + " for writing"
        raise
    # end try
    
    # Check given format string is valid
    try:
        bytesperpix = struct.calcsize(dataformat)
    except:
        raise ValueError, "Supplied data format " + str(dataformat) + " is invalid"
    # end try
        
    # Pack each data item into binary data and write it to the output file
    for dataitem in data:
        try:
            packeditem = struct.pack(dataformat, dataitem)
        except:
            datafile.close()
            print "Could not pack " + str(dataitem) + " into " + str(bytesperpix) + " bytes"
            raise
        # end try
        
        try:
            datafile.write(packeditem)
        except:
            print "Failed to write to data file. Reason: " + str(sys.exc_info()[1])
            datafile.close()
            raise
        # end try
    # end for
    
    datafile.close()
# end function

# Function write3DData
# Writes a BIL or BSQ file from a 3D list with 1st dimension band number, 2nd dimension line number
# and 3rd dimension pixel number (unfolds to a 1D array then calls writeDataFile)
#
# Arguments:
# data: List containing data to write to the file
# filename: Name of file to be written to
# writehdr: Flag denoting whether to write an ENVI header file (default true)
# dataformat: Format string for data, as Python struct definition
# interleave: "bil" or "bsq" appropriately
def write3DData(data, filename, writehdr=True, dataformat=defformat, interleave="bil"):

    # Store numbers of bands, lines and pixels per line for convenience
    numbands = len(data)
    numlines = len(data[0])
    pixperline = len(data[0][0])
    
    # Check given format string is valid
    try:
        bytesperpix = struct.calcsize(dataformat)
    except:
        raise ValueError, "Supplied data format " + str(dataformat) + " is invalid"
    # end try
    
    # Create list for unfolding to
    outdata = [0.0 for i in range(0, numbands * numlines * pixperline)] 
    
    # Run through the data array and put all the data in the right place in the unfolded list
    for bandnum in range(0, numbands):
        for linenum in range(0, numlines):
            for pixnum in range(0, pixperline):
                # Work out appropriate index within BIL file format for next pixel and store in 1D data array
                if (interleave == "bil"):
                    pixindex = (pixperline * numbands * linenum) + (pixperline * bandnum) + pixnum 
                else:
                    if (interleave == "bsq"):
                        pixindex = (pixperline * numbands * linenum) + (pixperline * linenum) + pixnum
                    else:
                        raise ValueError, "Interleave argument to write3DData must be either 'bil' or 'bsq', got: " + interleave
                    # end if
                # end if
                outdata[pixindex] = data[bandnum][linenum][pixnum]
            # end for
        # end for
    # end for
    
    # Write output file (or throw an error if there's no data to write)
    if (len(outdata) > 0):
        writeDataFile(outdata, filename, dataformat)
    else:
        raise ValueError, "One or more dimensions of the data array were 0"
    # end if
    
    # Write header file if requested
    if (writehdr):
        try:
            # Check ENVI data type
            datatype = getEnviType(dataformat)
        except:
            datafile.close()
            print "Unable to generate header for type " + dataformat + ", data type is not valid for ENVI"
        # end try
        
        writeHdrFile(filename + ".hdr", pixperline, numlines, numbands, datatype, interleave)
    # end if
# end function

# Function writeHdrFile
# Writes an ENVI .hdr file to be associated with a data file
#
# Arguments:
# filename: Name of .hdr file to be written
# samples: Number of pixels per line (samples)
# lines: Number of lines
# bands: Number of bands
# datatype: Numeric code for relevant data type
def writeHdrFile(filename, samples, lines, bands, datatype, interleave="bil"):
    try:
        hdrfile = open(filename, "w")
    except:
        print "Could not open header file " + str(filename) + " for writing"
        raise
    # end try
    
    hdrfile.write("ENVI\n")
    hdrfile.write("description = { Created by bil_handler.py }\n")
    hdrfile.write("samples = " + str(samples) + "\n")
    hdrfile.write("lines   = " + str(lines) + "\n")
    hdrfile.write("bands   = " + str(bands) + "\n")
    hdrfile.write("header offset = 0\n")
    hdrfile.write("file type = ENVI Standard\n")
    hdrfile.write("data type = " + str(datatype) + "\n")
    hdrfile.write("interleave = " + interleave + "\n")
    hdrfile.write("byte order = 0\n")
    
    hdrfile.flush()
    hdrfile.close()
# end function

# Function readHdrFile
# Reads data from a given ENVI-style header file
#
# Arguments
# hdrfilename: Name of header file to be read
#
# Returns: Dictionary containing keys/values from header file
def readHdrFile(hdrfilename):
    output = {}
    inblock = False
    
    try:
        hdrfile = open(hdrfilename, "r")
    except:
        print "Could not open hdr file '" + str(hdrfilename) + "'"
        raise
    # end try
    
    # Read line, split it on equals, strip whitespace from resulting strings and add key/value pair to output
    currentline = hdrfile.readline()
    while (currentline != ""):
        # ENVI headers accept blocks bracketed by curly braces - check for these
        if (not inblock):
            # Split line on first equals sign
            if (re.search("=", currentline) != None):
                linesplit = re.split("=", currentline, 1)
                key = linesplit[0].strip()
                value = linesplit[1].strip()
                
                # If value starts with an open brace, it's the start of a block - strip the brace off and read the rest of the block
                if (re.match("{", value) != None):
                    inblock = True
                    value = re.sub("^{", "", value, 1)
                    
                    # If value ends with a close brace it's the end of the block as well - strip the brace off
                    if (re.search("}$", value)):
                        inblock = False
                        value = re.sub("}$", "", value, 1)
                    # end if
                # end if
                value = value.strip()
                output[key] = value
            # end if
        else:
            # If we're in a block, just read the line, strip whitespace (and any closing brace ending the block) and add the whole thing
            value = currentline.strip()
            if (re.search("}$", value)):
                inblock = False
                value = re.sub("}$", "", value, 1)
                value = value.strip()
            # end if
            output[key] = output[key] + value
        # end if
        
        currentline = hdrfile.readline()
    # end while
    
    hdrfile.close()
    
    return output
# end function

# Function getEnviType
# Gets the ENVI type code equivalent to a particular Python struct format string
#
# Arguments
# formatstr: Struct format string to get ENVI type code for
#
# Returns: ENVI numeric type code for supplied format string
def getEnviType(formatstr):
    
    dtype = -1
    
    # Check the given format string is valid
    try:
        struct.calcsize(formatstr)
    except:
        raise ValueError, formatstr + " is not a valid format string"
    # end try
    
    # Do the conversion
    if (formatstr == "b"):
        dtype = 1 # Signed (?) byte
    elif (formatstr == "h"):
        dtype = 2 # 2-byte signed short int (ENVI calls it an int)
    elif (formatstr == "H"):
        dtype = 12 # 2-byte unsigned int (ENVI calls it an int)
    elif (formatstr == "i"):
        dtype = 3 # 4-byte signed int (ENVI calls it a Long)
    elif (formatstr == "I"):
        dtype = 13 # 4-byte unsigned int (ENVI calls it a Long)
    elif (formatstr == "f"):
        dtype = 4 # 4-byte float
    elif (formatstr == "d"):
        dtype = 5 # 8-byte double precision
    elif (formatstr == "l"):
        dtype = 14 # 8-byte long int (ENVI 64-bit int)
    elif (formatstr == "L"):
        dtype = 15 # 8-byte unsigned long int (ENVI 64-bit int)
    else:
        # If we get here then the format string is valid for Python but not for ENVI, raise an error
        raise ValueError, formatstr + " is a valid Python format string but does not have an ENVI equivalent"
    # end if
    
    return dtype
# end function

# Function getStructType
# Gets the Python struct format string equivalent to a particular ENVI type code
#
# Arguments
# typecode: ENVI type code to get Python format string for
#
# Returns: Single-character Python struct format string
def getStructType(typecode):
    
    try:
        inttype = int(typecode)
    except:
        raise ValueError, str(typecode) + " is not a valid ENVI type for conversion"
    # end try
    
    # Do the conversion
    if (inttype == 1):
        formatstr = "b" # Signed (?) byte
    elif (inttype == 2):
        formatstr = "h" # 2-byte signed short int (ENVI calls it an int)
    elif (inttype == 12):
        formatstr = "H" # 2-byte unsigned int (ENVI calls it an int)
    elif (inttype == 3):
        formatstr = "i" # 4-byte signed int (ENVI calls it a Long)
    elif (inttype == 13):
        formatstr = "I" # 4-byte unsigned int (ENVI calls it a Long)
    elif (inttype == 4):
        formatstr = "f" # 4-byte float
    elif (inttype == 5):
        formatstr = "d" # 8-byte double precision
    elif (inttype == 14):
        formatstr = "l" # 8-byte long int (ENVI 64-bit int)
    elif (inttype == 15):
        formatstr = "L" # 8-byte unsigned long int (ENVI 64-bit int)
    else:
        # If we get here then the type code doesn't have a Python equivalent, raise an error
        raise ValueError, str(typecode) + " does not have an equivalent Python format string"
    # end if
    
    return formatstr
# end function

# Function findHdrFile
# Tries to find an associated hdr file for the given filename
#
# Arguments
# rawfilename: File name to try and match to a header file
#
# Returns: The name of the header file found, or None if no header was found
#
# Known issues: Only works by filename matching and file extension, will break if the files in the directory
# are named such that this doesn't work - will either not find a header that exists (ie if the header file isn't 
# just the raw file name with ".hdr" either on the end or replacing the existing suffix) or potentially may return the
# wrong header file if for some reason the header for a different file is named to match the file given.
def findHdrFile(rawfilename):
    # Check the file exists  
    if (not os.path.isfile(rawfilename)):
        raise IOError, "Could not find file " + rawfilename
    # end if
    
    # Get the filename without path or extension
    filename = os.path.basename(rawfilename)
    filesplit = os.path.splitext(filename)
    filebase = filesplit[0]
    dirname = os.path.dirname(rawfilename)
    
    # See if we can find the header file to use
    if (os.path.isfile(os.path.join(dirname, filebase + ".hdr"))):
        hdrfile = os.path.join(dirname, filebase + ".hdr")
    elif (os.path.isfile(os.path.join(dirname, filename + ".hdr"))):
        hdrfile = os.path.join(dirname, filename + ".hdr")
    else:
        hdrfile = None
    # end if
    
    return hdrfile
# end function
