# -*- coding: utf-8 -*-
"""
Created on Tue Jan 22 12:25:03 2013

@author: tony.chang
"""

#SnoTel downloader
import urllib2
import csv

''' import csv snotel station field list '''
numstations =  855 #number of Snotel stations from list
def readData(fileName):
    f = open(fileName, "r")
    data = f.readlines()
    f.close()
    return data

def splitLines(data):
    newLines = []
    for line in data:
            myList = line.split(",")
            newLines.append(myList)
    return(newLines)
    
filepath = 'D:\\chang\\station_data\\snotel\\station_listcsv.csv'

rawdata = readData(filepath)
csvData = splitLines(rawdata)  #data array with all attribute fields

statelist = ['alaska', 'arizona', 'california', 'colorado', 'idaho', 'montana', 'nevada', 
             'new_mexico', 'oregon', 'south_dakota', 'utah', 'washington', 'wyoming']
             
FTPpath = "http://www.wcc.nrcs.usda.gov/ftpref/data/snow/snotel/cards/"
directory = FTPpath + statelist[0] + '/'
f = urllib2.urlopen(directory)
stationfile = open("test.txt", "w")
stationfile.write(f.read())
stationfile.close() 
destPath = 'D:\\chang\station_data\\snotel\\' + statelist + '\\'

