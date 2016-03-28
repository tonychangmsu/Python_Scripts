#Title: MODIS_masker.py
#Author: Tony Chang
#Abstract: Functions that examine the MODIS QA/QC band and generates masks to remove bad pixels
#Creation Date: 02/02/2016
#Modified Dates: 

#local directory : K:\\NASA_data\\scripts

import numpy as np
import matplotlib.pyplot as plt
import os
os.chdir("K:\\NASA_data\\scripts")
import time
import MODIS_acquire as moda
import MODIS_process as mproc

def get_qa(mod_filename):
	ds = moda.mod_acquire_by_band(mod_filename, 'qc')
	ds_array = ds.GetRasterBand(1).ReadAsArray()
	return(ds_array)

def get_mask(qc_array, bitpos=0, bitlen=2, value = '00'):
	'''input in a qc array and request a bitposition and bitlen to search for 
	and the mask will return those values for the qc_array.
	Generates mask with given bit information.
	Parameters
		bitpos		-	Position of the specific QA bits in the value string.
		bitlen		-	Length of the specific QA bits.
		value  		-	A value indicating the desired condition.
	Mask returns 1 if there is a match and 0 if false
	'''
	lenstr = ''
	for i in range(bitlen):
		lenstr += '1' #fills with 1's for the length of the QA bits to perform a comparison
	bitlen = int(lenstr, 2) #changes bitlen into integer representation of lenstr
	if type(value) == str:
		value = int(value, 2) #converts value to the integer representation of value if it is a string
	posValue = bitlen << bitpos #shift the bit string however many spaces over left to get a postive value tester with binary
	conValue = value << bitpos #does the same to the value that we desire
	mask = (qc_array & posValue) == conValue # the & is the bitwise and of qc_array and posValue
	return(mask.astype(int))
