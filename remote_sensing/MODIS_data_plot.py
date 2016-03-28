#Title: MODIS_data_plot.py
#Author: Tony Chang
#Abstract: Test for opening MODIS data and examining the various bands
#Creation Date: 04/14/2015
#Modified Dates: 01/20/2016, 01/26/2016, 01/28/2016, 01/29/2016, 02/01/2016, 02/02/2016

#local directory : K:\\NASA_data\\scripts

import numpy as np
import matplotlib.pyplot as plt
import os
os.chdir("K:\\NASA_data\\scripts")
import time
import MODIS_acquire as moda
import MODIS_tassel_cap as tas
import MODIS_process as mproc
import MODIS_plot as mplot
import grid_write as gw
import MODIS_masker as mm
import netCDF4 as nc

def normed(value):
	return((value-value.mean())/value.std())

if __name__ == "__main__":

	wd = r'G:\NASA_remote_data\MOD09A1_post_processed'
	mod_file = r'MOD09_GYE_tassel_2000.nc'
	filename = r'%s\%s'%(wd, mod_file)
	ds = nc.Dataset(filename)
	variables = ['brightness', 'greenness', 'wetness']
	scene_no = 9
	b = ds.variables[variables[0]][scene_no]
	g = ds.variables[variables[1]][scene_no]
	w = ds.variables[variables[2]][scene_no]
	date = ds.variables['date'][scene_no]
	outfile = r'D:\CHANG\PhD_Material\Proposals\NASA\NESSF2016\tex_docs\tas_plots.png'
	mplot.plot_tassel_cap(normed(b),normed(g),normed(w), savefile=outfile, date=date)