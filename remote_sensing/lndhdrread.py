import os

def get_metadata(filename):
	mtl_name =  [f for f in os.listdir(filename) if f.endswith('MTL.txt')] #find metadata file
	fname = r'%s\%s' %(filename, mtl_name[0])
	lines = iter(open(fname).readlines())
	group = []
	value = []
	for i in lines:
		line = i.strip().replace('"','').split('=') #cut and split each line
		if line[0] == 'END':
			break
		elif (line[0] =='BEGIN_GROUP ') or (line[0] =='END_GROUP ') or (line[0] =='GROUP '):
			continue
		else:
			group.append(line[0][:-1])
			value.append(line[1])
	metadata = dict(zip(group, value))
	return(metadata)

def lndhdrread(filename):
	'''
	# Revisions:
	# Read in the metadata for Landsat 8 (Zhe 04/04/2013)
	# Translated to python 3 (Chang 02/23/2016)
	# Read in the old or new metadata for Landsat TM/ETM+ images (Zhe 10/18/2012)
	# [Lmax,Lmin,Qcalmax,Qcalmin,Refmax,Refmin,ijdim_ref,ijdim_thm,reso_ref,...
	#    reso_thm,ul,zen,azi,zc,Lnum,doy]=lndhdrread(filename)
	# Where:
	# Inputs:
	# filename='L*MTL.txt'
	# Outputs:
	# 1) Lmax = Max radiances
	# 2) Lmin = Min radiances
	# 3) Qcalmax = Max calibrated DNs
	# 4) Qcalmin = Min calibrated DNs
	# 5) ijdim_ref = [nrows,ncols] # dimension of optical bands
	# 6) ijdim_ref = [nrows,ncols] # dimension of thermal band
	# 7) reo_ref = 28/30 # resolution of optical bands
	# 8) reo_thm = 60/120 # resolution of thermal band
	# 9) ul = [upperleft_mapx upperleft_mapy]
	# 10) zen = solar zenith angle (degrees)
	# 11) azi = solar azimuth angle (degrees)
	# 12) zc = Zone Number
	# 13) Lnum = 4,5,or 7 Landsat sensor number
	# 14) doy = day of year (1,2,3,...,356)
	#
	##
	# open and read hdr file
	'''
	md = get_metadata(filename)
	#fid_in = open(filename,'r')
	#geo_char = fscanf(fid_in,'#c',inf)
	#fclose(fid_in)
	#geo_char=geo_char'
	#geo_str=strread(geo_char,'#s')

	## initialize Refmax & Refmin
	Refmax = -1
	Refmin = -1

	## Identify Landsat Number (Lnum = 4, 5, 7, or 8)
	LID = md['SPACECRAFT_ID'].strip()
	Lnum = int(LID[-1])

	if (Lnum >= 4 and Lnum <=7):
		# determine the metadata type "old" or "new"
		typecheck = md['LANDSAT_SCENE_ID'].strip()
		if typecheck: #if this is empty it is the old type of LANDSAT metadata
		# Read in LMAX
			Lmax_B1 = float(md['RADIANCE_MAXIMUM_BAND_1'].strip())
			Lmax_B2 = float(md['RADIANCE_MAXIMUM_BAND_2'].strip())
			Lmax_B3 = float(md['RADIANCE_MAXIMUM_BAND_3'].strip())
			Lmax_B4 = float(md['RADIANCE_MAXIMUM_BAND_4'].strip())
			Lmax_B5 = float(md['RADIANCE_MAXIMUM_BAND_5'].strip())
			if Lnum == 7:
				Lmax_B6 = float(md['RADIANCE_MAXIMUM_BAND_6_VCID_1'].strip())
			else:
				Lmax_B6 = float(md['RADIANCE_MAXIMUM_BAND_6'].strip())
			Lmax_B7 = float(md['RADIANCE_MAXIMUM_BAND_7'].strip())
			Lmax = [Lmax_B1,Lmax_B2,Lmax_B3,Lmax_B4,Lmax_B5,Lmax_B6,Lmax_B7]

			# Read in LMIN
			Lmin_B1 = float(md['RADIANCE_MINIMUM_BAND_1'].strip())
			Lmin_B2 = float(md['RADIANCE_MINIMUM_BAND_2'].strip())
			Lmin_B3 = float(md['RADIANCE_MINIMUM_BAND_3'].strip())
			Lmin_B4 = float(md['RADIANCE_MINIMUM_BAND_4'].strip())
			Lmin_B5 = float(md['RADIANCE_MINIMUM_BAND_5'].strip())
			if Lnum == 7:
				Lmin_B6 = float(md['RADIANCE_MINIMUM_BAND_6_VCID_1'].strip())
			else:
				Lmin_B6 = float(md['RADIANCE_MINIMUM_BAND_6'].strip())
			Lmin_B7 = float(md['RADIANCE_MINIMUM_BAND_7'].strip())
			Lmin = [Lmin_B1,Lmin_B2,Lmin_B3,Lmin_B4,Lmin_B5,Lmin_B6,Lmin_B7]
			
			# Read in QCALMAX
			Qcalmax_B1 = float(md['QUANTIZE_CAL_MAX_BAND_1'].strip())
			Qcalmax_B2 = float(md['QUANTIZE_CAL_MAX_BAND_2'].strip())
			Qcalmax_B3 = float(md['QUANTIZE_CAL_MAX_BAND_3'].strip())
			Qcalmax_B4 = float(md['QUANTIZE_CAL_MAX_BAND_4'].strip())
			Qcalmax_B5 = float(md['QUANTIZE_CAL_MAX_BAND_5'].strip())
			if Lnum == 7:
				Qcalmax_B6 = float(md['QUANTIZE_CAL_MAX_BAND_6_VCID_1'].strip())
			else:
				Qcalmax_B6 = float(md['QUANTIZE_CAL_MAX_BAND_6'].strip())
			Qcalmax_B7 = float(md['QUANTIZE_CAL_MAX_BAND_7'].strip())
			Qcalmax = [Qcalmax_B1,Qcalmax_B2,Qcalmax_B3,Qcalmax_B4,Qcalmax_B5,Qcalmax_B6,Qcalmax_B7]
			
			# Read in QCALMIN
			Qcalmin_B1 = float(md['QUANTIZE_CAL_MIN_BAND_1'].strip())
			Qcalmin_B2 = float(md['QUANTIZE_CAL_MIN_BAND_2'].strip())
			Qcalmin_B3 = float(md['QUANTIZE_CAL_MIN_BAND_3'].strip())
			Qcalmin_B4 = float(md['QUANTIZE_CAL_MIN_BAND_4'].strip())
			Qcalmin_B5 = float(md['QUANTIZE_CAL_MIN_BAND_5'].strip())
			if Lnum == 7:
				Qcalmin_B6 = float(md['QUANTIZE_CAL_MIN_BAND_6_VCID_1'].strip())
			else:
				Qcalmin_B6 = float(md['QUANTIZE_CAL_MIN_BAND_6'].strip())
			Qcalmin_B7 = float(md['QUANTIZE_CAL_MIN_BAND_7'].strip())
			Qcalmin = [Qcalmin_B1,Qcalmin_B2,Qcalmin_B3,Qcalmin_B4,Qcalmin_B5,Qcalmin_B6,Qcalmin_B7]
			
			# Read in nrows & ncols of optical bands
			Sample_ref = int(md['REFLECTIVE_SAMPLES'].strip())
			Line_ref = int(md['REFLECTIVE_LINES'].strip())
			# record ijdimension of optical bands
			ijdim_ref = [Line_ref,Sample_ref]
			
			Sample_thm = float(md['THERMAL_SAMPLES'].strip())
			Line_thm = float(md['THERMAL_LINES'].strip())
			# record thermal band dimensions (i,j)
			ijdim_thm = [Line_thm,Sample_thm]
			
			# Read in resolution of optical and thermal bands
			reso_ref = float(md['GRID_CELL_SIZE_REFLECTIVE'].strip())
			reso_thm = float(md['GRID_CELL_SIZE_THERMAL'].strip())
			
			# Read in UTM Zone Number
			zc = float(md['UTM_ZONE'].strip())
			# Read in Solar Azimuth & Elevation angle (degrees)
			azi = float(md['SUN_AZIMUTH'].strip())
			zen = 90 - float(md['SUN_ELEVATION'].strip())
			# Read in upperleft mapx,y
			ulx = float(md['CORNER_UL_PROJECTION_X_PRODUCT'].strip())
			uly = float(md['CORNER_UL_PROJECTION_Y_PRODUCT'].strip())
			ul = [ulx,uly]
			# Read in date of year
			char_doy = md['LANDSAT_SCENE_ID'].strip()
			doy = int(char_doy[13:16])

		else: # "old" MTL.txt    
			# read in LMAX
			Lmax_B1 = float(md['LMAX_BAND1'].strip())
			Lmax_B2 = float(md['LMAX_BAND2'].strip())
			Lmax_B3 = float(md['LMAX_BAND3'].strip())
			Lmax_B4 = float(md['LMAX_BAND4'].strip())
			Lmax_B5 = float(md['LMAX_BAND5'].strip())
			if Lnum == 7:
				Lmax_B6 = float(md['LMAX_BAND61'].strip())
			else:
				Lmax_B6 = float(md['LMAX_BAND6'].strip())
			
			Lmax_B7 = float(md['LMAX_BAND7'].strip())
			Lmax = [Lmax_B1,Lmax_B2,Lmax_B3,Lmax_B4,Lmax_B5,Lmax_B6,Lmax_B7]
			
			# Read in LMIN
			Lmin_B1 = float(md['LMIN_BAND1'].strip())
			Lmin_B2 = float(md['LMIN_BAND2'].strip())
			Lmin_B3 = float(md['LMIN_BAND3'].strip())
			Lmin_B4 = float(md['LMIN_BAND4'].strip())
			Lmin_B5 = float(md['LMIN_BAND5'].strip())
			if Lnum == 7:
				Lmin_B6 = float(md['LMIN_BAND61'].strip())
			else:
				Lmin_B6 = float(md['LMIN_BAND6'].strip())
			
			Lmin_B7 = float(md['LMIN_BAND7'].strip())
			Lmin = [Lmin_B1,Lmin_B2,Lmin_B3,Lmin_B4,Lmin_B5,Lmin_B6,Lmin_B7]
			
			# Read in QCALMAX
			Qcalmax_B1 = float(md['QCALMAX_BAND1'].strip())
			Qcalmax_B2 = float(md['QCALMAX_BAND2'].strip())
			Qcalmax_B3 = float(md['QCALMAX_BAND3'].strip())
			Qcalmax_B4 = float(md['QCALMAX_BAND4'].strip())
			Qcalmax_B5 = float(md['QCALMAX_BAND5'].strip())
			if Lnum == 7:
				Qcalmax_B6 = float(md['QCALMAX_BAND61'].strip())
			else:
				Qcalmax_B6 = float(md['QCALMAX_BAND6'].strip())
			
			Qcalmax_B7 = float(md['QCALMAX_BAND7'].strip())
			Qcalmax = [Qcalmax_B1,Qcalmax_B2,Qcalmax_B3,Qcalmax_B4,Qcalmax_B5,Qcalmax_B6,Qcalmax_B7]
			
			# Read in QCALMIN
			Qcalmin_B1 = float(md['QCALMIN_BAND1'].strip())
			Qcalmin_B2 = float(md['QCALMIN_BAND2'].strip())
			Qcalmin_B3 = float(md['QCALMIN_BAND3'].strip())
			Qcalmin_B4 = float(md['QCALMIN_BAND4'].strip())
			Qcalmin_B5 = float(md['QCALMIN_BAND5'].strip())
			if Lnum == 7:
				Qcalmin_B6 = float(md['QCALMIN_BAND61'].strip())
			else:
				Qcalmin_B6 = float(md['QCALMIN_BAND6'].strip())
			
			Qcalmin_B7 = float(md['QCALMIN_BAND7'].strip())
			Qcalmin = [Qcalmin_B1,Qcalmin_B2,Qcalmin_B3,Qcalmin_B4,Qcalmin_B5,Qcalmin_B6,Qcalmin_B7]
			
			# Read in nrows & ncols of optical bands
			Sample_ref = int(md['PRODUCT_SAMPLES_REF'].strip())
			Line_ref = int(md['PRODUCT_LINES_REF'].strip())
			# record ijdimension of optical bands
			ijdim_ref = [Line_ref,Sample_ref]
			
			Sample_thm = float(md['PRODUCT_SAMPLES_THM'].strip())
			Line_thm = float(md['PRODUCT_LINES_THM'].strip())
			# record thermal band dimensions (i,j)
			ijdim_thm = [Line_thm,Sample_thm]
			
			# Read in resolution of optical and thermal bands
			reso_ref = float(md['GRID_CELL_SIZE_REF'].strip())
			reso_thm = float(md['GRID_CELL_SIZE_THM'].strip())
			
			# Read in UTM Zone Number
			zc = float(md['ZONE_NUMBER'].strip())
			# Read in Solar Azimuth & Elevation angle (degrees)
			azi = float(md['SUN_AZIMUTH'].strip())
			zen = 90-float(md['SUN_ELEVATION'].strip())
			# Read in upperleft mapx,y
			ulx = float(md['PRODUCT_UL_CORNER_MAPX'].strip())
			uly = float(md['PRODUCT_UL_CORNER_MAPY'].strip())
			ul = [ulx,uly]
			# Read in date of year
			char_doy = md['DATEHOUR_CONTACT_PERIOD'].strip()
			doy = int(char_doy[2:5])
	elif Lnum == 8:
		# read in LMAX
		Lmax_B2 = float(md['RADIANCE_MAXIMUM_BAND_2'].strip())
		Lmax_B3 = float(md['RADIANCE_MAXIMUM_BAND_3'].strip())
		Lmax_B4 = float(md['RADIANCE_MAXIMUM_BAND_4'].strip())
		Lmax_B5 = float(md['RADIANCE_MAXIMUM_BAND_5'].strip())
		Lmax_B6 = float(md['RADIANCE_MAXIMUM_BAND_6'].strip())
		Lmax_B7 = float(md['RADIANCE_MAXIMUM_BAND_7'].strip())
		Lmax_B9 = float(md['RADIANCE_MAXIMUM_BAND_9'].strip())
		Lmax_B10 = float(md['RADIANCE_MAXIMUM_BAND_10'].strip())
		
		Lmax = [Lmax_B2,Lmax_B3,Lmax_B4,Lmax_B5,Lmax_B6,Lmax_B7,Lmax_B9,Lmax_B10]
			
		# read in LMIN
		Lmin_B2 = float(md['RADIANCE_MINIMUM_BAND_2'].strip())
		Lmin_B3 = float(md['RADIANCE_MINIMUM_BAND_3'].strip())
		Lmin_B4 = float(md['RADIANCE_MINIMUM_BAND_4'].strip())
		Lmin_B5 = float(md['RADIANCE_MINIMUM_BAND_5'].strip())
		Lmin_B6 = float(md['RADIANCE_MINIMUM_BAND_6'].strip())
		Lmin_B7 = float(md['RADIANCE_MINIMUM_BAND_7'].strip())
		Lmin_B9 = float(md['RADIANCE_MINIMUM_BAND_9'].strip())
		Lmin_B10 = float(md['RADIANCE_MINIMUM_BAND_10'].strip())
		
		Lmin = [Lmin_B2,Lmin_B3,Lmin_B4,Lmin_B5,Lmin_B6,Lmin_B7,Lmin_B9,Lmin_B10]
		
		# Read in QCALMAX
		Qcalmax_B2 = float(md['QUANTIZE_CAL_MAX_BAND_2'].strip())
		Qcalmax_B3 = float(md['QUANTIZE_CAL_MAX_BAND_3'].strip())
		Qcalmax_B4 = float(md['QUANTIZE_CAL_MAX_BAND_4'].strip())
		Qcalmax_B5 = float(md['QUANTIZE_CAL_MAX_BAND_5'].strip())
		Qcalmax_B6 = float(md['QUANTIZE_CAL_MAX_BAND_6'].strip())
		Qcalmax_B7 = float(md['QUANTIZE_CAL_MAX_BAND_7'].strip())
		Qcalmax_B9 = float(md['QUANTIZE_CAL_MAX_BAND_9'].strip())
		Qcalmax_B10 = float(md['QUANTIZE_CAL_MAX_BAND_10'].strip())
		Qcalmax = [Qcalmax_B2,Qcalmax_B3,Qcalmax_B4,Qcalmax_B5,Qcalmax_B6,Qcalmax_B7,Qcalmax_B9,Qcalmax_B10]
		
		# Read in QCALMIN
		Qcalmin_B2 = float(md['QUANTIZE_CAL_MIN_BAND_2'].strip())
		Qcalmin_B3 = float(md['QUANTIZE_CAL_MIN_BAND_3'].strip())
		Qcalmin_B4 = float(md['QUANTIZE_CAL_MIN_BAND_4'].strip())
		Qcalmin_B5 = float(md['QUANTIZE_CAL_MIN_BAND_5'].strip())
		Qcalmin_B6 = float(md['QUANTIZE_CAL_MIN_BAND_6'].strip())
		Qcalmin_B7 = float(md['QUANTIZE_CAL_MIN_BAND_7'].strip())
		Qcalmin_B9 = float(md['QUANTIZE_CAL_MIN_BAND_9'].strip())
		Qcalmin_B10 = float(md['QUANTIZE_CAL_MIN_BAND_10'].strip())
		Qcalmin = [Qcalmin_B2,Qcalmin_B3,Qcalmin_B4,Qcalmin_B5,Qcalmin_B6,Qcalmin_B7,Qcalmin_B9,Qcalmin_B10]
		
		# read in Refmax
		Refmax_B2 = float(md['REFLECTANCE_MAXIMUM_BAND_2'].strip())
		Refmax_B3 = float(md['REFLECTANCE_MAXIMUM_BAND_3'].strip())
		Refmax_B4 = float(md['REFLECTANCE_MAXIMUM_BAND_4'].strip())
		Refmax_B5 = float(md['REFLECTANCE_MAXIMUM_BAND_5'].strip())
		Refmax_B6 = float(md['REFLECTANCE_MAXIMUM_BAND_6'].strip())
		Refmax_B7 = float(md['REFLECTANCE_MAXIMUM_BAND_7'].strip())
		Refmax_B9 = float(md['REFLECTANCE_MAXIMUM_BAND_9'].strip())
		
		Refmax = [Refmax_B2,Refmax_B3,Refmax_B4,Refmax_B5,Refmax_B6,Refmax_B7,Refmax_B9]

		# read in Refmin
		Refmin_B2 = float(md['REFLECTANCE_MINIMUM_BAND_2'].strip())
		Refmin_B3 = float(md['REFLECTANCE_MINIMUM_BAND_3'].strip())
		Refmin_B4 = float(md['REFLECTANCE_MINIMUM_BAND_4'].strip())
		Refmin_B5 = float(md['REFLECTANCE_MINIMUM_BAND_5'].strip())
		Refmin_B6 = float(md['REFLECTANCE_MINIMUM_BAND_6'].strip())
		Refmin_B7 = float(md['REFLECTANCE_MINIMUM_BAND_7'].strip())
		Refmin_B9 = float(md['REFLECTANCE_MINIMUM_BAND_9'].strip())
		
		Refmin = [Refmin_B2,Refmin_B3,Refmin_B4,Refmin_B5,Refmin_B6,Refmin_B7,Refmin_B9]

		# Read in nrows & ncols of optical bands
		Sample_ref = int(md['REFLECTIVE_SAMPLES'].strip())
		Line_ref = int(md['REFLECTIVE_LINES'].strip())
		# record ijdimension of optical bands
		ijdim_ref = [Line_ref,Sample_ref]
		
		Sample_thm = float(md['THERMAL_SAMPLES'].strip())
		Line_thm = float(md['THERMAL_LINES'].strip())
		# record thermal band dimensions (i,j)
		ijdim_thm = [Line_thm,Sample_thm]
		
		# Read in resolution of optical and thermal bands
		reso_ref = float(md['GRID_CELL_SIZE_REFLECTIVE'].strip())
		reso_thm = float(md['GRID_CELL_SIZE_THERMAL'].strip())
		
		# Read in UTM Zone Number
		zc = float(md['UTM_ZONE'].strip())
		# Read in Solar Azimuth & Elevation angle (degrees)
		azi = float(md['SUN_AZIMUTH'].strip())
		zen = 90-float(md['SUN_ELEVATION'].strip())
		# Read in upperleft mapx,y
		ulx = float(md['CORNER_UL_PROJECTION_X_PRODUCT'].strip())
		uly = float(md['CORNER_UL_PROJECTION_Y_PRODUCT'].strip())
		ul = [ulx,uly]
		# Read in date of year
		char_doy = md['LANDSAT_SCENE_ID'].strip()
		doy = int(char_doy[13:16])
	else:
		print('This sensor is not Landsat 4, 5, 7, or 8!\n')
	return([Lmax,Lmin,Qcalmax,Qcalmin,Refmax,Refmin,ijdim_ref,ijdim_thm,reso_ref,reso_thm,ul,zen,azi,zc,Lnum,doy])

