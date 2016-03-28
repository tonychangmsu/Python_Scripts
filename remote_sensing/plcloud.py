from nd2toarbt import *
from scipy import ndimage as ndi

def plcloud(image,num_Lst, cldprob = False, ledap_processed = True):
	# Function for potential Cloud, cloud Shadow, and Snow Masking 3.2v
	# Works for Landsat TM/ETM+/OLITIRS/ read in the first thermal band
	#
	# History of revisions:
	# Translated code to python 3 (Chang 03/08/2016)
	# Temperature < 283K for all snow (Zhe 11/01/2013)
	# Remove the split window method in cirrus cloud detection (Zhe 10/29/2013)
	# Add dynamic water threshold (Zhe 10/27/2013)
	# Remove majority filter (Zhe 10/27/2013)
	# Fix bugs in probs < 0 (Brightness_prob & wTemp_prob) (Zhe 09/11/2013)
	# Capable of processing Landsat 4~7 images (Zhe 09/09/2013)
	# Use metadata to covert DN to TOA ref (Zhe 09/05/2013)
	# Use the news bands from L8 for better cloud masking (Zhe 06/20/2013) 
	# Capable of processing Landsat 8 (Zhe 04/04/2013)
	# Remove Snow(mask==0)=255; (Zhe 05/24/2012)
	# Made it a stand alone version (Zhe 01/09/2012)
	# Impoved accuracy in TOA reflectance computing in two ways:
	# Firstly, used the more accurate ESUN provied by Chander et al. RSE (2009)
	# Secondly, used more Sun-Earth distance table provied by Chander et al.
	# RSE (2009) (Zhe 01/07/2012)
	# Fixed visible bands abnormal saturation problem (Zhe 04/24/2011)
	# Cloud/shadow prob mask (-1) when no cloud prob computed (Zhe 01/01/2011)
	# Flood fill for band 5 in shadow detection (Zhe 12/23/2010)
	# Fixed saturation pixels (Zhe 12/16/2010)
	# Include the BT test for snow (Zhe 12/06/2010)
	# Include probability mask (Zhe 11/01/2010)
	# Temp < 300k for all clouds (Zhe 12/01/2009)
	# Reduced computing memory from double to single (Zhe 12/01/2009)
	# Detecting clouds for Landsat 8 without using new bands (Zhe 09/04/2013)
	# Add customized snow dilation pixel number (Zhe 09/12/2013)
	# Fix problem in snow detection because of temperature (Zhe 09/12/2013)

	# 'cldprob' is the cloud probability threshold with default
	# values of 22.5 for Landsat 4~7 and 50 for Landsat 8 (range from 0~100).
	# resolution of Fmask results
	
	Temp,data,dim,ul,zen,azi,zc,satu_B1,satu_B2,satu_B3,resolu = nd2toarbt(image, ledap_processed = ledap_processed)

	if num_Lst < 8: # Landsat 4~7
		Thin_prob = 0 # there is no contribution from the new bands
		if not cldprob:
			cldprob = 22.5
	else:
		Thin_prob = data[:,:,-1]/400
		if not cldprob:
			cldprob = 80

	# fprintf('Read in TOA ref ...\n');
	Cloud = np.zeros(dim,'uint8')  # cloud mask
	Snow = np.zeros(dim,'uint8') # Snow mask
	WT = np.zeros(dim,'uint8') # Water msk
	# process only the overlap area
	mask = data[:,:,0] > -9999
	Shadow = np.zeros(dim,'uint8') # shadow mask

	NDVI = (data[:,:,3]-data[:,:,2]) / (data[:,:,3] + data[:,:,2])
	NDSI = (data[:,:,1]-data[:,:,4]) / (data[:,:,1] + data[:,:,4])

	NDVI[(data[:,:,3] + data[:,:,2]) == 0] = 0.01
	NDSI[(data[:,:,1] + data[:,:,4]) == 0] = 0.01

	##############################################saturation in the three visible bands
	satu_Bv = ((satu_B1+satu_B2+satu_B3) >= 1)
	del satu_B1 # clear satu_B;
	################################################## Basic cloud test
	idplcd = ((NDSI < 0.8) & (NDVI < 0.8) & (data[:,:,5] > 300) & (Temp < 2700))
	################################################## Snow test
	# It takes every snow pixels including snow pixel under thin clouds or icy clouds
	Snow[((NDSI>0.15) & (Temp<1000) & (data[:,:,4]>1100) & (data[:,:,1]>1000))] = 1
	# Snow(mask==0)=255;
	################################################## Water test
	# Zhe's water test (works over thin cloud)
	WT[(((NDVI<0.01) & (data[:,:,3]<1100)) | ((NDVI<0.1) & (NDVI>0) & (data[:,:,3] < 500)))] = 1
	# WT(NDVI<0.1&data(:,:,4)<1500) = 1;
	WT[mask == 0] = 255
	# ################################################ Whiteness test
	# visible bands flatness (sum(abs)/mean < 0.6 => brigt and dark cloud )
	visimean = (data[:,:,0] + data[:,:,1] + data[:,:,2]) / 3
	whiteness = (np.abs(data[:,:,0] - visimean) + np.abs(data[:,:,1] - visimean) + np.abs(data[:,:,2] - visimean)) / visimean
	del visimean
	# update idplcd
	whiteness[satu_Bv == 1] = 0# If one visible is saturated whiteness == 0
	idplcd = ((idplcd == True) & (whiteness<0.7))
	# clear whiteness;
	################################################## Haze test
	HOT = data[:,:,0] - 0.5 * data[:,:,2] - 800# Haze test
	idplcd = ((idplcd == True) & ((HOT > 0) | (satu_Bv == 1)))
	del HOT; # need to find thick warm cloud
	################################################## Ratio4/5>0.75 cloud test
	Ratio4_5 = data[:,:,3] / data[:,:,4]
	idplcd = ((idplcd == True) & (Ratio4_5 > 0.75))
	# clear Ratio4_5;
	del Ratio4_5
	############################### Cirrus tests from Landsat 8
	idplcd = ((idplcd == True) | (Thin_prob > 0.25))
	# enviwrite(['Eq6_cloud_potential'],tmpidplcd,'uint8',resolu,ul,'bsq',zc); # Equation 6
	####################################constants##########################
	l_pt = 0.175 # low percent
	h_pt = 1-l_pt # high percent
	################################################(temperature & snow test )
	# test whether use thermal or not
	idclr = ((idplcd==False) & (mask==1)) #&TempDif<3&Cirrus<3; # thin_prob < ...
	ptm = 100 * np.sum(idclr)/np.sum(mask) # percent of clear pixel
	idlnd = (idclr & WT) == False
	idwt = (idclr & WT) == True #&data(:,:,6)<=300;
	lndptm = 100 * np.sum(idlnd) / np.sum(mask)

	if (ptm <= 0.1): # no thermal test => meanless for snow detection (0~1)
		# fprintf('No clear pixel in this image (clear prct = #.2f)\n',ptm);
		Cloud[idplcd==True] = 1 # all cld
		# # improving by majority filtering
		# Cloud=bwmorph(Cloud,'majority');# exclude <5/9
		Shadow[Cloud == 0] = 1
		#Temp = -1
		t_templ = -1
		t_temph = -1
	else:
		# fprintf('Clear pixel EXIST in this scene (clear prct = #.2f)\n',ptm);
		#################################################(temperature test )
		if (lndptm >= 0.1):
			F_temp = Temp[idlnd] # get land temperature
		else:
			F_temp = Temp[idclr] # get clear temperature
		# Get cloud prob over water
		## temperature test (over water)
		F_wtemp = Temp[idwt] # get clear water temperature
		t_wtemp = np.percentile(F_wtemp, 100*h_pt)
		wTemp_prob = (t_wtemp-Temp)/400
		wTemp_prob[wTemp_prob<0] = 0
		#    wTemp_prob(wTemp_prob > 1) = 1;
		## Brightness test (over water)
		t_bright = 1100
		Brightness_prob = data[:,:,5]/t_bright
		Brightness_prob[Brightness_prob>1] = 1
		Brightness_prob[Brightness_prob<0] = 0

		## Final prob mask (water)
		wfinal_prob = 100 * wTemp_prob * Brightness_prob + 100 * Thin_prob # cloud over water probability
		wclr_max = np.percentile(wfinal_prob[idwt], 100 * h_pt) + cldprob# dynamic threshold (land)
		#    wclr_max=50;# fixed threshold (water)

		# release memory
		del wTemp_prob
		del Brightness_prob

		## Temperature test
		t_buffer = 4*100
		# 0.175 percentile background temperature (low)
		t_templ = np.percentile(F_temp,100*l_pt)
		# 0.825 percentile background temperature (high)
		t_temph = np.percentile(F_temp,100*h_pt)

		t_tempL = t_templ-t_buffer
		t_tempH = t_temph+t_buffer
		Temp_l = t_tempH-t_tempL
		Temp_prob = (t_tempH-Temp)/Temp_l
		# Temperature can have prob > 1
		Temp_prob[Temp_prob<0] = 0
		#    Temp_prob(Temp_prob > 1) = 1;

		NDSI[((satu_B2==True) & (NDSI<0))] = 0
		NDVI[((satu_B3==True)& (NDVI>0))] = 0
		vari = np.max(np.array([np.abs(NDSI),np.abs(NDVI)]),axis = 0)
		Vari_prob = 1 - np.max(np.array([vari,whiteness]),axis = 0)

		# release memory
		del satu_B2
		del satu_B3
		del NDSI
		del NDVI
		del whiteness

		## Final prob mask (land) 
		final_prob = 100 * Temp_prob * Vari_prob + 100 * Thin_prob # cloud over land probability
		clr_max = np.percentile(final_prob[idlnd], 100 * h_pt) + cldprob # dynamic threshold (land)

		# release memory
		del Vari_prob
		del Temp_prob
		del Thin_prob
		cloud_over_land = ((idplcd == True) & (final_prob > clr_max) & (WT == 0))
		thin_cloud_over_water = ((idplcd == True) & (wfinal_prob > wclr_max) & (WT == 1))
		extremely_cold_cloud = Temp < (t_templ - 3500)
		id_final_cld = (cloud_over_land | thin_cloud_over_water | extremely_cold_cloud)

		## Star with potential cloud mask
		# # potential cloud mask
		Cloud[id_final_cld] = 1
		# # improving by majority filtering
		# Cloud=bwmorph(Cloud,'majority');# exclude <5/9

		# release memory
		del final_prob
		#      final_prob = final_prob - clr_max;
		del wfinal_prob
		del id_final_cld
		## Star with potential cloud shadow mask

		# band 4 flood fill
		nir = data[:,:,3]
		# estimating background (land) Band 4 ref
		backg_B4 = np.percentile(nir[idlnd], 100*l_pt)
		nir[mask==0] = backg_B4
		# fill in regional minimum Band 4 ref
		#imfill(nir)
		nir = ndi.binary_fill_holes(nir)
		nir = nir - data[:,:,3]

		# band 5 flood fill
		swir = data[:,:,4]
		# estimating background (land) Band 4 ref
		backg_B5 = np.percentile(swir[idlnd],100*l_pt)
		swir[mask==0] = backg_B5
		# fill in regional minimum Band 5 ref
		swir = ndi.binary_fill_holes(swir)
		swir = swir - data[:,:,4]

		# compute shadow probability
		shadow_prob = np.min(np.array([nir,swir]), axis = 0)
		# release remory
		del nir
		del swir

		Shadow[shadow_prob>200] = 1
		# release remory
		del shadow_prob

	# refine Water mask - Zhe's water mask (no confusion water/cloud)
	WT[((WT==1) & (Cloud==0))] = 1
	# bwmorph changed Cloud to Binary
	#Cloud = Cloud.astype('uint8')
	Cloud[mask==0] = 255
	Shadow[mask==0] = 255
	# enviwrite('TOA3.2',data,'int16',resolu,ul,'bsq',zc);
	# enviwrite('BT3.2',Temp,'int16',resolu,ul,'bsq',zc);
	# enviwrite('final_prob',final_prob,'single',resolu,ul,'bsq',zc);
	# enviwrite('thin_prob',100*Thin_prob,'single',resolu,ul,'bsq',zc);

	return([zen,azi,ptm,Temp,t_templ,t_temph,WT,Snow,Cloud,Shadow,dim,ul,resolu,zc])
