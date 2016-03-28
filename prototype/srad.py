'''
#solar radiation function 
% PUPROSE: Calculate solar radiation for a digital elevation model (DEM)
%          over one year for clear sky conditions in W/m2
% -------------------------------------------------------------------
% USAGE: srad = solarradiation(dem,lat,cs)
% where: dem is the DEM to calculate hillshade for
%        lat is the latitude vector for the DEM - same size as size(dem,2)
%        cs is the cellsize in meters
%        r is the ground reflectance (global value or map, default is 0.2)
%
%       srad is the solar radiation in W/m2 over one year per grid cell
%
% Reference: Kumar, L, Skidmore AK and Knowles E 1997: Modelling topographic variation in solar radiation in 
%            a GIS environment. Int.J.Geogr.Info.Sys. 11(5), 475-497
%
%
% Felix Hebeler, Dept. of Geography, University Zurich, May 2008.

%% parameters
%It ;               % total hours of daily sunshine (calculated inline)
%M ;                % air mass ratio parameter (calculated inline)

%L=lat;             %latitude
'''
r = 0.20           #ground reflectance coefficient (more sensible to give as input)
n = 1              #% timestep of calculation over sunshine hours: 1=hourly, 0.5=30min, 2=2hours etc
tau_a = 365	       #% length of the year in days
#tau_m = [31,28,31,30,31,30,31,31,30,31,30,31] #% length of the month in days
tau_m = np.array([31,59,90,120,151,181,212,243,273,304,334,365])
S0 = 1367          #% solar constant W m^-2   default 1367

dr= 0.0174532925   #% degree to radians conversion factor


import numpy as np
from math import *
from matplotlib import pyplot as plt

def Headerextract(gcm='n'):
	#takes arbitrary PRISM or GCM dataset and extracts the header parameters
	if gcm =='n':
		filename = "E:\\PRISM\\ppt\\PRISM800m_ppt1895_1.tif"    
	elif gcm =='y':
		filename = "E:\\CMIP5\\GCM\\CanESM2\\rcp45\\pr\\CanESSM2_rcp45_pr_2006_1.tif" 
	dataset = gdal.Open(filename, GA_ReadOnly)
	ncols = dataset.RasterXSize
	nrows = dataset.RasterYSize
	bands = dataset.RasterCount
	driver = dataset.GetDriver().LongName
	geotransform = dataset.GetGeoTransform()
	xul = geotransform[0]
	yul = geotransform[3]
	csize = geotransform[1]
	header = {'ncols':ncols, 'nrows':nrows,'bands':bands,'driver':driver, 'xul':xul, 'yul':yul, 'csize':csize}
	return(header) #returns header as directory for readability

def Topoextract():
	#since shape of the GCM is different from the PRISM extent, a modified grid is required if gcm parameter is set to 'y'
	aspectPath = "E:\\gye_topo\\aspect_800m.tif"   
	slopePath = "E:\\gye_topo\\slope_800m.tif"   
	elevPath = "E:\\gye_topo\\dem_800m.tif"   
	ds = gdal.Open(aspectPath)
	aspect = np.array(ds.GetRasterBand(1).ReadAsArray())
	ds = gdal.Open(slopePath)
	slope = np.array(ds.GetRasterBand(1).ReadAsArray())
	ds = gdal.Open(elevPath)
	elev = np.array(ds.GetRasterBand(1).ReadAsArray())
	ds = None #close files
	return(aspect[:,:], slope[:,:], elev[:,:]) 
	
def Gridcreate(gcm='n'):
    header = Headerextract(gcm=gcm)
    lat_list = []
    lon_list = []
    nrows = header['nrows']
    ncols = header['ncols']    
    xmin = header['xul']
    ymin = header['yul']
    csize = header['csize']
    for ystep in range(nrows):
        latstep = ymin-ystep*csize
        lat_list.append(latstep)
    for xstep in range(ncols):
        lonstep = xmin+xstep*csize
        lon_list.append(lonstep)
    lat = np.array([lat_list,]*ncols).transpose()
    lon = np.array([lon_list,]*nrows)
    return(lat,lon)

aspect, slope, elevation = Topoextract()
head = Headerextract()
cellsize = 800 #800 meter grid cells
lat,lon = Gridcreate()
L = lat * dr #convert to radians
fcirc = 360 * dr # 360 degrees in radians

slop = slope*dr
asp = (aspect * dr) * -1 + np.pi
#%% some setup calculations

sinL=np.sin(L)
cosL=np.cos(L)
tanL=np.tan(L)
sinSlop=np.sin(slop)
cosSlop=np.cos(slop)
cosSlop2=cosSlop*cosSlop
sinSlop2=sinSlop*sinSlop
sinAsp=np.sin(asp)
cosAsp=np.cos(asp)
term1 = ( sinL*cosSlop - cosL*sinSlop*cosAsp)
term2 = ( cosL*cosSlop + sinL*sinSlop*cosAsp)
term3 = sinSlop*sinAsp
#%% loop over year
srad=0
srad_m = np.zeros((12, head['nrows'],head['ncols']))
for d in range(1,tau_a+1):
#%display(['Calculating melt for day ',num2str(d)])  
#% clear sky solar radiation
	I0 = S0 * (1 + 0.0344*np.cos(fcirc*d/tau_a)) #% extraterr rad per day     
	#% sun declination dS
	dS = 23.45 * dr* np.sin(fcirc * ( (284+d)/tau_a ) ) #%in radians, correct/verified
	#% angle at sunrise/sunset
	#% t = 1:It; % sun hour    
	hsr = np.arccos(-tanL*np.tan(dS)) # % angle at sunrise
	#% this only works for latitudes up to 66.5 deg N! Workaround:
	#% hsr(hsr<-1)=acos(-1);
	#% hsr(hsr>1)=acos(1);
	It = round(12*(1+np.mean(hsr)/np.pi)-12*(1-np.mean(hsr)/np.pi)) #% calc daylength
	#%%  daily loop
	I=0
	for t in range(1,int(It)+1): #% loop over sunshine hours
		#% if accounting for shading should be included, calc hillshade here
		#% hourangle of sun hs  
		hs=hsr-(np.pi*t/It)               #% hs(t)
		#%solar angle and azimuth
		#%alpha = asin(sinL*sin(dS)+cosL*cos(dS)*cos(hs));% solar altitude angle
		sinAlpha = sinL*np.sin(dS)+cosL*np.cos(dS)*np.cos(hs)
		#%alpha_s = asin(cos(dS)*sin(hs)/cos(alpha)); % solar azimuth angle
		#% correction  using atmospheric transmissivity taub_b
		M = np.sqrt(1229+((614*sinAlpha))**2)-614*sinAlpha #% Air mass ratio
		tau_b = 0.56 * (np.exp(-0.65*M) + np.exp(-0.095*M))
		tau_d = 0.271-0.294*tau_b #% radiation diffusion coefficient for diffuse insolation
		tau_r = 0.271+0.706*tau_b #% reflectance transmitivity
		#% correct for local incident angle
		cos_i = (np.sin(dS)*term1) + (np.cos(dS)*np.cos(hs)*term2) + (np.cos(dS)*term3*np.sin(hs))
		Is = I0 * tau_b # % potential incoming shortwave radiation at surface normal (equator)
		#% R = potential clear sky solar radiation W m2
		R = Is * cos_i
		R[R<0]=0  #% kick out negative values
		Id = I0 * tau_d * cosSlop2/ 2*sinAlpha #%diffuse radiation;
		Ir = I0 * r * tau_r * sinSlop2/ 2* sinAlpha# % reflectance
		R= R + Id + Ir
		R[R<0]=0
		I=I+R #% solar radiation per day (sunshine hours)  
		#end % end of sun hours in day loop
		#%%  add up radiation part melt for every day
		srad = srad + I
		for i in range(len(tau_m)):
			if d == tau_m[i]:
				srad_m[i] = srad

#additional loop to calculate the monthly srad totals
for i in range(len(srad_m)-1,0, -1):
	srad_m[i] = srad_m[i]-srad_m[i-1]
		
#end