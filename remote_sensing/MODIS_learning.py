###Title: MODIS_learning.py
#Author: Tony Chang
#Abstract: Test for opening MODIS data and examining the various bands, and relearning the product to work with it
#Date: 01/20/2016

#We are working with Level 2 MODIS product for surface reflectance
'''
The MODIS Surface-Reflectance Product (MOD 09) is computed from the MODIS Level 1B land 
bands 1, 2, 3, 4, 5, 6, and 7 (centered at 648 nm, 858 nm, 470 nm, 555 nm, 1240 nm, 1640 nm, and 2130 nm, 
respectively). The product is an estimate of the surface spectral reflectance for each 
band as it would have been measured at ground level if there were no atmospheric scattering or absorption.

The surface reflectance products are generated from the first two, or seven, bands of the
corresponding full 36 band scenes. These provide an estimated “at surface” spectral
reflectance. Several algorithms are applied to various MODIS bands to remove the
effects of cirrus clouds, water vapor, aerosols and atmospheric gases. Global surface
reflectance products can be obtained at either 250m with 2 bands or 500m with 7 bands,
as daily or 8-day composite images.
The data type is 16 bit signed integer, which has a theoretical range of values from -
32,768 to +32,768. The documented data range is from -100 to +16000 with a fill value
of -28,672. If you wish to convert these numbers to a valid reflectance data range, cell
values should be divided by 10,000. These data must then be stored with a floating point
data type.

The data are provided in the HDF-EOS format. MODIS data at version 4 and above use
the Sinusoidal projection with the WGS84 datum. A very small sample of common
products are listed below.

#MODIS file name as 
# 7 char (product name .)
# 8 char (A YYYYDDD .)
# 6 char (h XX v YY .) #tile index
# 3 char (collection version .) #typically 005
# 14 char (julian date of production YYYYDDDHHMMSS)

MOD09A1 - MODIS Surface Reflectance 8-Day L3 Global 500m
This file is a composite using eight consecutive daily 500 m images. The “best”
observation during each eight day period, for every cell in the image, is retained. This
helps reduce or eliminate clouds from a scene. The file contains the same seven spectral
bands of data as the daily file listed above. Files with 500m resolution contain the 7 bands 
of data in the Visible (400-700 nm), Near-IR (700-2000 nm), and Mid-IR (2000-4000 nm) parts of the spectrum. It also has an additional 
6 bands of information concerning quality control, solar zenith, view zenith, relative 
azimuth, surface reflectance 500 m state flags, and surface reflectance day of year.

Some information regarding MOD09A1
BAND, RANGE nm reflected, RANGE um emitted, KEY USE
1 620-670, ,Absolute Land Cover Transformation, Vegetation Chlorophyll
2 841-876, ,Cloud Amount, Vegetation Land Cover Transformation
3 459-479, ,Soil/Vegetation Differences
4 545-565, ,Green Vegetation
5 1230-1250, ,Leaf/Canopy Differences
6 1628-1652, ,Snow/Cloud Differences
7 2105-2155, ,Cloud Properties, Land Properties

Surface reflectance is the amount of light reflected by the surface of the earth; it is a ratio of surface radiance to surface irradiance, and as such is unitless, and typically has values between 0.0 and 1.0. MOD09's surface reflectance values are scaled by 10000 and then cast to 16-bit integers, so surface reflectance values in MOD09 files are typically between 0 and 10000. The atmospheric correction algorithm that is used results in values normally between -100 and 16000. Any values outside of this range are either uncorrected L1B data (e. g., data at high solar zenith angles) or fill values (e. g., data between orbits in L2G or L2G-lite files).

This information was obtained from the Land Process Distributed Active Archive Center
website on 17 May 2010 at the following URL:
https://lpdaac.usgs.gov/lpdaac/products/modis_overview

Nice summary!!!Thanks YALE.

Next is the Tassel Cap coefficients for MODIS MOD09 from Zhang et al 2002

MODIS (nm), 620–670, 841–876, 459–479, 545–565, 1230–1250, 1628–1652, 2105–2155
Band name, Red, Near-IR, Blue, Green, M-IR, M-IR, M-IR

Brightness, 0.3956, 0.4718, 0.3354, 0.3834, 0.3946, 0.3434, 0.2964
Greenness, −0.3399, 0.5952, −0.2129, −0.2222, 0.4617, −0.1037, −0.46
Wetness, 0.10839, 0.0912, 0.5065, 0.404, −0.241, −0.4658, −0.5306


so if we use our get_mask function, value represents the value we want which is 0 at the least significant
bit (far right of a string of bits) i.e. 32 bit integer value 1073741824 is '1000000000000000000000000000000'
meaning it has adjacency correction performed and is a corrected product at ideal quality
or integer value 1073742657 returns '1000000000000000000001101000001' which is corrected as less than ideal quality
has a correction out of bounds for band 2 and the pixel was constrained to max allowable value.
One way to deal with this is to exclude everything but bit numbers 0 and 1. 

#let's test this thing....so if we have imagine a 6 bit integer that is 000010 then that should be 2
or we can have 100110 which is 38 or 111011 which is 59 and finally, 100000 which is 32, 111111 as 63, 100101 as 37.
so in an array that is 
[000010,100110,111011,100000,111111,100101]
[2,38,59,32,63,37] we want to get the first two LSB to equal 0
[0,0,0,1,0,0]
'''




