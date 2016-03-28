#Title: Write kml screen overlays
#Author: Tony Chang
#Date: 10/10/14
#Abstract: 	Code to draw screen overlays of the date and colorbar for wbp timeseries ground overlays 
#Dependencies: Python 3.X
import datetime as dt
dt = date.today().strftime("%m%d%y")
workSpace = 'E:\earth'
name = 'wbp_screen_overlays_%s'%(dt)
kmlName = '%s\%s.kml' %(workSpace,name)
f = open(kmlName, 'w')
#write the header
f.write('<?xml version="1.0" encoding="UTF-8"?>\n<kml xmlns="http://www.opengis.net/kml/2.2"\n xmlns:gx="http://www.google.com/kml/ext/2.2">\n')
#write the document
f.write('<Document>\n\t<name>%s</name>\n'%name)

#loop through GroundOverlays
bBox = [46.1988223352,42.2821556701,-108.202443828,-112.394110493]
labelPoint =[-109.021,45.583]
folderName = 'WBP_projections\PNG6'
fileHeader = 'WBP_prob_CESM1-BGC_'
startYear = 2010
endYear = 2099
for i in range(startYear, endYear+1):
	#write the raster portion
	n = 'WBP_%i_climate_envelope'%(i)
	f.write('\t<GroundOverlay>\n\t\t<name>%s</name>\n'%(n))
	#write timeSpan
	f.write('\t\t<TimeSpan>\n\t\t<begin>%i-01</begin>\n\t\t<end>%i-01</end>\n\t\t</TimeSpan>\n'%(i,i+1))
	#write Icon
	pngname = "%s\%s\%s%i.png" %(workSpace,folderName,fileHeader,i)
	#pngname = "files\%s%i.png" %(fileHeader,i)
	f.write('\t\t<Icon>\n\t\t\t<href>%s</href>\n\t\t</Icon>\n'%(pngname))
	#write LatLonBox
	f.write('\t\t<LatLonBox>\n\t\t\t<north>%.10f</north>\n\t\t\t<south>%.10f</south>\n' %(bBox[0],bBox[1]))
	f.write('\t\t\t<east>%.9f</east>\n\t\t\t<west>%.9f</west>\n\t\t</LatLonBox>\n' %(bBox[2],bBox[3]))
	f.write('\t</GroundOverlay>\n')
	
#loop through ScreenOverlays
#define the years of the overlays
startYear = 2010
endYear = 2099
folderName = "E:\earth\screen_overlays"
x = -0.1
y = 2

for i in range(startYear, endYear+1):
	f.write('\t<ScreenOverlay>\n\t\t<name>WBP_year_%i</name>\n'%(i))
	#write timeSpan
	f.write('\t\t<TimeSpan>\n\t\t<begin>%i-01</begin>\n\t\t<end>%i-12</end>\n\t\t</TimeSpan>\n'%(i,i))
	#write Icon
	pngname = "%s\%i.png" %(folderName,i)
	f.write('\t\t<Icon>\n\t\t\t<href>%s</href>\n\t\t</Icon>\n'%(pngname))
	#write overlayXY
	f.write('\t\t<overlayXY x="%s" y="%s" xunits="fraction" yunits="fraction"/>\n' %(str(x),str(y)))
	f.write('\t\t<screenXY x="0" y="1" xunits="fraction" yunits="fraction"/>\n')
	f.write('\t\t<rotationXY x="0" y="0" xunits="fraction" yunits="fraction"/>\n')
	f.write('\t\t<size x="0" y="0" xunits="fraction" yunits="fraction"/>\n')
	f.write('\t</ScreenOverlay>\n')


colorbar = 'legend'#'colorbar_alpha'
#add screen overlay for the colorbar
x = -0.07
y = 2.9
f.write('\t<ScreenOverlay>\n\t\t<name>WBP_colorbar</name>\n')
#write Icon
pngname = "%s\%s.png" %(folderName,colorbar)
f.write('\t\t<Icon>\n\t\t\t<href>%s</href>\n\t\t</Icon>\n'%(pngname))
#write overlayXY
f.write('\t\t<overlayXY x="%s" y="%s" xunits="fraction" yunits="fraction"/>\n' %(str(x),str(y)))
f.write('\t\t<screenXY x="0" y="1" xunits="fraction" yunits="fraction"/>\n')
f.write('\t\t<rotationXY x="0" y="0" xunits="fraction" yunits="fraction"/>\n')
f.write('\t\t<size x="0" y="0" xunits="fraction" yunits="fraction"/>\n')
f.write('</ScreenOverlay>\n')

#end the kml
f.write('</Document>\n</kml>')
f.close()
	