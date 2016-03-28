#Title: Write kml animation
#Author: Tony Chang
#Date: 09/11/14
#Abstract: 	Code to write data WBP probability surfaces to kml for time-series overlay animation
#			To be run after probs_to_png.py file, that converts the numpy arrays to png files of the 
#			correct pixel dimensions for display
#Dependencies: Python 3.X

workSpace = 'E:\earth'
name = 'overlay_animation'
kmlName = '%s\%s.kml' %(workSpace,name)
f = open(kmlName, 'w')
#write the header
f.write('<?xml version="1.0" encoding="UTF-8"?>\n<kml xmlns="http://www.opengis.net/kml/2.2"\n xmlns:gx="http://www.google.com/kml/ext/2.2">\n')
#write the document
f.write('<Document>\n\t<name>%s</name>\n'%name)

#loop through GroundOverlays
#define the years of the overlays
startYear = 2010
endYear = 2099
bBox = [46.1988223352,42.2821556701,-108.202443828,-112.394110493]
folderName = 'WBP_projections\PNG'
fileHeader = '\WBP_prob_CESM1-BGC_'
for i in range(startYear, endYear+1):
	f.write('\t<GroundOverlay>\n\t\t<name>WBP_%i_climate_envelope</name>\n'%(i))
	#write timeSpan
	f.write('\t\t<TimeSpan>\n\t\t<begin>%i-01</begin>\n\t\t<end>%i-12</end>\n\t\t</TimeSpan>\n'%(i,i))
	#write Icon
	pngname = "%s\%s%s%i.png" %(workSpace,folderName,fileHeader,i)
	f.write('\t\t<Icon>\n\t\t\t<href>%s</href>\n\t\t</Icon>\n'%(pngname))
	#write LatLonBox
	f.write('\t\t<LatLonBox>\n\t\t\t<north>%.10f</north>\n\t\t\t<south>%.10f</south>\n' %(bBox[0],bBox[1]))
	f.write('\t\t\t<east>%.9f</east>\n\t\t\t<west>%.9f</west>\n\t\t</LatLonBox>\n' %(bBox[2],bBox[3]))
	f.write('\t</GroundOverlay>\n')
#end the kml
f.write('</Document>\n</kml>')
f.close()
	