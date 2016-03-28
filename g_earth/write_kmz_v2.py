#Title: Write kml animation with the vector of previous extent
#Author: Tony Chang
#Date: 09/16/14
#Abstract: 	Code to write data WBP probability surfaces to kml for time-series overlay animation
#			To be run after probs_to_png.py file, that converts the numpy arrays to png files of the 
#			correct pixel dimensions for display
#			Additional adds the polygons that delineate the edges of the 0.421 probability areas to 
#			illustrate the previous (2010) extent
#Dependencies: Python 3.X, ogr, tarfile, os

import ogr
import tarfile
import os

def getPolygon():
	#input the file object: f, the desired line width: lineWidth, and 8 char colorCode
	#opens the shapefile and writes each feature as a new polygon into the kml file
	workSpace = 'E:\earth'
	nm = 'WBP_ref_poly'
	shpFile = '%s\%s\%s.shp' %(workSpace, nm, nm)
	shp = ogr.Open(shpFile)
	lyr = shp.GetLayer()
	nf = lyr.GetFeatureCount()
	fpointsX = []; fpointsY = []
	for n in range(nf):
		feat = lyr.GetNextFeature()
		geom = feat.GetGeometryRef()
		ring = geom.GetGeometryRef(0)
		numpoints = ring.GetPointCount()
		pointsX = []; pointsY = []
		for p in range(numpoints):
			lon, lat, z = ring.GetPoint(p)
			pointsX.append(lon)
			pointsY.append(lat)
		fpointsX.append(pointsX)
		fpointsY.append(pointsY)
	return(fpointsX, fpointsY)

workSpace = 'E:\earth'
name = 'overlay_animation3'
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
#labelPoint =[((bBox[2]+bBox[3])/2),((bBox[0]+bBox[1])/2)]
labelPoint =[-109.021,45.583]
folderName = 'WBP_projections\PNG3'
fileHeader = 'WBP_prob_CESM1-BGC_'
for i in range(startYear, endYear+1):
	#write the raster portion
	f.write('\t<GroundOverlay>\n\t\t<name>WBP_%i_climate_envelope</name>\n'%(i))
	#write timeSpan
	f.write('\t\t<TimeSpan>\n\t\t<begin>%i-01</begin>\n\t\t<end>%i-01</end>\n\t\t</TimeSpan>\n'%(i,i+1))
	#write Icon
	#pngname = "%s\%s%s%i.png" %(workSpace,folderName,fileHeader,i)
	pngname = "files3/%s%i.png" %(fileHeader,i)
	f.write('\t\t<Icon>\n\t\t\t<href>%s</href>\n\t\t</Icon>\n'%(pngname))
	#write LatLonBox
	f.write('\t\t<LatLonBox>\n\t\t\t<north>%.10f</north>\n\t\t\t<south>%.10f</south>\n' %(bBox[0],bBox[1]))
	f.write('\t\t\t<east>%.9f</east>\n\t\t\t<west>%.9f</west>\n\t\t</LatLonBox>\n' %(bBox[2],bBox[3]))
	f.write('\t</GroundOverlay>\n')
	'''
	#write point portion
	f.write('\t<Placemark>\n\t\t<name>%i</name>\n\t\t<Snippet maxLines="0">empty</Snippet>\n'%(i))
	#write Point
	f.write('\t\t<TimeSpan>\n\t\t<begin>%i-01</begin>\n\t\t<end>%i-01</end>\n\t\t</TimeSpan>\n'%(i,i+1))
	f.write('\t\t\t<Point>\n\t\t\t\t<altitudeMode>clampToGround</altitudeMode>\n')
	f.write('\t\t\t\t<coordinates>%.8f,%.8f</coordinates>\n'%(labelPoint[0],labelPoint[1]))
	f.write('\t\t\t</Point>\n')
	f.write('\t</Placemark>\n')
	'''
#write the vector portion
	
# lineWidth = 4
# colorCode = '641400FF' # colors can be found on http://www.zonums.com/gmaps/kml_color/
# fpointsX, fpointsY = getPolygon()

# # write the style
# f.write('\n\t<Style id="customStyle">\n')
# f.write('\t\t<LineStyle>\n\t\t\t<width>%i</width>\n\t\t\t<color>%s</color>\n\t\t</LineStyle>\n' %(lineWidth, colorCode))
# f.write('\t\t<PolyStyle>\n\t\t\t<color>0FFFFFF</color>\n\t\t</PolyStyle>\n')
# f.write('\t</Style>\n')
# f.write('\t<Placemark>\n\t\t<name>WBP_bounds</name>\n\t\t<styleUrl>#customStyle</styleUrl>\n')
# f.write('\t\t<MultiGeometry>\n')
# #now write each of the polygons
# for i in range(len(fpointsX)):
	# f.write('\t\t<Polygon>\n\t\t\t<altitudeMode>clampToGround</altitudeMode>\n')
	# f.write('\t\t\t<outerBoundaryIs>\n\t\t\t\t<LinearRing>\n\t\t\t\t\t<coordinates>\n')
	# for j in range(len(fpointsX[i])):
		# f.write("\t\t\t\t\t\t%.12f,%.13f\n" %(fpointsX[i][j],fpointsY[i][j]))
	# f.write('\t\t\t\t\t</coordinates>\n\t\t\t\t</LinearRing>\n\t\t\t</outerBoundaryIs>\n')
	# f.write('\t\t</Polygon>\n')
# f.write('\t\t</MultiGeometry>\n\t</Placemark>\n')
# #end the kml
f.write('</Document>\n</kml>')
f.close()

'''
#now generate the kmz file
os.chdir(workSpace)
kmzName = 'ts_animation.kmz'
kmlFile = '%s.kml' %(name)
k = tarfile.open(kmzName, 'w')
k.add(kmlFile)
k.add(folderName)
k.close()	
'''