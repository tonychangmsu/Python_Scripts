# pixel to map
def lndpixel2map(jiUL,j,i,resolu):
	# from pixel to map
	# Input jiUL (upperleft corner of the pixel) meters (x,y)
	# resolution [x, y]
	# output x, y of center of the pixel i,j
	# i.e. [x,y]=lndpixel2maplndpixel2map(jiUL,j,i,resolu)
	x = jiUL[0] + resolu[0] / 2 + (j - 1) * resolu[0]
	y = jiUL[1] - resolu[1] / 2 + (1 - i) * resolu[1]
	return([x,y])

# map to pixel
def lndmap2pixel(jiUL,x,y,resolu):
	# from map to pixel
	# Input jiUL (upperleft corner of the pixel), and x meters, and y meters, and
	# resolution [x, y]
	# output j cols, i rows
	# i.e. [j,i]=lndmap2pixel(jiUL,x,y,resolu)
	j = np.ceil((x - jiUL[0]) / resolu[0]).astype(int)
	i = np.ceil((jiUL[1] - y) / resolu[1]).astype(int)
	return([j,i])

def pixel2pixv(jiul1,jiul2,resolu1,resolu2,im2,jidim1,jidim2):
#function outim=pixel2pixv(jiul1,jiul2,resolu1,resolu2,im2,jidim1,jidim2)
# # from stack images of different resolution and different projection 
# # Improved version of lndpixel2pixv
# # match image2 with image1
# # Input 1) jiul1 (upperleft corner of the pixel) of image 1
# # Input 2) jiul2 (upperleft corner of the pixel) of image 2
# # Input 3&4) j cols, i rows of image 1
# # Input 5&6) resolution of images 1&2
# # Input 7) image 2 data 
# # Input 8&9) image 1 and 2 dimension
# # output matched data = outim(f(i),f(j))=>im1(i,j)
# # i.e. matchdata=pixel2pixv(ul1,ul2,res1,res2,data2,dim1,dim2);

	j = np.arange(int(jidim1[0]))
	i = np.arange(int(jidim1[1]))
	[x,y] = lndpixel2map(jiul1,j,i,resolu1)
	[j2,i2] = lndmap2pixel(jiul2,x,y,resolu2)

# the first data is assume to be the filled value
	fill_v = im2[0,0]
	outim = fill_v * np.ones((jidim1[1],jidim1[0]), dtype=fill_v.dtype) # give filled data first

	jexist = (((j2 >= 0) & (j2 <= jidim2[0]))) # matched data i,j exit in data2
	iexist = (((i2 >= 0) & (i2 <= jidim2[1])))
	#jexistv=j2(((j2 > 0)&(j2 <= jidim2(1)))); # exist i,j => data2 (f(i),f(j))
	#iexistv=i2(((i2 > 0)&(i2 <= jidim2(2))));
	jexistv = j2[jexist] # exist i,j => data2 (f(i),f(j))
	iexistv = i2[iexist]
	outim = im2[iexistv[0]:iexistv[-1],jexistv[0]:jexistv[-1]]
	return(outim)







