#patch analysis tools
#author: Tony Chang

import numpy as np
import scipy as sp
from scipy import ndimage

def get_patch(mat, method = 8):
	if method ==8: #default 8 neighbor cell method
		s = sp.ndimage.generate_binary_structure(2,2)
		labeled_array, numpatches = sp.ndimage.label(mat, s)
	else: #4 neighbor cell method
		labeled_array, numpatches = sp.ndimage.label(mat)
	sizes = sp.ndimage.sum(mat, labeled_array, range(1, numpatches+1))
	return(labeled_array, numpatches, sizes)
	
def extract_edges(mat, size):
	edge = sp.ndimage.distance_transform_edt(mat==0) ==1
	if (size > 1):
		s = sp.ndimage.generate_binary_structure(2,1)
		edge = sp.ndimage.binary_dilation(edge, s, iterations=size-1)
	return(edge)
	
