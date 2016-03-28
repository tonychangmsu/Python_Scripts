#PRISM data analysis
""" To be called after PRISM_extract, will write functions for the
PRISM_extract code once I'm done with this section.

Data is stored in a list of classes called PData in the following format:

PData[index].year
PData[index].month
PData[index].ncols
PData[index].nrows
PData[index].xll
PData[index].yll
PData[index].csize
PData[index].NODATA
PData[index].PRval

PRval contains a NumPy array of the correponding PRISM data
"""

n = Pdata[0].nrows
m = Pdata[0].ncols
df = len(Pdata)

sumy = zeros((n,m)) #declare an empty array 
sumx = ones((n,m))
sumxy = zeros((n,m))
sumxsq = zeros((n,m))

for i in range(df):
    if Pdata[i].month == 14:
        sumy = sumy + Pdata[i].PRval.astype(float)
        sumx = sumx + (ones((n,m)) * Pdata[i].year)
        sumxsq = sumxsq + (sumx*sumx)
        sumxy = sumxy + (sumx * sumy)

ybars = sumy/df
xbars = sumx/df

Sxx = sumxsq - ((sumx*sumx)/df)
Sxy = sumxy - ((sumx*sumy)/df)

gMat = Sxy/Sxx
