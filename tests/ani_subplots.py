# Uncomment the next two lines if you want to save the animation
#import matplotlib
#matplotlib.use("Agg")

import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import host_subplot
import matplotlib.animation as animation
import pandas as pd


# Sent for figure
font = {'size'   : 9}
plt.rc('font', **font)

# Setup figure and subplots
f0 = plt.figure(num = 0, figsize = (12, 8))#, dpi = 100)
f0.suptitle("Oscillation decay", fontsize=12)

ax05 = plt.subplot2grid((1,2),(0,0))
ax06 = plt.subplot2grid((1,2),(0,1))
#tight_layout()

# Set titles of subplots
ax05.set_title('sample image')
'''
# set y-limits
ax01.set_ylim(0,2)
ax02.set_ylim(-6,6)
ax03.set_ylim(-0,5)
ax04.set_ylim(-10,10)

# sex x-limits
ax01.set_xlim(0,5.0)
ax02.set_xlim(0,5.0)
ax03.set_xlim(0,5.0)
ax04.set_xlim(0,5.0)

# Turn on grids
ax01.grid(True)
ax02.grid(True)
ax03.grid(True)
ax06.grid(True)

# set label names
ax01.set_xlabel("x")
ax01.set_ylabel("py")
ax02.set_xlabel("t")
ax02.set_ylabel("vy")
ax03.set_xlabel("t")
ax03.set_ylabel("py")
ax04.set_ylabel("vy")

# Data Placeholders
yp1=np.zeros(0)
yv1=np.zeros(0)
yp2=np.zeros(0)
yv2=np.zeros(0)
'''
t=np.zeros(0)

#Set of random images
images = np.random.randn(250,10,10)
imean = []
for j in range (len(images)):
	imean.append(np.mean(images[j]))
imean=np.array(imean)
j = np.arange(len(imean))
b1,b0 = np.polyfit(j,imean,1)
isim = b1*j+b0
# set plots
'''
p011, = ax01.plot(t,yp1,'b-', label="yp1")
p012, = ax01.plot(t,yp2,'g-', label="yp2")

p021, = ax02.plot(t,yv1,'b-', label="yv1")
p022, = ax02.plot(t,yv2,'g-', label="yv2")

p031, = ax03.plot(t,yp1,'b-', label="yp1")
p032, = ax04.plot(t,yv1,'g-', label="yv1")
'''
im = ax05.imshow(images[0])
p04, = ax06.plot(j,imean)
p05, = ax06.plot(j,isim, '--',color='red', label ="trend line")

# set lagends
'''
ax01.legend([p011,p012], [p011.get_label(),p012.get_label()])
ax02.legend([p021,p022], [p021.get_label(),p022.get_label()])
ax03.legend([p031,p032], [p031.get_label(),p032.get_label()])
'''
# Data Update
i=0
def updateData(self):
	global x
	global yp1
	global yv1
	global yp2
	global yv2
	global t
	global images
	global i
	global imean

	tmpp1 = 1 + np.exp(-x) *np.sin(2 * np.pi * x)
	tmpv1 = - np.exp(-x) * np.sin(2 * np.pi * x) + np.exp(-x) * np.cos(2 * np.pi * x) * 2 * np.pi
	yp1=np.append(yp1,tmpp1)
	yv1=np.append(yv1,tmpv1)
	yp2=np.append(yp2,0.5*tmpp1)
	yv2=np.append(yv2,0.5*tmpv1)
	t=np.append(t,x)

	x += 0.05
	i += 1
	p011.set_data(t,yp1)
	p012.set_data(t,yp2)

	p021.set_data(t,yv1)
	p022.set_data(t,yv2)

	p031.set_data(t,yp1)
	p032.set_data(t,yv1)
	
	im.set_data(images[i])
	
	p04.set_data(j[:i],imean[:i])
	p05.set_data(j[:i],isim[:i])
	if x >= xmax-1.00:
		p011.axes.set_xlim(x-xmax+1.0,x+1.0)
		p021.axes.set_xlim(x-xmax+1.0,x+1.0)
		p031.axes.set_xlim(x-xmax+1.0,x+1.0)
		p032.axes.set_xlim(x-xmax+1.0,x+1.0)

	return (p011, p012, p021, p022, p031, p032)

# interval: draw new frame every 'interval' ms
# frames: number of frames to draw
simulation = animation.FuncAnimation(f0, updateData, blit=False, frames=200, interval=20, repeat=False)

# Uncomment the next line if you want to save the animation
#simulation.save(filename='sim.mp4',fps=30,dpi=300)

plt.show()
