#script to screen capture everything and then clip it to automate the video
import sys
import os
from PyQt4.QtGui import QPixmap, QApplication
from PIL import Image

os.chdir('D:\\CHANG\\PhD_Material\\Conferences_workshops\\102014_GYCC_WBP')
app = QApplication(sys.argv)

QPixmap.grabWindow(QApplication.desktop().winId()).save('temp.png', 'png')
im = Image.open('temp.png')

l = 200 
u = 139
r = 1833
d = 943
name = 'PIAL_2070.png'
im.crop((l,u,r,d)).save(name)