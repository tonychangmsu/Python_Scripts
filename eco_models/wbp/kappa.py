#kappa function test...
import numpy as np
from matplotlib import pyplot as plt

def kappa(po,pe):
	return((po-pe)/(1-pe))

def po(p, sn, sp):
	return(p*sn+((1-p)*sp))

def pe(p, po, sn, sp):
	return(-2 * (sn + sp - 1) * p * (1 - p) + po)

def tss(sn,sp):
	return(sn+sp-1)
	
	
p = 932/2545
sn = 0.923819
sp = 0.83
p = np.arange(0,1,0.05)
p_o = po(p,sn,sp)
p_e = pe(p,p_o,sn,sp)
k = kappa(p_o, p_e)
plt.plot(p,k)
plt.xlabel('prevalence')
plt.ylabel('kappa')	
plt.grid()

ts = tss(sn,sp)
plt.plot(p,tss)
plt.grid()

#0.407 threshold : sn = 0.873, sp = 0.873 (equal sn and sp)
#0.334 threshold : sn = 0.924, sp = 0.831 (max sst)
#0.467 threshold : sn = 0.852, sp = 0.897 (max kappa)
