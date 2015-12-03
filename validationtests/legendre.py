from math import *
def legendre(l,x,m=0):
	if m > l or x > 1.:
		return 'bad arguments!'
	if m >=0:
		return legendre_posm(l,x,m)
	else:
		am = abs(m)
		return -1.**am*factorial(l-am)/factorial(l+am)*legendre_posm(l,x,am)

def legendre_posm(l,x,m=0):

	pmm = 1.
	if m > 0:
		somx2 = sqrt((1.-x)*(1.+x))
		fact = 1.
		for i in range(1,m+1):
			pmm *= -1.*fact*somx2
			fact += 2.
	if l == m:
		return pmm
	
	pmmp1 = x*(2.*m+1.)*pmm
	if l == m+1:
		return pmmp1
	
	for ll in range(m+2,l+1):
		pll = (x*(2.*ll-1.)*pmmp1-(ll+m-1.)*pmm)/float((ll-m))
		pmm = pmmp1
		pmmp1 = pll
	
	return pll
		