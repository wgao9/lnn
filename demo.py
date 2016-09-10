import numpy.random as nr
from math import log, pi, exp
import lnn 

def main():
	r = 0.9999
	data = nr.multivariate_normal([0,0],[[1,r],[r,1]],100)
	print "Entropy: "
	print "Ground Truth = ", log(2*pi*exp(1))+0.5*log(1-r**2)
	print "LNN: H(X) =  ", lnn.entropy(data)
	print "KDE: H(X) = ", lnn.KDE_entropy(data)
	print "KL: H(X) = ", lnn.KL_entropy(data)
	print "LNN(1st order): H(X) = ", lnn.LNN_1_entropy(data), "\n"

	print "Mutual Information: "
	print "Ground Truth = ", -0.5*log(1-r**2)
	print "LNN: I(X;Y) =  ", lnn.mi(data,split=1)
	print "KDE: I(X;Y) =  ", lnn._3KDE_mi(data,split=1)
	print "3KL: I(X;Y) =  ", lnn._3KL_mi(data,split=1)
	print "KSG: I(X;Y) =  ", lnn._KSG_mi(data,split=1)
	print "LNN(1st order): I(X;Y) =  ", lnn._3LNN_1_mi(data,split=1)
	print "LNN(1st order, KSG trick): I(X;Y) =  ", lnn._3LNN_1_KSG_mi(data,split=1)
	print "LNN(2nd order, KSG trick): I(X;Y) =  ", lnn._3LNN_2_KSG_mi(data,split=1)


if __name__ == '__main__':
	main()


