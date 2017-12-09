import skimage.io as skio
import skimage as sk
from skimage.transform import rotate
import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as interpolate
import scipy.optimize as optimize

def intersectionPoints(xarray, yarray1, yarray2):
	p1 = interpolate.interp1d(xarray, yarray1, bounds_error = False, fill_value=0.)
	p2 = interpolate.interp1d(xarray, yarray2, bounds_error = False, fill_value=0.)
	
	def pdiff(x):
		return p1(x)-p2(x)

	x_min = xarray[0]
	x_max = xarray[-1]
	x_mid = np.linspace(xarray[0], xarray[-1], num = 1000);
	roots = set()
	for val in x_mid:
		root, infodict, ier, mesg = optimize.fsolve(pdiff, val, full_output = True)
		if ier == 1 and x_min < root < x_max:
			roots.add(round(root[0], 5))
	return roots

def getscanLine(img):
	i = img.shape[0] / 2
	return i
	
def compensation(img, white):
	img = sk.img_as_float(img)
	white = sk.img_as_float(white)
	img /= white
	img = ((img - img.min()) / (img.max() - img.min())) * 255 
	return img.astype('uint8')
	
def getH(img, whiteList, angel):
	img = compensation(img, whiteList[:, :, 0])
	img = sk.img_as_ubyte(rotate(img, angel))
	scanLine = int(getscanLine(img))
	skio.imsave("out.png", img)
	line = img[scanLine,]
	
	black = np.arange(128)
	numOfBlack, temp = np.histogram(line, bins = range(0, 129))
	B = (numOfBlack * black).sum()/numOfBlack.sum()
	print(B)

	white = np.arange(128, 256)
	numOfWhite, temp = np.histogram(line, bins = range(128, 257))
	W = (numOfWhite * white).sum()/numOfWhite.sum()
	print(W)

	G = (W - B) / 2 + B

	print(int(G))

	gray = np.zeros(img.shape[1]) + G

	#print(intersectionPoints(range(0, img.shape[1]), line, gray))
	
	coordinates = list(intersectionPoints(range(0, img.shape[1]), line, gray))
	
	#coordinates = np.argwhere(np.diff(np.sign(line - gray)) != 0).reshape(-1) + 0
	#print(coordinates)
	coordinates.sort()
	print(coordinates)
	
	if (len(coordinates) % 2 != 0):
		coordinates = coordinates[0 : - 1]
	
	s = abs(coordinates[0] - coordinates[-1])/ (len(coordinates) - 1)
	print(s)

	print("\n-----------------------------------------\n")

	plt.plot(range(0, img.shape[1]), line, gray)
	plt.show()
	return s

def getFocus(h1, h2, H, R1, R2):
	l = (h2 * (H + h1) * R2 - h1 * (H + h2) * R1)/(h2 * (H + h1) - h1 * (H + h2))
	nu = (h2 * (R2 - l))/(H + h2)
	return (nu * H)/(H + h1), (nu * H)/(H + h2)

img1 = skio.imread('board_r.png')
whiteList = skio.imread('white_r.png')
img_r = img1[:, :, 0]
h_r = getH(img_r, whiteList, -1.3)

img2 = skio.imread('board_g.png')
whiteList = skio.imread('white_g.png')
img_g = img2[:, :, 0]
h_g = getH(img_g, whiteList, 1.6)

m = 0.0053
h_r *= m
h_g *= m 
H = 5
R1 = 320
R2 = 306
F1, F2 = getFocus(h_r, h_g, H, R1, R2)
print(F1, F2)
