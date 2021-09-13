import matplotlib.pyplot as plt
import numpy as np

from argparse import ArgumentParser
from itertools import product
from scipy.optimize import curve_fit
from sklearn.preprocessing import PolynomialFeatures


def ratpoly(xdata, *coef):
  n = len(coef) // 2 + 1
  cnum, cdenom = coef[:n], coef[n:]
  return (cnum @ xdata) / (1 + cdenom @ xdata[1:])


if __name__ == '__main__':
  parser = ArgumentParser()
  parser.add_argument('--input', help='The look-up table to fit', default='resources/conductors.csv')
  parser.add_argument('--deg', help='The degree of the polynomial', type=int, default=3)
  parser.add_argument('--rational', help='Either to use rational functions of not', action='store_true')

  args = parser.parse_args()

  Z = np.genfromtxt(args.input, delimiter=',', dtype=np.float32)
  
  size = Z.shape[-1]
  x = np.linspace(0, 1, size)

  if Z.shape[0] == size:
    X = np.asarray(list(product(x, x)))
  else:
    Z = Z.reshape(size, size, size)
    X = np.asarray(list(product(x, x, x)))

  poly = PolynomialFeatures(args.deg).fit_transform(X)

  if not args.rational:
    popt, *_ = np.linalg.lstsq(poly, Z.ravel(), rcond=None)
    res = poly @ popt
  else:
    p0 = np.ones(poly.shape[-1] * 2 - 1)
    popt, _ = curve_fit(ratpoly, poly.T, Z.ravel(), p0, method='trf')
    res = ratpoly(poly.T, *popt)

  plt.figure()
  plt.suptitle('Directional Albedo')

  plt.subplot(1, 2, 1)
  plt.imshow(Z, extent=[0, 1, 1, 0], cmap=plt.get_cmap('gray'), interpolation=None)
  plt.colorbar()

  plt.title('Original')
  plt.xlabel('cos(theta)')
  plt.ylabel('roughness')

  plt.subplot(1, 2, 2)
  plt.imshow(res.reshape(size, size), extent=[0, 1, 1, 0], cmap=plt.get_cmap('gray'), interpolation=None)
  plt.colorbar()

  plt.title('Fit')
  plt.xlabel('cos(theta)')
  plt.ylabel('roughness')

  plt.show()

  print('Optimal coefficients:')
  print(popt)
