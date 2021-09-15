import matplotlib.pyplot as plt
import numpy as np

from argparse import ArgumentParser
from itertools import product
from numpy.core.fromnumeric import reshape
from scipy.optimize import curve_fit
from sklearn.preprocessing import PolynomialFeatures


def ratpoly(xdata, *coef):
  n = len(coef) // 2 + 1
  cnum, cdenom = coef[:n], coef[n:]
  return (cnum @ xdata) / (1 + cdenom @ xdata[1:])


if __name__ == '__main__':
  parser = ArgumentParser()
  parser.add_argument('table', help='The look-up table to fit')
  parser.add_argument('--deg', help='The degree of the polynomial to use', type=int, default=3)
  parser.add_argument('--rational', help='Either to use rational functions of not', action='store_true')

  args = parser.parse_args()

  Z = np.genfromtxt(args.table, delimiter=',', dtype=np.float32)
  
  size = Z.shape[-1]
  Z = Z.reshape(size, size, size)

  x = np.linspace(0, 1, size)
  X = np.asarray(list(product(x, x)))

  poly = PolynomialFeatures(args.deg).fit_transform(X)
  coef = []
  results = []

  for slice in range(size):
    if not args.rational:
      popt, *_ = np.linalg.lstsq(poly, Z[slice].ravel(), rcond=None)
      coef.append(popt)
      results.append(poly @ popt)
    else:
      p0 = np.ones(poly.shape[-1] * 2 - 1)
      # sigma = 0.1 * np.ones_like(Z[slice])
      # sigma[1:-1, :] = 1
      # popt, _ = curve_fit(ratpoly, poly.T, Z[slice].ravel(), p0, sigma=sigma.ravel(), method='trf')
      popt, _ = curve_fit(ratpoly, poly.T, Z[slice].ravel(), p0, method='trf')
      coef.append(popt)
      results.append(ratpoly(poly.T, *popt))

  coef = np.concatenate(coef, dtype=np.float32)
  results = np.stack([res.reshape(size, size) for res in results])

  tables = [Z, results]
  titles = ['Original', 'Fit']

  for table, title in zip(tables, titles):
    plt.figure()
    plt.suptitle(f'Directional Albedo ({title})')

    for i in range(16):
      plt.subplot(4, 4, i + 1)
      plt.imshow(table[i * size // 16], extent=[0, 1, 1, 0], cmap=plt.get_cmap('gray'), interpolation=None)
      plt.colorbar()

      plt.title(f'Reflectivity = {i / 15:.2f}')
      plt.xlabel('cos(theta)')
      plt.ylabel('roughness')

    plt.show()

  print('Optimal coefficients:')
  print(coef)

  residuals = np.abs(Z - results)

  print('Mean absolute error:   ', np.mean(residuals))
  print('Minimum absolute error:', np.min(residuals))
  print('Maximum absolute error:', np.max(residuals))
