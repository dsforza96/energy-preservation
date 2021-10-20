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
  parser.add_argument('table', help='The look-up table to fit')
  parser.add_argument('--deg', help='The degree of the polynomial to use', type=int, default=3)
  parser.add_argument('--rational', help='Either to use rational functions of not', action='store_true')
  parser.add_argument('--reduce-range', help='Fit the look-up table in the ior range [1.25, 3]', action='store_true')

  args = parser.parse_args()

  Z = np.genfromtxt(args.table, delimiter=',', dtype=np.float32)
  
  size = Z.shape[-1]
  x = y = np.linspace(0, 1, size)

  if Z.shape[0] == size:
    X = np.asarray(list(product(y, x)))
  else:
    Z = Z.reshape(size, size, size)
    w = np.linspace(0.0125, 0.25, size) if args.reduce_range else x
    X = np.asarray(list(product(x, y, x)))

  poly = PolynomialFeatures(args.deg).fit_transform(X)

  if not args.rational:
    popt, *_ = np.linalg.lstsq(poly, Z.ravel(), rcond=None)
    results = poly @ popt
  else:
    p0 = np.ones(poly.shape[-1] * 2 - 1)
    # sigma = 0.1 * np.ones_like(Z)
    # sigma[2:, ...] = 1
    # popt, _ = curve_fit(ratpoly, poly.T, Z.ravel(), p0, sigma=sigma.ravel(), method='trf', maxfev=10000)
    popt, _ = curve_fit(ratpoly, poly.T, Z.ravel(), p0, method='trf', maxfev=10000)
    results = ratpoly(poly.T, *popt)

  if Z.ndim == 2:
    plt.figure()
    plt.suptitle('Directional Albedo')

    plt.subplot(1, 2, 1)
    plt.imshow(Z, extent=[0, 1, 1, 0], cmap=plt.get_cmap('gray'), interpolation=None)
    plt.colorbar()

    plt.title('Original')
    plt.xlabel('cos(theta)')
    plt.ylabel('roughness')

    plt.subplot(1, 2, 2)
    plt.imshow(results.reshape(size, size), extent=[0, 1, 1, 0], cmap=plt.get_cmap('gray'), interpolation=None)
    plt.colorbar()

    plt.title('Fit')
    plt.xlabel('cos(theta)')
    plt.ylabel('roughness')

    plt.show()
  else:
    tables = [Z, results.reshape(size, size, size)]
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
  print(popt.astype(np.float32))

  residuals = np.abs(Z.ravel() - results)

  print('Mean absolute error:   ', np.mean(residuals))
  print('Minimum absolute error:', np.min(residuals))
  print('Maximum absolute error:', np.max(residuals))
