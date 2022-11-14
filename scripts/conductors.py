import matplotlib.pyplot as plt

from argparse import ArgumentParser
from itertools import product
from multiprocessing import cpu_count, Pool
from numba import carray, cfunc, types
from scipy import LowLevelCallable
from scipy.integrate import dblquad
from tqdm import tqdm

from ggx import *


@jit(nopython=True)
def eval_metallic(roughness, mu_out, mu_in, phi):
  if mu_in * mu_out <= 0: return 0
  outgoing = (math.sqrt(1 - mu_out * mu_out), 0, mu_out)
  incoming = (math.sqrt(1 - mu_in * mu_in) * math.cos(phi),
              math.sqrt(1 - mu_in * mu_in) * math.sin(phi),
              mu_in)
  halfway = halfway_vector(incoming, outgoing)
  D = microfacet_distribution(roughness, halfway)
  G = microfacet_shadowing(roughness, halfway, outgoing, incoming)
  return D * G / (4 * mu_out * mu_in) * abs(mu_in)


@cfunc(types.double(types.intc, types.CPointer(types.double)))
def integrand(argc, argv):
  mu_in, phi, alpha, mu_out = carray(argv, argc)
  return eval_metallic(alpha, mu_out, mu_in, phi)


if __name__ == '__main__':
  parser = ArgumentParser()
  parser.add_argument('--size', help='Look-up table size', type=int, default=32)
  parser.add_argument('--output', help='Output filename', default='conductors.csv')

  args = parser.parse_args()

  cos_theta_eps = 0.02
  roughness_eps = 0.035

  xrange = np.linspace(0, 1, args.size)
  xrange = np.where(xrange < cos_theta_eps, cos_theta_eps, xrange)

  yrange = np.linspace(0, 1, args.size)
  yrange = np.where(yrange < roughness_eps, roughness_eps, yrange)
  yrange = np.square(yrange)

  integrand = LowLevelCallable(integrand.ctypes)

  def integrate(args):
    return dblquad(integrand, 0, 2 * math.pi, lambda x: 0, lambda x: 1, args=args)

  with Pool(cpu_count()) as pool:
    results = list(tqdm(pool.imap(integrate, product(yrange, xrange)), total=args.size * args.size))

  albedo, errors = zip(*results)
  table = np.asarray(albedo, dtype=np.float32).reshape(args.size, args.size)

  print('Mean absolute error:', np.mean(errors))
  print('Maximum absolute error:', np.max(errors))

  np.savetxt(args.output, table, fmt='%a', delimiter=',')

  plt.figure()
  plt.imshow(table, extent=[0, 1, 1, 0], cmap=plt.get_cmap('gray'), interpolation=None)
  plt.colorbar()

  plt.title('Directional Albedo')
  plt.xlabel('cos(theta)')
  plt.ylabel('roughness')

  plt.show()
