import matplotlib.pyplot as plt
import numpy as np

from argparse import ArgumentParser
from itertools import product
from multiprocessing import cpu_count, Pool
from numba import carray, cfunc, jit, types
from scipy import LowLevelCallable
from scipy.integrate import dblquad
from tqdm import tqdm


@jit(nopython=True)
def microfacet_distribution(roughness, halfway):
  cosine = halfway[-1]
  if cosine <= 0: return 0
  roughness2 = roughness * roughness
  cosine2 = cosine * cosine
  return roughness2 / (np.pi * (cosine2 * roughness2 + 1 - cosine2) * (cosine2 * roughness2 + 1 - cosine2))


@jit(nopython=True)
def microfacet_shadowing1(roughness, halfway, direction):
  cosine = direction[-1]
  cosineh = np.dot(halfway, direction)
  if cosine * cosineh <= 0: return 0
  roughness2 = roughness * roughness
  cosine2 = cosine * cosine
  return 2 * np.abs(cosine) / (np.abs(cosine) + np.sqrt(cosine2 - roughness2 * cosine2 + roughness2))


@jit(nopython=True)
def microfacet_shadowing(roughness, halfway, outgoing, incoming):
  return microfacet_shadowing1(roughness, halfway, outgoing) * microfacet_shadowing1(roughness, halfway, incoming)


@jit(nopython=True)
def normalize(v):
  l = np.linalg.norm(v)
  return v / l if l != 0 else v


@jit(nopython=True)
def eval_metallic(roughness, mu_out, mu_in, phi):
  if mu_in * mu_out <= 0: return 0
  outgoing = np.asarray([np.sqrt(1 - mu_out * mu_out),
                         0,
                         mu_out])
  incoming = np.asarray([np.sqrt(1 - mu_in * mu_in) * np.cos(phi),
                         np.sqrt(1 - mu_in * mu_in) * np.sin(phi),
                         mu_in])
  halfway = normalize(incoming + outgoing)
  D = microfacet_distribution(roughness, halfway)
  G = microfacet_shadowing(roughness, halfway, outgoing, incoming)
  return D * G / (4 * mu_out * mu_in) * np.abs(mu_in)


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
    return dblquad(integrand, 0, 2 * np.pi, lambda x: 0, lambda x: 1, args=args)

  with Pool(cpu_count()) as pool:
    results = list(tqdm(pool.imap(integrate, product(yrange, xrange)), total=args.size * args.size))

  albedo, errors = zip(*results)

  table = np.asarray(albedo, dtype=np.float32).reshape(args.size, args.size)

  plt.figure()
  plt.imshow(table, extent=[0, 1, 1, 0], cmap=plt.get_cmap('gray'), interpolation=None)
  plt.colorbar()

  plt.title('Directional Albedo')
  plt.xlabel('cos(theta)')
  plt.ylabel('roughness')

  plt.show()

  print('Mean absolute error:', np.mean(errors))
  print('Maximum absolute error:', np.max(errors))

  np.savetxt(args.output, table, fmt='%a', delimiter=',')
