import matplotlib.pyplot as plt
import numpy as np

from argparse import ArgumentParser
from itertools import product
from multiprocessing import cpu_count, Pool
from numba import carray, cfunc, jit, types
from scipy import LowLevelCallable
from scipy.integrate import dblquad
from tqdm import tqdm


SPECULAR_ALBEDO = np.genfromtxt('resources/tables/hi-res/conductors.csv', delimiter=',', dtype=np.float32)


def reflectivity_to_eta(reflectivity):
  reflectivity = np.clip(reflectivity, 0, 0.99)
  return (1 + np.sqrt(reflectivity)) / (1 - np.sqrt(reflectivity))


@jit(nopython=True)
def dot(v, w):
  return v[0] * w[0] + v[1] * w[1] + v[2] * w[2]


@jit(nopython=True)
def fresnel_dielectric(eta, normal, outgoing):
  cosw = abs(dot(normal, outgoing))
  sin2 = 1 - cosw * cosw
  eta2 = eta * eta
  cos2t = 1 - sin2 / eta2
  if cos2t < 0: return 1  # tir
  t0 = np.sqrt(cos2t)
  t1 = eta * t0
  t2 = eta * cosw
  rs = (cosw - t1) / (cosw + t1)
  rp = (t0 - t2) / (t0 + t2)
  return (rs * rs + rp * rp) / 2


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
  cosineh = dot(halfway, direction)
  if cosine * cosineh <= 0: return 0
  roughness2 = roughness * roughness
  cosine2 = cosine * cosine
  return 2 * abs(cosine) / (abs(cosine) + np.sqrt(cosine2 - roughness2 * cosine2 + roughness2))


@jit(nopython=True)
def microfacet_shadowing(roughness, halfway, outgoing, incoming):
  return microfacet_shadowing1(roughness, halfway, outgoing) * microfacet_shadowing1(roughness, halfway, incoming)


@jit(nopython=True)
def microfacet_compensation(roughness, mu_out):
  leny, lenx = SPECULAR_ALBEDO.shape

  s = np.sqrt(roughness) * (leny - 1)
  t = mu_out * (lenx - 1)

  i, j = int(s), int(t)
  ii = min(i + 1, leny - 1)
  jj = min(j + 1, lenx - 1)
  u, v = s - i, t - j

  E = SPECULAR_ALBEDO[i, j] * (1 - u) * (1 - v) + \
      SPECULAR_ALBEDO[i, jj] * (1 - u) * v + \
      SPECULAR_ALBEDO[ii, j] * u * (1 - v) + \
      SPECULAR_ALBEDO[ii, jj] * u * v

  return 1 / E


@jit(nopython=True)
def halfway_vector(v, w):
  s = (v[0] + w[0], v[1] + w[1], v[2] + w[2])
  l = np.sqrt(s[0] * s[0] + s[1] * s[1] + s[2] * s[2])
  return (s[0] / l, s[1] / l, s[2] / l) if l != 0 else s


@jit(nopython=True)
def eval_specular(ior, roughness, mu_out, mu_in, phi):
  if mu_in * mu_out <= 0: return 0
  outgoing = (np.sqrt(1 - mu_out * mu_out), 0, mu_out)
  incoming = (np.sqrt(1 - mu_in * mu_in) * np.cos(phi),
              np.sqrt(1 - mu_in * mu_in) * np.sin(phi),
              mu_in)
  halfway = halfway_vector(incoming, outgoing)
  C = microfacet_compensation(roughness, mu_out)
  F = fresnel_dielectric(ior, halfway, outgoing)
  D = microfacet_distribution(roughness, halfway)
  G = microfacet_shadowing(roughness, halfway, outgoing, incoming)
  return C * F * D * G / (4 * mu_out * mu_in) * abs(mu_in)


@cfunc(types.double(types.intc, types.CPointer(types.double)))
def integrand(argc, argv):
  mu_in, phi, ior, alpha, mu_out = carray(argv, argc)
  return eval_specular(ior, alpha, mu_out, mu_in, phi)


if __name__ == '__main__':
  parser = ArgumentParser()
  parser.add_argument('--size', help='Look-up table size', type=int, default=32)
  parser.add_argument('--output', help='Output filename', default='glossy.csv')

  args = parser.parse_args()

  cos_theta_eps = 0.02
  roughness_eps = 0.035

  xrange = np.linspace(0, 1, args.size)
  xrange = np.where(xrange < cos_theta_eps, cos_theta_eps, xrange)

  yrange = np.linspace(0, 1, args.size)
  yrange = np.where(yrange < roughness_eps, roughness_eps, yrange)
  yrange = np.square(yrange)

  zrange = np.linspace(0, 1, args.size)
  zrange = reflectivity_to_eta(zrange)

  integrand = LowLevelCallable(integrand.ctypes)

  def integrate(args):
    return dblquad(integrand, 0, 2 * np.pi, lambda x: 0, lambda x: 1, args=args)

  with Pool(cpu_count()) as pool:
    results = list(tqdm(pool.imap(integrate, product(zrange, yrange, xrange)), total=args.size * args.size * args.size))

  albedo, errors = zip(*results)
  table = np.asarray(albedo, dtype=np.float32).reshape(args.size, args.size, args.size)

  print('Mean absolute error:', np.mean(errors))
  print('Maximum absolute error:', np.max(errors))

  np.savetxt(args.output, table.reshape(-1, args.size), fmt='%a', delimiter=',')

  plt.figure()
  plt.suptitle('Directional Albedo')

  for i in range(16):
    plt.subplot(4, 4, i + 1)
    plt.imshow(table[i * args.size // 16], extent=[0, 1, 1, 0], cmap=plt.get_cmap('gray'), interpolation=None)
    plt.colorbar()

    plt.title(f'Reflectivity = {i / 15:.2f}')
    plt.xlabel('cos(theta)')
    plt.ylabel('roughness')

  plt.show()
