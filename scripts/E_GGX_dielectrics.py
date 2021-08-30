import matplotlib.pyplot as plt
import numpy as np

from itertools import product
from multiprocessing import cpu_count, Pool
from numba import carray, cfunc, jit, types
from scipy import LowLevelCallable
from scipy.integrate import dblquad
from tqdm import tqdm


def reflectivity_to_eta(reflectivity):
  reflectivity = np.clip(reflectivity, 0, 0.98)
  return (1 + np.sqrt(reflectivity)) / (1 - np.sqrt(reflectivity))


@jit(nopython=True)
def fresnel_dielectric(eta, normal, outgoing):
  cosw = np.abs(np.dot(normal, outgoing))
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
def eval_refractive(ior, roughness, mu_out, mu_in, phi):
  if mu_in * mu_out == 0: return 0
  outgoing = np.asarray([np.sqrt(1 - mu_out * mu_out),
                         0,
                         mu_out])
  incoming = np.asarray([np.sqrt(1 - mu_in * mu_in) * np.cos(phi),
                         np.sqrt(1 - mu_in * mu_in) * np.sin(phi),
                         mu_in])
  if mu_in * mu_out > 0:
    halfway = normalize(incoming + outgoing)
    F = fresnel_dielectric(ior, halfway, outgoing)
    D = microfacet_distribution(roughness, halfway)
    G = microfacet_shadowing(roughness, halfway, outgoing, incoming)
    return F * D * G / np.abs(4 * mu_out * mu_in) * np.abs(mu_in)
  else:
    halfway = -normalize(ior * incoming + outgoing) * (-1 if ior < 1 else 1)
    F = fresnel_dielectric(ior, halfway, outgoing)
    D = microfacet_distribution(roughness, halfway)
    G = microfacet_shadowing(roughness, halfway, outgoing, incoming)
    # [Walter 2007] equation 21
    return np.abs((np.dot(outgoing, halfway) * np.dot(incoming, halfway)) / (mu_out * mu_in)) * \
           (1 - F) * D * G / np.square(ior * np.dot(halfway, incoming) + np.dot(halfway, outgoing)) * np.abs(mu_in)


@cfunc(types.double(types.intc, types.CPointer(types.double)))
def integrand(argc, argv):
  mu_in, phi, ior, alpha, mu_out = carray(argv, argc)
  return eval_refractive(ior, alpha, mu_out, mu_in, phi)


if __name__ == '__main__':
  WIDTH = 16
  HEIGHT = 16
  DEPTH = 16

  cos_theta_eps = 0.02
  roughness_eps = 0.035
  ior_eps = 0.02

  xrange = np.linspace(0, 1, WIDTH)
  xrange = np.where(xrange < cos_theta_eps, cos_theta_eps, xrange)

  yrange = np.linspace(0, 1, HEIGHT)
  yrange = np.where(yrange < roughness_eps, roughness_eps, yrange)
  yrange = np.square(yrange)

  zrange = np.linspace(0, 1, HEIGHT)
  zrange = np.where(zrange < ior_eps, ior_eps, zrange)
  zrange = reflectivity_to_eta(zrange)

  integrand = LowLevelCallable(integrand.ctypes)
  
  # Reflection
  print('===== Reflection =====')

  def integrate(args):
    return dblquad(integrand, 0, 2 * np.pi, lambda x: 0, lambda x: 1, args=args)

  with Pool(cpu_count()) as pool:
    results = list(tqdm(pool.imap(integrate, product(zrange, yrange, xrange)), total=WIDTH * HEIGHT * DEPTH))

  albedo_r, errors_r = zip(*results)

  # Transmission
  print('===== Transmission =====')

  def integrate(args):
    return dblquad(integrand, 0, 2 * np.pi, lambda x: -1, lambda x: 0, args=args)

  with Pool(cpu_count()) as pool:
    results = list(tqdm(pool.imap(integrate, product(zrange, yrange, xrange)), total=WIDTH * HEIGHT * DEPTH))

  albedo_t, errors_t = zip(*results)

  img_r = np.asarray(albedo_r, dtype=np.float32).reshape(DEPTH, HEIGHT, WIDTH)
  img_t = np.asarray(albedo_t, dtype=np.float32).reshape(DEPTH, HEIGHT, WIDTH)
  img = img_r + img_t

  for im, title in zip([img_r, img_t, img], [' (Reflection)', ' (Transmission)', '']):
    plt.figure()
    plt.suptitle('Directional Albedo' + title)

    for i in range(DEPTH):
      plt.subplot(4, 4, i + 1)
      plt.imshow(im[i * DEPTH // 16], extent=[0, 1, 1, 0], cmap=plt.get_cmap('gray'), interpolation=None)
      plt.colorbar()

      plt.title(f'Reflectivity = {i / 15:.2f}')
      plt.xlabel('cos(theta)')
      plt.ylabel('roughness')

    plt.show()

  np.savetxt('dielectrics_reflection.csv', img_r.reshape(-1, WIDTH), fmt='%a', delimiter=',')
  np.savetxt('dielectrics_transmission.csv', img_t.reshape(-1, WIDTH), fmt='%a', delimiter=',')
  np.savetxt('dielectrics.csv', img.reshape(-1, WIDTH), fmt='%a', delimiter=',')
