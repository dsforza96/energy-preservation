import math
import numpy as np

from numba import jit


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
  t0 = math.sqrt(cos2t)
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
  return roughness2 / (math.pi * (cosine2 * roughness2 + 1 - cosine2) * (cosine2 * roughness2 + 1 - cosine2))


@jit(nopython=True)
def microfacet_shadowing1(roughness, halfway, direction):
  cosine = direction[-1]
  cosineh = dot(halfway, direction)
  if cosine * cosineh <= 0: return 0
  roughness2 = roughness * roughness
  cosine2 = cosine * cosine
  return 2 * abs(cosine) / (abs(cosine) + math.sqrt(cosine2 - roughness2 * cosine2 + roughness2))


@jit(nopython=True)
def microfacet_shadowing(roughness, halfway, outgoing, incoming):
  return microfacet_shadowing1(roughness, halfway, outgoing) * microfacet_shadowing1(roughness, halfway, incoming)


@jit(nopython=True)
def halfway_vector(v, w):
  s = (v[0] + w[0], v[1] + w[1], v[2] + w[2])
  l = math.sqrt(s[0] * s[0] + s[1] * s[1] + s[2] * s[2])
  return (s[0] / l, s[1] / l, s[2] / l) if l != 0 else s
