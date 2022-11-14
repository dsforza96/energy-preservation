import matplotlib.pyplot as plt

from argparse import ArgumentParser
from itertools import product
from multiprocessing import cpu_count, Pool
from numba import carray, cfunc, types
from os.path import splitext
from scipy import LowLevelCallable
from scipy.integrate import nquad
from tqdm import tqdm

from ggx import *


@jit(nopython=True)
def eval_refractive(ior, roughness, mu_out, mu_in, phi):
  if mu_in * mu_out == 0: return 0
  outgoing = (math.sqrt(1 - mu_out * mu_out), 0, mu_out)
  incoming = (math.sqrt(1 - mu_in * mu_in) * math.cos(phi),
              math.sqrt(1 - mu_in * mu_in) * math.sin(phi),
              mu_in)
  if mu_in * mu_out > 0:
    halfway = halfway_vector(incoming, outgoing)
    F = fresnel_dielectric(ior, halfway, outgoing)
    D = microfacet_distribution(roughness, halfway)
    G = microfacet_shadowing(roughness, halfway, outgoing, incoming)
    return F * D * G / abs(4 * mu_out * mu_in) * abs(mu_in)
  else:
    halfway_ = halfway_vector((incoming[0] * ior, incoming[1] * ior, incoming[2] * ior), outgoing)
    halfway = halfway_ if ior < 1 else (-halfway_[0], -halfway_[1], -halfway_[2])
    F = fresnel_dielectric(ior, halfway, outgoing)
    D = microfacet_distribution(roughness, halfway)
    G = microfacet_shadowing(roughness, halfway, outgoing, incoming)
    # [Walter 2007] equation 21
    return abs((dot(outgoing, halfway) * dot(incoming, halfway)) / (mu_out * mu_in)) * \
           (1 - F) * D * G / (ior * dot(halfway, incoming) + dot(halfway, outgoing)) ** 2 * abs(mu_in)


@cfunc(types.double(types.intc, types.CPointer(types.double)))
def integrand(argc, argv):
  mu_in, phi, ior, alpha, mu_out = carray(argv, argc)
  return eval_refractive(ior, alpha, mu_out, mu_in, phi)


if __name__ == '__main__':
  parser = ArgumentParser()
  parser.add_argument('--size', help='Look-up table size', type=int, default=32)
  parser.add_argument('--output', help='Output filename', default='dielectrics.csv')

  args = parser.parse_args()

  cos_theta_eps = 0.02
  roughness_eps = 0.035
  reflectivity_range = (0.0125, 0.25)

  xrange = np.linspace(0, 1, args.size)
  xrange = np.where(xrange < cos_theta_eps, cos_theta_eps, xrange)
  xrange = np.where(xrange > 1 - cos_theta_eps, 1 - cos_theta_eps, xrange)

  yrange = np.linspace(0, 1, args.size)
  yrange = np.where(yrange < roughness_eps, roughness_eps, yrange)
  yrange = np.square(yrange)

  zrange = np.linspace(*reflectivity_range, args.size)
  zrange = reflectivity_to_eta(zrange)

  integrand = LowLevelCallable(integrand.ctypes)

  # Entering medium
  print('Entering medium, reflection...')

  def integrate(args):
    return nquad(integrand, ((0, 1), (0, 2 * math.pi)), args=args, opts={'limit': 200})

  with Pool(cpu_count()) as pool:
    results = list(tqdm(pool.imap(integrate, product(zrange, yrange, xrange)), total=args.size * args.size * args.size))

  enter_albedo_r, enter_errors_r = zip(*results)

  print('Entering medium, transmission...')

  def integrate(args):
    return nquad(integrand, ((-1, 0), (0, 2 * math.pi)), args=args, opts={'limit': 200})

  with Pool(cpu_count()) as pool:
    results = list(tqdm(pool.imap(integrate, product(zrange, yrange, xrange)), total=args.size * args.size * args.size))

  enter_albedo_t, enter_errors_t = zip(*results)

  # Leaving medium
  zrange = 1 / zrange

  print('Leaving medium, reflection...')

  def integrate(args):
    return nquad(integrand, ((0, 1), (0, 2 * math.pi)), args=args, opts={'limit': 200})

  with Pool(cpu_count()) as pool:
    results = list(tqdm(pool.imap(integrate, product(zrange, yrange, xrange)), total=args.size * args.size * args.size))

  leave_albedo_r, leave_errors_r = zip(*results)

  print('Leaving medium, transmission...')

  def integrate(args):
    return nquad(integrand, ((-1, 0), (0, 2 * math.pi)), args=args, opts={'limit': 200})

  with Pool(cpu_count()) as pool:
    results = list(tqdm(pool.imap(integrate, product(zrange, yrange, xrange)), total=args.size * args.size * args.size))

  leave_albedo_t, leave_errors_t = zip(*results)

  # Save and show results
  enter_table_r = np.asarray(enter_albedo_r, dtype=np.float32).reshape(args.size, args.size, args.size)
  enter_table_t = np.asarray(enter_albedo_t, dtype=np.float32).reshape(args.size, args.size, args.size)
  enter_table = enter_table_r + enter_table_t

  leave_table_r = np.asarray(leave_albedo_r, dtype=np.float32).reshape(args.size, args.size, args.size)
  leave_table_t = np.asarray(leave_albedo_t, dtype=np.float32).reshape(args.size, args.size, args.size)
  leave_table = leave_table_r + leave_table_t

  errors = [enter_errors_r, enter_errors_t, leave_errors_r, leave_errors_t]
  names = ['entering medium, reflection', 'entering medium, transmission',
            'leaving medium, reflection', 'leaving medium, transmission']

  for errors, name in zip(errors, names):
    print(f'Mean absolute error ({name}):', np.mean(errors))
    print(f'Maximum absolute error ({name}):', np.max(errors))

  filename, ext = splitext(args.output)

  np.savetxt(f'{filename}_r{ext}', enter_table_r.reshape(-1, args.size), fmt='%a', delimiter=',')
  np.savetxt(f'{filename}_t{ext}', enter_table_t.reshape(-1, args.size), fmt='%a', delimiter=',')
  np.savetxt(args.output, enter_table.reshape(-1, args.size), fmt='%a', delimiter=',')

  np.savetxt(f'{filename}_inv_eta_r{ext}', leave_table_r.reshape(-1, args.size), fmt='%a', delimiter=',')
  np.savetxt(f'{filename}_inv_eta_t{ext}', leave_table_t.reshape(-1, args.size), fmt='%a', delimiter=',')
  np.savetxt(f'{filename}_inv_eta{ext}', leave_table.reshape(-1, args.size), fmt='%a', delimiter=',')

  tables = [enter_table_r, enter_table_t, enter_table, leave_table_r, leave_table_t, leave_table]
  titles = ['Entering Medium, Reflection', 'Entering Medium, Transmission', 'Entering Medium',
            'Leaving Medium, Reflection', 'Leaving Medium, Transmission', 'Leaving Medium']

  for table, title in zip(tables, titles):
    plt.figure()
    plt.suptitle(f'Directional Albedo ({title})')

    for i in range(16):
      plt.subplot(4, 4, i + 1)
      plt.imshow(table[i * args.size // 16], extent=[0, 1, 1, 0], cmap=plt.get_cmap('gray'), interpolation=None)
      plt.colorbar()

      plt.title(f'Reflectivity = {i / 15:.2f}')
      plt.xlabel('cos(theta)')
      plt.ylabel('roughness')

    plt.show()
