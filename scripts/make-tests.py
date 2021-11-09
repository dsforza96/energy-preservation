import os

from argparse import ArgumentParser
from itertools import product
from pathlib import Path
from PIL import Image
from shutil import copy


def make_furnace_json(material, roughness, ior):
  return f'''{{
  "asset": {{
    "copyright": "Model by Fabio Pellacini from github.com/~xelatihy/yocto-gl"
  }},
  "cameras": {{
    "default": {{
      "lens": 100,
      "aspect": 1,
      "lookat": [
        0,
        0.075,
        550,
        0,
        0.075,
        0,
        0,
        1,
        0
      ]
    }}
  }},
  "environments": {{
    "furnace": {{
      "emission": [
        0.5,
        0.5,
        0.5
      ]
    }}
  }},
  "instances": {{
    "sphere": {{
      "shape": "sphere",
      "material": "sphere"
    }}
  }},
  "materials": {{
    "sphere": {{
      "type": "{material}",
      "color": [
        1,
        1,
        1
      ],
      "roughness": {roughness},
      "ior": {ior}
    }}
  }}
}}'''


def make_checker_json(material, roughness, ior):
  return f'''{{
  "asset": {{
    "copyright": "Model by Fabio Pellacini from github.com/~xelatihy/yocto-gl"
  }},
  "cameras": {{
    "default": {{
      "lens": 100,
      "aspect": 1,
      "lookat": [
        0,
        0.075,
        550,
        0,
        0.075,
        0,
        0,
        1,
        0
      ]
    }}
  }},
  "environments": {{
    "furnace": {{
      "emission": [
        1,
        1,
        1
      ],
      "emission_tex": "quattro_canti_4k"
    }}
  }},
  "instances": {{
    "checker": {{
      "shape": "quad",
      "frame": [
        1,
        0,
        0,
        0,
        1,
        0,
        0,
        0,
        1,
        0,
        0.075,
        -0.15
      ],
      "material": "checker"
    }},
    "sphere": {{
      "shape": "sphere",
      "material": "sphere"
    }}
  }},
  "materials": {{
    "checker": {{
      "type": "matte",
      "color": [
        1,
        1,
        1
      ],
      "color_tex": "checker"
    }},
    "sphere": {{
      "type": "{material}",
      "color": [
        1,
        1,
        1
      ],
      "roughness": {roughness},
      "ior": {ior}
    }}
  }}
}}'''


def make_furnace_scene(path, material, roughness, ior=1.5):
  Path(f'{path}/shapes').mkdir(parents=True, exist_ok=True)
  copy('assets/shapes/sphere.ply', f'{path}/shapes')

  fname = f'{os.path.basename(path)}.json'
  with open(f'{path}/{fname}', 'w') as f:
    f.write(make_furnace_json(material, roughness, ior))


def make_checker_scene(path, material, roughness, ior=1.5):
  Path(f'{path}/shapes').mkdir(parents=True, exist_ok=True)
  copy('assets/shapes/sphere.ply', f'{path}/shapes')
  copy('assets/shapes/quad.ply', f'{path}/shapes')

  Path(f'{path}/textures').mkdir(parents=True, exist_ok=True)
  copy('assets/textures/quattro_canti_4k.hdr', f'{path}/textures')
  copy('assets/textures/checker.png', f'{path}/textures')

  fname = f'{os.path.basename(path)}.json'
  with open(f'{path}/{fname}', 'w') as f:
    f.write(make_checker_json(material, roughness, ior))


if __name__ == '__main__':
  parser = ArgumentParser()
  parser.add_argument('command', help='The action to execute', default='all')
  parser.add_argument('--nspheres', help='Number of spheres in each row', type=int, default=7)
  parser.add_argument('--outpath', help='The output path', default='.')

  args = parser.parse_args()

  if args.command in ['make', 'all']:
    comp_types = ['', '_comp_mytab', '_comp_myfit']
    rogh_values = [i / (args.nspheres - 1) for i in range(args.nspheres)]

    for i, (comp, rough) in enumerate(product(comp_types, rogh_values)):
      path = f'{args.outpath}/tests/metallic_furnace/{i}'
      make_furnace_scene(path, f'metallic{comp}', rough)

      path = f'{args.outpath}/tests/metallic_checker/{i}'
      make_checker_scene(path, f'metallic{comp}', rough)

    materials = ['transparent', 'refractive', 'glossy']
    comp_types = ['', '_comp', '_comp_fit']
    ior_values = [1.25, 1.5, 2, 2.5]

    for material, ior in product(materials, ior_values):
      for i, (comp, rough) in enumerate(product(comp_types, rogh_values)):
        path = f'{args.outpath}/tests/{material}_ior{ior}_furnace/{i}'
        make_furnace_scene(path, f'{material}{comp}', rough, ior)

        path = f'{args.outpath}/tests/{material}_ior{ior}_checker/{i}'
        make_checker_scene(path, f'{material}{comp}', rough, ior)

  if args.command in ['run', 'all']:
    wd = os.getcwd()

    for root, dirs, files in os.walk('tests'):
      for fname in [f for f in files if f.endswith('.json')]:
        head, tail = os.path.split(root)
        path = os.path.basename(head)

        Path(f'tmp/{path}').mkdir(parents=True, exist_ok=True)
        os.system(f'../yocto-gl/bin/yscene render "{wd}/{root}/{fname}" --output "{wd}/tmp/{path}/{tail}.png" \
                    --sampler naive --resolution 256 --samples 1024 --clamp 10000 --bounces 64')

  if args.command in ['compose', 'all']:
    for root, dirs, files in os.walk('tmp'):
      if not files: continue

      fnames = sorted(files, key=lambda f: int(os.path.splitext(f)[0]))
      images = [Image.open(f'{root}/{f}') for f in fnames]
      size = images[0].size[0]

      res = Image.new('RGB', (size * len(fnames) // 3, size * 3))

      for img, (j, i) in zip(images, product(range(3), range(len(fnames) // 3))):
        res.paste(img, (i * size, j * size))

      Path(f'out').mkdir(parents=True, exist_ok=True)
      res.save(f'out/{os.path.basename(root)}.png')
