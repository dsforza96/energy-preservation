# Enforcing Energy Preservation in Microfacet Models

Software implementation for the paper SFORZA, Davide; PELLACINI, Fabio.
Enforcing Energy Preservation in Microfacet Models. In: *Smart Tools and
Applications in Graphics - Eurographics Italian Chapter Conference*. 2022.

![Enforcing Energy Preservation in Microfacet Models](teaser.png)

Microfacet models tend to lose energy with the increase in surface roughness,
causing an undesired darkening of rough materials. Our method allows compensating
for the loss of energy without the use of precomputed look-up tables. Instead, we
propose a set of analytic approximations of the directional albedo of conductive
(*left*), glossy (*center*) and dielectric (*right*) materials.

Use the [`conductors.py`](scripts/conductors.py), [`glossy.py`](scripts/glossy.py)
and [`dielectrics.py`](scripts/dielectrics.py) scripts to generate the look-up
tables containing the values of the directional albedo of conductors, glossy materials
and dielectrics, respectively. You can then use [`fit.py`](scripts/fit.py)
to approximate them with a polynomial or a rational function of the desired degree.

## Citation
If you want to include this code in your work, please cite us as:

```latex
@inproceedings {10.2312:stag.20221258,
  booktitle = {Smart Tools and Applications in Graphics - Eurographics Italian Chapter Conference},
  editor = {Cabiddu, Daniela and Schneider, Teseo and Allegra, Dario and Catalano, Chiara Eva and Cherchi, Gianmarco and Scateni, Riccardo},
  title = {{Enforcing Energy Preservation in Microfacet Models}},
  author = {Sforza, Davide and Pellacini, Fabio},
  year = {2022},
  publisher = {The Eurographics Association},
  ISSN = {2617-4855},
  ISBN = {978-3-03868-191-5},
  DOI = {10.2312/stag.20221258}
}
```

## License

Our code is released under [MIT license](LICENSE).
