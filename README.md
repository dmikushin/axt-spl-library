# AXT SPL Library

This project contains the code to perform the SpMV product on NVIDIA's GPUs:

* AXT format.
        - Uncompacted with tileHeight = 1.
        - Uncompacted with tileHeight > 1.
        - Compacted with tileHeight = 1 (not completed)

* AXC format.
	- New version.

* K1 format.
	- Original version from paper.
	- Improved version (reordering integrated)

* CSR format.
        - Naive version
        - CUSPARSE function.

## Citation

```
@article{S2021102997,
title = {A new AXT format for an efficient SpMV product using AVX-512 instructions and CUDA},
journal = {Advances in Engineering Software},
volume = {156},
pages = {102997},
year = {2021},
issn = {0965-9978},
doi = {https://doi.org/10.1016/j.advengsoft.2021.102997},
url = {https://www.sciencedirect.com/science/article/pii/S0965997821000260},
author = {E. Coronado-Barrientos and M. Antonioletti and A. Garcia-Loureiro},
keywords = {Sparse Matrix Vector product, AVX-512 instructions, MKL Library, CUDA, cuSPARSE Library, Segmented Scan algorithm},
}
```

## Licence

The AXT format's software is part of the AXT SPL library registered under a GPL license. The Universidad de Santiago de Compostela partially owns the
intellectual rights.

