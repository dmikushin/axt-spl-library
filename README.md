# gpu_spmv

This project contains the code to perform the SpMV product on NVIDIA's GPUs.

CSR format.
	- Naive version
	- CUSPARSE function.

AXC format.
	- New version.

K1 format.
	- Original version from paper.
	- Improved version (reordering integrated)

AXT format.
	- Uncompacted with tileHeight = 1.
	- Uncompacted with tileHeight > 1. 
	- Compacted with tileHeight = 1 (not completed)


The AXT software is part of the AXT_SPL library registered under a GPL license. The Universidad de Santiago de Compostela partially owns the intellectual rights.
