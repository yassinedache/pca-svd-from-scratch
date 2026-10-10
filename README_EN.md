# PCA and singular value decomposition (SVD), from scratch

![tests](https://github.com/yassinedache/pca-svd-from-scratch/actions/workflows/tests.yml/badge.svg)

*Version française et détails : [README.md](README.md) · Full report (in French): [report/rapport.pdf](report/rapport.pdf)*

Personal project in Python (NumPy, Matplotlib). Two Jacobi rotation algorithms are written from scratch, **without `numpy.linalg.eig` or `numpy.linalg.svd`**: cyclic Jacobi for symmetric eigenproblems, and one-sided Jacobi for the SVD (vectorised, about 17 s for a 512 x 512 image). NumPy is only used to check the results.

## Key results

- **PCA of the Wine dataset** (178 wines, 3 cultivars, 13 standardised variables), computed both by diagonalising the covariance matrix and by the SVD of the data: the two routes agree to $3 \times 10^{-15}$. PC1 and PC2 explain 55.4 % of the variance, and without using the labels the first plane separates the cultivars (97.2 % leave-one-out nearest-centroid accuracy).
- **Why the SVD route is safer**: forming $X^\top X$ squares the condition number; on the Läuchli matrix, the covariance route loses the smallest singular value entirely for $\epsilon = 10^{-8}$ while the SVD stays exact.
- **Image compression and Eckart–Young**: the measured error of the rank-$k$ approximation matches $\big(\sum_{i>k}\sigma_i^2\big)^{1/2}$ to $3.5 \times 10^{-15}$, and $\lVert A - A_k\rVert_2 = \sigma_{k+1}$. Rank 50 keeps 19.6 % of the numbers for a 27.6 dB PSNR.

![Compression](figures/5_compression.png)

## Usage

```bash
pip install -r requirements.txt
python experiments/run_all.py   # about a minute
pytest                          # 16 tests
```

Data: Wine (Aeberhard & Forina, UCI, CC BY 4.0); photo of astronaut Eileen Collins (NASA, public domain, via scikit-image).
Author: **Yassine Abderrezak Dache**, BSc in Computer Science (USTHB). MIT licence.
