# Every computed diagnostic, before vs after optimisation

`before` = epoch 0 (the initial approximation), `after` = the final analysed snapshot. Cells are `before → after`, averaged over the runs that have the measure. A cell marked `(k/n)` means only k of the n runs in that row produced a value — the persistent-homology columns write a sentinel instead of a number when the complex is too large, and that sentinel is excluded rather than averaged.

GI-NCF is `GINCF/patience_300`: lr_isomap=0.001, frozen item projection, 300 outer epochs. An earlier lr_isomap=0.05 hyperparameter-search tree exists on disk and is deliberately excluded from every aggregate. The non-parametric heads had five measures computed, not nineteen.

## Ollivier-Ricci curvature

| dataset | model | scale | runs | mean | median | fraction negative | graph edges |
|---|---|---|---|---|---|---|---|
| MovieLens-1M | GI-NCF (paper cfg) | 300x800 | 3 | -0.089 → -0.145 | -0.089 → -0.144 | 0.730 → 0.838 | 6701 → 5682 |
| MovieLens-1M | GI-NCF (paper cfg) | 1000x1200 | 3 | -0.083 → -0.157 | -0.064 → -0.158 | 0.658 → 0.816 | 11002 → 8950 |
| MovieLens-1M | GI-NCF (paper cfg) | 3000x2500 | 3 | -0.164 → -0.227 | -0.082 → -0.171 | 0.644 → 0.748 | 24562 → 21864 |
| MovieLens-1M | GI-SASRec (raw init) | 300x800 | 3 | -0.089 → -0.131 | -0.089 → -0.127 | 0.730 → 0.807 | 6704 → 5867 |
| MovieLens-1M | GI-SASRec (raw init) | 1000x1200 | 3 | -0.083 → -0.124 | -0.065 → -0.117 | 0.656 → 0.757 | 11021 → 9631 |
| MovieLens-1M | GI-SASRec (raw init) | 3000x2500 | 3 | -0.163 → -0.210 | -0.083 → -0.135 | 0.644 → 0.701 | 24564 → 22896 |
| MovieLens-1M | GI-SASRec (IDF init) | 300x800 | 3 | -0.092 → -0.117 | -0.092 → -0.114 | 0.740 → 0.788 | 6704 → 6144 |
| MovieLens-1M | GI-SASRec (IDF init) | 1000x1200 | 3 | -0.083 → -0.110 | -0.061 → -0.098 | 0.647 → 0.724 | 11139 → 10174 |
| MovieLens-1M | GI-SASRec (IDF init) | 3000x2500 | 3 | -0.183 → -0.235 | -0.076 → -0.118 | 0.617 → 0.654 | 24673 → 23506 |
| MovieLens-1M | bary-cos (np) | 300x800 | 1 | -0.110 → -0.276 | — | — | — |
| MovieLens-1M | bary-cos (np) | 1000x1200 | 1 | -0.085 → -0.337 | — | — | — |
| MovieLens-1M | bary-cos (np) | 3000x2500 | 1 | -0.026 → -0.036 | — | — | — |
| MovieLens-1M | ItemKNN (np) | 300x800 | 1 | -0.103 → -0.143 | — | — | — |
| MovieLens-1M | ItemKNN (np) | 1000x1200 | 1 | -0.074 → -0.130 | — | — | — |
| MovieLens-1M | ItemKNN (np) | 3000x2500 | 1 | -0.024 → -0.046 | — | — | — |
| Amazon Beauty | GI-SASRec (raw init) | 300x800 | 3 | -0.065 → -0.109 | -0.059 → -0.099 | 0.626 → 0.718 | 7210 → 6118 |
| Amazon Beauty | GI-SASRec (raw init) | 1000x1200 | 3 | -0.060 → -0.101 | -0.049 → -0.084 | 0.637 → 0.710 | 11565 → 9985 |
| Amazon Beauty | GI-SASRec (raw init) | 3000x2500 | 3 | -0.055 → -0.104 | -0.038 → -0.080 | 0.648 → 0.725 | 24738 → 22613 |
| Amazon Beauty | GI-SASRec (IDF init) | 300x800 | 3 | -0.060 → -0.090 | -0.052 → -0.081 | 0.617 → 0.692 | 7337 → 6500 |
| Amazon Beauty | GI-SASRec (IDF init) | 1000x1200 | 3 | -0.055 → -0.085 | -0.041 → -0.072 | 0.629 → 0.691 | 11655 → 10520 |
| Amazon Beauty | GI-SASRec (IDF init) | 3000x2500 | 3 | -0.050 → -0.086 | -0.031 → -0.063 | 0.627 → 0.708 | 24813 → 23229 |
| Amazon Beauty | bary-cos (np) | 300x800 | 1 | 0.090 → 0.067 | — | — | — |
| Amazon Beauty | bary-cos (np) | 1000x1200 | 1 | 0.049 → 0.036 | — | — | — |
| Amazon Beauty | bary-cos (np) | 3000x2500 | 1 | 0.003 → -0.020 | — | — | — |
| Amazon Beauty | ItemKNN (np) | 300x800 | 1 | 0.089 → 0.088 | — | — | — |
| Amazon Beauty | ItemKNN (np) | 1000x1200 | 1 | 0.048 → -0.043 | — | — | — |
| Amazon Beauty | ItemKNN (np) | 3000x2500 | 1 | -0.000 → -0.413 | — | — | — |
| Arts & Crafts | GI-SASRec (raw init) | 300x800 | 3 | -0.074 → -0.103 | -0.067 → -0.088 | 0.639 → 0.703 | 7014 → 6123 |
| Arts & Crafts | GI-SASRec (raw init) | 1000x1200 | 3 | -0.051 → -0.055 | -0.045 → -0.030 | 0.599 → 0.569 | 11510 → 10461 |
| Arts & Crafts | GI-SASRec (raw init) | 3000x2500 | 3 | -0.040 → -0.040 | -0.024 → -0.024 | 0.579 → 0.579 | 24626 → 24626 |
| Arts & Crafts | GI-SASRec (IDF init) | 300x800 | 3 | -0.079 → -0.109 | -0.055 → -0.088 | 0.605 → 0.688 | 7216 → 6430 |
| Arts & Crafts | GI-SASRec (IDF init) | 1000x1200 | 3 | -0.054 → -0.049 | -0.041 → -0.017 | 0.584 → 0.538 | 11566 → 10900 |
| Arts & Crafts | GI-SASRec (IDF init) | 3000x2500 | 3 | -0.043 → -0.044 | -0.023 → -0.009 | 0.571 → 0.526 | 24723 → 23748 |
| Grocery | GI-SASRec (raw init) | 300x800 | 3 | -0.096 → -0.144 | -0.098 → -0.143 | 0.737 → 0.814 | 7131 → 6227 |
| Grocery | GI-SASRec (raw init) | 1000x1200 | 3 | -0.076 → -0.123 | -0.059 → -0.110 | 0.668 → 0.771 | 11504 → 10046 |
| Grocery | GI-SASRec (raw init) | 3000x2500 | 3 | -0.055 → -0.085 | -0.035 → -0.055 | 0.621 → 0.662 | 24693 → 23173 |
| Grocery | GI-SASRec (IDF init) | 300x800 | 3 | -0.098 → -0.137 | -0.095 → -0.133 | 0.700 → 0.777 | 7256 → 6574 |
| Grocery | GI-SASRec (IDF init) | 1000x1200 | 3 | -0.067 → -0.094 | -0.040 → -0.071 | 0.610 → 0.686 | 11596 → 10668 |
| Grocery | GI-SASRec (IDF init) | 3000x2500 | 2 | -0.051 → -0.064 | -0.027 → -0.032 | 0.592 → 0.600 | 24732 → 23688 |
| Office Products | GI-SASRec (raw init) | 300x800 | 3 | -0.064 → -0.080 | -0.040 → -0.033 | 0.624 → 0.592 | 7625 → 7097 |
| Office Products | GI-SASRec (raw init) | 1000x1200 | 3 | -0.047 → -0.047 | -0.022 → -0.022 | 0.586 → 0.586 | 11777 → 11777 |
| Office Products | GI-SASRec (raw init) | 3000x2500 | 3 | -0.047 → -0.047 | -0.033 → -0.033 | 0.614 → 0.614 | 24823 → 24823 |
| Office Products | GI-SASRec (IDF init) | 300x800 | 3 | -0.061 → -0.167 | -0.041 → -0.024 | 0.619 → 0.568 | 7741 → 7472 |
| Office Products | GI-SASRec (IDF init) | 1000x1200 | 3 | -0.042 → -0.012 | -0.022 → 0.023 | 0.586 → 0.409 | 11809 → 11704 |
| Office Products | GI-SASRec (IDF init) | 3000x2500 | 3 | -0.046 → -0.029 | -0.029 → -0.000 | 0.593 → 0.501 | 24799 → 24257 |

## Graph Laplacian spectrum

| dataset | model | scale | runs | lambda2 | lambda max | spectral gap | components |
|---|---|---|---|---|---|---|---|
| MovieLens-1M | GI-NCF (paper cfg) | 300x800 | 3 | 0.667 → 0.622 | 47.0 → 22.8 | 0.014 → 0.027 | 1.000 → 1.000 |
| MovieLens-1M | GI-NCF (paper cfg) | 1000x1200 | 3 | 0.593 → 0.544 | 135.6 → 22.9 | 0.004 → 0.024 | 1.000 → 1.000 |
| MovieLens-1M | GI-NCF (paper cfg) | 3000x2500 | 3 | 0.499 → 0.503 | 198.5 → 21.7 | 0.003 → 0.023 | 1.000 → 1.000 |
| MovieLens-1M | GI-SASRec (raw init) | 300x800 | 3 | 0.666 → 0.595 | 46.1 → 26.8 | 0.014 → 0.022 | 1.000 → 1.000 |
| MovieLens-1M | GI-SASRec (raw init) | 1000x1200 | 3 | 0.605 → 0.558 | 136.3 → 45.0 | 0.004 → 0.012 | 1.000 → 1.000 |
| MovieLens-1M | GI-SASRec (raw init) | 3000x2500 | 3 | 0.501 → 0.520 | 199.4 → 48.2 | 0.003 → 0.011 | 1.000 → 1.000 |
| MovieLens-1M | GI-SASRec (IDF init) | 300x800 | 3 | 0.707 → 0.657 | 46.7 → 36.3 | 0.015 → 0.018 | 1.000 → 1.000 |
| MovieLens-1M | GI-SASRec (IDF init) | 1000x1200 | 3 | 0.661 → 0.638 | 146.4 → 65.3 | 0.005 → 0.010 | 1.000 → 1.000 |
| MovieLens-1M | GI-SASRec (IDF init) | 3000x2500 | 3 | 0.448 → 0.446 | 242.3 → 52.7 | 0.002 → 0.008 | 1.000 → 1.000 |
| MovieLens-1M | bary-cos (np) | 300x800 | 1 | 0.805 → 1.391 | — | — | — |
| MovieLens-1M | bary-cos (np) | 1000x1200 | 1 | 0.533 → 0.749 | — | — | — |
| MovieLens-1M | bary-cos (np) | 3000x2500 | 1 | 4.209 → 6.431 | — | — | — |
| MovieLens-1M | ItemKNN (np) | 300x800 | 1 | 0.777 → 0.756 | — | — | — |
| MovieLens-1M | ItemKNN (np) | 1000x1200 | 1 | 0.521 → 0.526 | — | — | — |
| MovieLens-1M | ItemKNN (np) | 3000x2500 | 1 | 4.190 → 4.896 | — | — | — |
| Amazon Beauty | GI-SASRec (raw init) | 300x800 | 3 | 0.456 → 0.336 | 245.2 → 109.0 | 0.002 → 0.003 | 1.000 → 1.000 |
| Amazon Beauty | GI-SASRec (raw init) | 1000x1200 | 3 | 0.754 → 0.464 | 384.3 → 215.0 | 0.002 → 0.002 | 1.000 → 1.000 |
| Amazon Beauty | GI-SASRec (raw init) | 3000x2500 | 3 | 1.192 → 0.852 | 701.6 → 533.7 | 0.002 → 0.002 | 1.000 → 1.000 |
| Amazon Beauty | GI-SASRec (IDF init) | 300x800 | 3 | 0.697 → 0.573 | 257.9 → 152.6 | 0.003 → 0.004 | 1.000 → 1.000 |
| Amazon Beauty | GI-SASRec (IDF init) | 1000x1200 | 3 | 1.046 → 0.808 | 398.4 → 274.4 | 0.003 → 0.003 | 1.000 → 1.000 |
| Amazon Beauty | GI-SASRec (IDF init) | 3000x2500 | 3 | 1.333 → 1.110 | 694.2 → 545.4 | 0.002 → 0.002 | 1.000 → 1.000 |
| Amazon Beauty | bary-cos (np) | 300x800 | 1 | 4.237 → 6.250 | — | — | — |
| Amazon Beauty | bary-cos (np) | 1000x1200 | 1 | 5.917 → 7.600 | — | — | — |
| Amazon Beauty | bary-cos (np) | 3000x2500 | 1 | 6.923 → 6.254 | — | — | — |
| Amazon Beauty | ItemKNN (np) | 300x800 | 1 | 4.032 → 2.062 | — | — | — |
| Amazon Beauty | ItemKNN (np) | 1000x1200 | 1 | 5.744 → 2.795 | — | — | — |
| Amazon Beauty | ItemKNN (np) | 3000x2500 | 1 | 6.773 → 3.811 | — | — | — |
| Arts & Crafts | GI-SASRec (raw init) | 300x800 | 3 | 0.880 → 0.646 | 233.5 → 121.0 | 0.004 → 0.005 | 1.000 → 1.000 |
| Arts & Crafts | GI-SASRec (raw init) | 1000x1200 | 3 | 1.350 → 1.031 | 386.3 → 355.1 | 0.003 → 0.003 | 1.000 → 1.000 |
| Arts & Crafts | GI-SASRec (raw init) | 3000x2500 | 3 | 1.080 → 1.077 | 703.8 → 701.8 | 0.002 → 0.002 | 1.000 → 1.000 |
| Arts & Crafts | GI-SASRec (IDF init) | 300x800 | 3 | 1.220 → 0.921 | 256.9 → 159.8 | 0.005 → 0.006 | 1.000 → 1.000 |
| Arts & Crafts | GI-SASRec (IDF init) | 1000x1200 | 3 | 1.493 → 1.233 | 409.3 → 393.6 | 0.004 → 0.003 | 1.000 → 1.000 |
| Arts & Crafts | GI-SASRec (IDF init) | 3000x2500 | 3 | 1.342 → 1.017 | 712.8 → 674.5 | 0.002 → 0.002 | 1.000 → 1.000 |
| Grocery | GI-SASRec (raw init) | 300x800 | 3 | 1.065 → 0.700 | 200.1 → 88.2 | 0.005 → 0.008 | 1.000 → 1.000 |
| Grocery | GI-SASRec (raw init) | 1000x1200 | 3 | 1.416 → 0.894 | 388.9 → 190.8 | 0.004 → 0.005 | 1.000 → 1.000 |
| Grocery | GI-SASRec (raw init) | 3000x2500 | 3 | 1.391 → 0.992 | 733.3 → 487.3 | 0.002 → 0.002 | 1.000 → 1.000 |
| Grocery | GI-SASRec (IDF init) | 300x800 | 3 | 1.072 → 0.898 | 122.0 → 65.0 | 0.009 → 0.014 | 1.000 → 1.000 |
| Grocery | GI-SASRec (IDF init) | 1000x1200 | 3 | 1.559 → 1.408 | 425.2 → 310.8 | 0.004 → 0.005 | 1.000 → 1.000 |
| Grocery | GI-SASRec (IDF init) | 3000x2500 | 2 | 1.448 → 1.210 | 725.7 → 544.1 | 0.002 → 0.002 | 1.000 → 1.000 |
| Office Products | GI-SASRec (raw init) | 300x800 | 3 | 0.768 → 0.804 | 273.3 → 197.3 | 0.003 → 0.004 | 1.000 → 1.000 |
| Office Products | GI-SASRec (raw init) | 1000x1200 | 3 | 0.739 → 0.737 | 383.4 → 382.3 | 0.002 → 0.002 | 1.000 → 1.000 |
| Office Products | GI-SASRec (raw init) | 3000x2500 | 3 | 0.741 → 0.739 | 602.2 → 600.5 | 0.001 → 0.001 | 1.000 → 1.000 |
| Office Products | GI-SASRec (IDF init) | 300x800 | 3 | 0.878 → 0.935 | 281.6 → 241.3 | 0.003 → 0.004 | 1.000 → 1.000 |
| Office Products | GI-SASRec (IDF init) | 1000x1200 | 3 | 0.573 → 0.555 | 362.3 → 328.5 | 0.002 → 0.002 | 1.000 → 1.000 |
| Office Products | GI-SASRec (IDF init) | 3000x2500 | 3 | 0.601 → 0.417 | 601.2 → 506.5 | 0.001 → 0.001 | 1.000 → 1.000 |

## Isometry of the embedding

| dataset | model | scale | runs | Kruskal stress | Spearman rho | residual variance |
|---|---|---|---|---|---|---|
| MovieLens-1M | GI-NCF (paper cfg) | 300x800 | 3 | 0.397 → 0.450 | 0.863 → 0.850 | 0.269 → 0.271 |
| MovieLens-1M | GI-NCF (paper cfg) | 1000x1200 | 3 | 0.340 → 0.400 | 0.932 → 0.907 | 0.141 → 0.185 |
| MovieLens-1M | GI-NCF (paper cfg) | 3000x2500 | 3 | 0.351 → 0.308 | 0.941 → 0.956 | 0.106 → 0.104 |
| MovieLens-1M | GI-SASRec (raw init) | 300x800 | 3 | 0.237 → 0.263 | 0.895 → 0.884 | 0.216 → 0.222 |
| MovieLens-1M | GI-SASRec (raw init) | 1000x1200 | 3 | 0.226 → 0.219 | 0.931 → 0.928 | 0.131 → 0.139 |
| MovieLens-1M | GI-SASRec (raw init) | 3000x2500 | 3 | 0.298 → 0.252 | 0.930 → 0.930 | 0.118 → 0.126 |
| MovieLens-1M | GI-SASRec (IDF init) | 300x800 | 3 | 0.401 → 0.407 | 0.867 → 0.859 | 0.261 → 0.264 |
| MovieLens-1M | GI-SASRec (IDF init) | 1000x1200 | 3 | 0.342 → 0.348 | 0.936 → 0.931 | 0.134 → 0.145 |
| MovieLens-1M | GI-SASRec (IDF init) | 3000x2500 | 3 | 0.368 → 0.332 | 0.925 → 0.933 | 0.119 → 0.135 |
| MovieLens-1M | bary-cos (np) | 300x800 | 1 | — | 0.824 → 0.788 | — |
| MovieLens-1M | bary-cos (np) | 1000x1200 | 1 | — | 0.847 → 0.798 | — |
| MovieLens-1M | bary-cos (np) | 3000x2500 | 1 | — | 0.931 → 0.936 | — |
| MovieLens-1M | ItemKNN (np) | 300x800 | 1 | — | 0.829 → 0.823 | — |
| MovieLens-1M | ItemKNN (np) | 1000x1200 | 1 | — | 0.853 → 0.860 | — |
| MovieLens-1M | ItemKNN (np) | 3000x2500 | 1 | — | 0.927 → 0.876 | — |
| Amazon Beauty | GI-SASRec (raw init) | 300x800 | 3 | 0.336 → 0.135 | 0.631 → 0.886 | 0.328 → 0.208 |
| Amazon Beauty | GI-SASRec (raw init) | 1000x1200 | 3 | 0.446 → 0.149 | 0.424 → 0.845 | 0.406 → 0.260 |
| Amazon Beauty | GI-SASRec (raw init) | 3000x2500 | 3 | 0.670 → 0.238 | 0.382 → 0.725 | 0.504 → 0.356 |
| Amazon Beauty | GI-SASRec (IDF init) | 300x800 | 3 | 0.321 → 0.237 | 0.477 → 0.845 | 0.397 → 0.268 |
| Amazon Beauty | GI-SASRec (IDF init) | 1000x1200 | 3 | 0.442 → 0.211 | 0.321 → 0.733 | 0.500 → 0.357 |
| Amazon Beauty | GI-SASRec (IDF init) | 3000x2500 | 3 | 0.623 → 0.221 | 0.326 → 0.645 | 0.636 → 0.449 |
| Amazon Beauty | bary-cos (np) | 300x800 | 1 | — | 0.393 → 0.689 | — |
| Amazon Beauty | bary-cos (np) | 1000x1200 | 1 | — | 0.195 → 0.630 | — |
| Amazon Beauty | bary-cos (np) | 3000x2500 | 1 | — | 0.254 → 0.862 | — |
| Amazon Beauty | ItemKNN (np) | 300x800 | 1 | — | 0.470 → 0.850 | — |
| Amazon Beauty | ItemKNN (np) | 1000x1200 | 1 | — | 0.227 → 0.884 | — |
| Amazon Beauty | ItemKNN (np) | 3000x2500 | 1 | — | 0.210 → 0.764 | — |
| Arts & Crafts | GI-SASRec (raw init) | 300x800 | 3 | 0.321 → 0.159 | 0.718 → 0.872 | 0.370 → 0.267 |
| Arts & Crafts | GI-SASRec (raw init) | 1000x1200 | 3 | 0.646 → 0.327 | 0.243 → 0.582 | 0.770 → 0.525 |
| Arts & Crafts | GI-SASRec (raw init) | 3000x2500 | 3 | 0.759 → 0.759 | 0.330 → 0.330 | 0.751 → 0.751 |
| Arts & Crafts | GI-SASRec (IDF init) | 300x800 | 3 | 0.316 → 0.250 | 0.373 → 0.746 | 0.612 → 0.449 |
| Arts & Crafts | GI-SASRec (IDF init) | 1000x1200 | 3 | 0.587 → 0.355 | 0.159 → 0.336 | 0.805 → 0.754 |
| Arts & Crafts | GI-SASRec (IDF init) | 3000x2500 | 3 | 0.741 → 0.397 | 0.289 → 0.525 | 0.691 → 0.697 |
| Grocery | GI-SASRec (raw init) | 300x800 | 3 | 0.185 → 0.175 | 0.785 → 0.844 | 0.358 → 0.294 |
| Grocery | GI-SASRec (raw init) | 1000x1200 | 3 | 0.400 → 0.151 | 0.492 → 0.807 | 0.507 → 0.354 |
| Grocery | GI-SASRec (raw init) | 3000x2500 | 3 | 0.662 → 0.258 | 0.330 → 0.611 | 0.711 → 0.549 |
| Grocery | GI-SASRec (IDF init) | 300x800 | 3 | 0.313 → 0.351 | 0.737 → 0.797 | 0.453 → 0.354 |
| Grocery | GI-SASRec (IDF init) | 1000x1200 | 3 | 0.473 → 0.263 | 0.166 → 0.596 | 0.768 → 0.556 |
| Grocery | GI-SASRec (IDF init) | 3000x2500 | 2 | 0.671 → 0.295 | 0.242 → 0.440 | 0.814 → 0.723 |
| Office Products | GI-SASRec (raw init) | 300x800 | 3 | 0.478 → 0.270 | 0.432 → 0.671 | 0.590 → 0.435 |
| Office Products | GI-SASRec (raw init) | 1000x1200 | 3 | 0.704 → 0.704 | 0.229 → 0.229 | 0.810 → 0.810 |
| Office Products | GI-SASRec (raw init) | 3000x2500 | 3 | 0.801 → 0.801 | 0.386 → 0.386 | 0.689 → 0.689 |
| Office Products | GI-SASRec (IDF init) | 300x800 | 3 | 0.489 → 0.311 | 0.360 → 0.495 | 0.735 → 0.666 |
| Office Products | GI-SASRec (IDF init) | 1000x1200 | 3 | 0.719 → 0.421 | 0.254 → 0.447 | 0.738 → 0.667 |
| Office Products | GI-SASRec (IDF init) | 3000x2500 | 3 | 0.757 → 0.333 | 0.427 → 0.584 | 0.626 → 0.574 |

## Persistent homology

| dataset | model | scale | runs | H0 geo pers. | H0 geo count | H1 geo pers. | H1 geo count | H0 lat pers. | H0 lat count | H1 lat pers. | H1 lat count |
|---|---|---|---|---|---|---|---|---|---|---|---|
| MovieLens-1M | GI-NCF (paper cfg) | 300x800 | 3 | 0.831 → 0.875 | 800.0 → 800.0 | 0.368 → 0.437 | 1185 → 1757 | 1.299 → 1.650 | 800.0 → 800.0 | 0.212 → 0.404 | 261.7 → 477.3 |
| MovieLens-1M | GI-NCF (paper cfg) | 1000x1200 | 3 | 0.648 → 0.715 | 1200 → 1200 | 0.294 → 0.358 | 2200 → 2731 | 0.703 → 1.211 | — | — | — |
| MovieLens-1M | GI-NCF (paper cfg) | 3000x2500 | 3 | 0.262 → 0.303 | — | — | — | 0.151 → 0.444 | — | — | — |
| MovieLens-1M | GI-SASRec (raw init) | 300x800 | 3 | 0.832 → 0.828 | 800.0 → 800.0 | 0.368 → 0.407 | 1179 → 1772 | 1.017 → 1.162 | 800.0 → 800.0 | 0.201 → 0.254 | 291.0 → 405.0 |
| MovieLens-1M | GI-SASRec (raw init) | 1000x1200 | 3 | 0.647 → 0.640 | 1200 → 1200 | 0.281 → 0.289 | 2309 → 2634 | 0.516 → 0.753 | — | — | — |
| MovieLens-1M | GI-SASRec (raw init) | 3000x2500 | 3 | 0.263 → 0.257 | — | — | — | 0.107 → 0.251 | — | — | — |
| MovieLens-1M | GI-SASRec (IDF init) | 300x800 | 3 | 0.813 → 0.797 | 800.0 → 800.0 | 0.353 → 0.361 | 1188 → 1636 | 1.248 → 1.281 | 800.0 → 800.0 | 0.232 → 0.263 | 225.3 → 237.7 |
| MovieLens-1M | GI-SASRec (IDF init) | 1000x1200 | 3 | 0.615 → 0.604 | — | — | — | 0.642 → 0.801 | — | — | — |
| MovieLens-1M | GI-SASRec (IDF init) | 3000x2500 | 3 | 0.224 → 0.207 | — | — | — | 0.096 → 0.248 | — | — | — |
| MovieLens-1M | bary-cos (np) | 300x800 | 1 | 1.060 → 0.899 | — | — | — | 1.735 → 1.670 | — | — | — |
| MovieLens-1M | bary-cos (np) | 1000x1200 | 1 | 1.037 → 0.623 | — | — | — | 1.933 → 1.234 | — | — | — |
| MovieLens-1M | bary-cos (np) | 3000x2500 | 1 | 0.393 → 0.247 | — | — | — | 0.368 → 0.191 | — | — | — |
| MovieLens-1M | ItemKNN (np) | 300x800 | 1 | 1.051 → 1.085 | — | — | — | 1.675 → 1.859 | — | — | — |
| MovieLens-1M | ItemKNN (np) | 1000x1200 | 1 | 1.034 → 1.040 | — | — | — | 1.918 → 1.985 | — | — | — |
| MovieLens-1M | ItemKNN (np) | 3000x2500 | 1 | 0.395 → 0.320 | — | — | — | 0.361 → 0.421 | — | — | — |
| Amazon Beauty | GI-SASRec (raw init) | 300x800 | 3 | 0.605 → 0.559 | 800.0 → 800.0 | 0.265 → 0.267 | 412.0 → 1737 | 0.147 → 0.546 | 800.0 → 800.0 | 0.026 → 0.143 | 76.7 → 355.3 |
| Amazon Beauty | GI-SASRec (raw init) | 1000x1200 | 3 | 0.574 → 0.517 | 1200 → 1200 | 0.243 → 0.232 | 1336 → 2662 | 0.124 → 0.398 | 1200 → 1200 | 0.072 → 0.102 | 582.3 → 565.0 |
| Amazon Beauty | GI-SASRec (raw init) | 3000x2500 | 3 | 0.464 → 0.426 | — | — | — | 0.027 → 0.248 | — | — | — |
| Amazon Beauty | GI-SASRec (IDF init) | 300x800 | 3 | 0.586 → 0.567 | 800.0 → 800.0 | 0.202 → 0.265 | 315.7 → 1445 | 0.292 → 0.637 | 800.0 → 800.0 | 0.059 → 0.128 | 141.0 → 363.7 |
| Amazon Beauty | GI-SASRec (IDF init) | 1000x1200 | 3 | 0.577 → 0.538 | 1200 → 1200 | 0.221 → 0.244 | 882.3 → 2198 | 0.143 → 0.469 | 1200 → 1200 (1/3) | 0.098 → 0.089 (1/3) | 624.0 → 670.0 (1/3) |
| Amazon Beauty | GI-SASRec (IDF init) | 3000x2500 | 3 | 0.453 → 0.427 | — | — | — | 0.079 → 0.321 | — | — | — |
| Amazon Beauty | bary-cos (np) | 300x800 | 1 | 0.411 → 0.407 | — | — | — | 0.486 → 0.499 | — | — | — |
| Amazon Beauty | bary-cos (np) | 1000x1200 | 1 | 0.381 → 0.385 | — | — | — | 0.298 → 0.292 | — | — | — |
| Amazon Beauty | bary-cos (np) | 3000x2500 | 1 | 0.454 → 0.379 | — | — | — | 0.140 → 0.126 | — | — | — |
| Amazon Beauty | ItemKNN (np) | 300x800 | 1 | 0.411 → 0.254 | — | — | — | 0.483 → 0.209 | — | — | — |
| Amazon Beauty | ItemKNN (np) | 1000x1200 | 1 | 0.384 → 0.209 | — | — | — | 0.295 → 0.225 | — | — | — |
| Amazon Beauty | ItemKNN (np) | 3000x2500 | 1 | 0.446 → 0.110 | — | — | — | 0.164 → 0.087 | — | — | — |
| Arts & Crafts | GI-SASRec (raw init) | 300x800 | 3 | 0.582 → 0.548 | 800.0 → 800.0 | 0.250 → 0.261 | 604.3 → 1327 | 0.142 → 0.459 | 798.7 → 798.3 | 0.031 → 0.074 | 51.7 → 133.0 |
| Arts & Crafts | GI-SASRec (raw init) | 1000x1200 | 3 | 0.499 → 0.458 | 1200 → 1200 | 0.190 → 0.196 | 582.3 → 849.3 | 0.024 → 0.216 | — | — | — |
| Arts & Crafts | GI-SASRec (raw init) | 3000x2500 | 3 | 0.403 → 0.401 | — | — | — | 0.016 → 0.016 | — | — | — |
| Arts & Crafts | GI-SASRec (IDF init) | 300x800 | 3 | 0.528 → 0.535 | 800.0 → 800.0 | 0.217 → 0.259 | 496.7 → 1394 | 0.345 → 0.597 | 799.3 → 798.7 | 0.062 → 0.132 | 258.3 → 337.0 |
| Arts & Crafts | GI-SASRec (IDF init) | 1000x1200 | 3 | 0.502 → 0.479 | — | — | — | 0.095 → 0.268 | — | — | — |
| Arts & Crafts | GI-SASRec (IDF init) | 3000x2500 | 3 | 0.395 → 0.383 | — | — | — | 0.023 → 0.195 | — | — | — |
| Grocery | GI-SASRec (raw init) | 300x800 | 3 | 0.648 → 0.613 | 800.0 → 800.0 | 0.290 → 0.305 | 942.7 → 2007 | 0.474 → 0.681 | 800.0 → 800.0 | 0.139 → 0.145 | 279.3 → 353.0 |
| Grocery | GI-SASRec (raw init) | 1000x1200 | 3 | 0.614 → 0.574 | 1200 → 1200 | 0.271 → 0.271 | 976.3 → 2922 | 0.179 → 0.488 | — | — | — |
| Grocery | GI-SASRec (raw init) | 3000x2500 | 3 | 0.470 → 0.431 | — | — | — | 0.034 → 0.260 | — | — | — |
| Grocery | GI-SASRec (IDF init) | 300x800 | 3 | 0.552 → 0.535 | 800.0 → 800.0 | 0.248 → 0.264 | 1063 → 2052 | 0.671 → 0.835 | 800.0 → 800.0 | 0.118 → 0.154 | 391.3 → 670.0 |
| Grocery | GI-SASRec (IDF init) | 1000x1200 | 3 | 0.572 → 0.550 | 1200 → 1200 | 0.242 → 0.246 | 1576 → 2492 | 0.215 → 0.495 | — | — | — |
| Grocery | GI-SASRec (IDF init) | 3000x2500 | 2 | 0.448 → 0.414 | — | — | — | 0.064 → 0.262 | — | — | — |
| Office Products | GI-SASRec (raw init) | 300x800 | 3 | 0.541 → 0.489 | 800.0 → 800.0 | 0.192 → 0.199 | 245.0 → 607.7 | 0.074 → 0.319 | 771.0 → 793.3 | 0.010 → 0.077 | 20.0 → 117.7 |
| Office Products | GI-SASRec (raw init) | 1000x1200 | 3 | 0.448 → 0.446 | — | — | — | 0.022 → 0.022 | — | — | — |
| Office Products | GI-SASRec (raw init) | 3000x2500 | 3 | 0.336 → 0.335 | — | — | — | 0.009 → 0.009 | — | — | — |
| Office Products | GI-SASRec (IDF init) | 300x800 | 3 | 0.508 → 0.448 | 800.0 → 800.0 | 0.187 → 0.190 | 215.3 → 333.0 | 0.096 → 0.281 | 795.3 → 795.3 | 0.017 → 0.073 | 33.7 → 77.0 |
| Office Products | GI-SASRec (IDF init) | 1000x1200 | 3 | 0.417 → 0.349 | — | — | — | 0.012 → 0.120 | — | — | — |
| Office Products | GI-SASRec (IDF init) | 3000x2500 | 3 | 0.324 → 0.285 | — | — | — | 0.022 → 0.167 | — | — | — |
