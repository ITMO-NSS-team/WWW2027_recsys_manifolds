# Topology of the learned manifold, before vs after optimisation

`before` = epoch 0 (the initial approximation), `after` = the final analysed snapshot; cells are `before → after`, averaged over the available seeds. H0 persistence is the lifetime of the longest connected-component feature: larger means the cloud separates into parts that stay apart over a wider range of scales.

## H0 persistence — all runs

| dataset | model | scale | seeds | geodesic H0 | latent H0 |
|---|---|---|---|---|---|
| MovieLens-1M | GI-NCF (paper cfg) | 300x800 | 3 | 0.831 → 0.875 | 1.299 → 1.650 |
| MovieLens-1M | GI-NCF (paper cfg) | 1000x1200 | 3 | 0.648 → 0.715 | 0.703 → 1.211 |
| MovieLens-1M | GI-NCF (paper cfg) | 3000x2500 | 3 | 0.262 → 0.303 | 0.151 → 0.444 |
| MovieLens-1M | GI-SASRec (raw init) | 300x800 | 3 | 0.832 → 0.828 | 1.017 → 1.162 |
| MovieLens-1M | GI-SASRec (raw init) | 1000x1200 | 3 | 0.647 → 0.640 | 0.516 → 0.753 |
| MovieLens-1M | GI-SASRec (raw init) | 3000x2500 | 3 | 0.263 → 0.257 | 0.107 → 0.251 |
| MovieLens-1M | GI-SASRec (IDF init) | 300x800 | 3 | 0.813 → 0.797 | 1.248 → 1.281 |
| MovieLens-1M | GI-SASRec (IDF init) | 1000x1200 | 3 | 0.615 → 0.604 | 0.642 → 0.801 |
| MovieLens-1M | GI-SASRec (IDF init) | 3000x2500 | 3 | 0.224 → 0.207 | 0.096 → 0.248 |
| Amazon Beauty | GI-SASRec (raw init) | 300x800 | 3 | 0.605 → 0.559 | 0.147 → 0.546 |
| Amazon Beauty | GI-SASRec (raw init) | 1000x1200 | 3 | 0.574 → 0.517 | 0.124 → 0.398 |
| Amazon Beauty | GI-SASRec (raw init) | 3000x2500 | 3 | 0.464 → 0.426 | 0.027 → 0.248 |
| Amazon Beauty | GI-SASRec (IDF init) | 300x800 | 3 | 0.586 → 0.567 | 0.292 → 0.637 |
| Amazon Beauty | GI-SASRec (IDF init) | 1000x1200 | 3 | 0.577 → 0.538 | 0.143 → 0.469 |
| Amazon Beauty | GI-SASRec (IDF init) | 3000x2500 | 3 | 0.453 → 0.427 | 0.079 → 0.321 |
| Arts & Crafts | GI-SASRec (raw init) | 300x800 | 3 | 0.582 → 0.548 | 0.142 → 0.459 |
| Arts & Crafts | GI-SASRec (raw init) | 1000x1200 | 3 | 0.499 → 0.458 | 0.024 → 0.216 |
| Arts & Crafts | GI-SASRec (raw init) | 3000x2500 | 3 | 0.403 → 0.401 | 0.016 → 0.016 |
| Arts & Crafts | GI-SASRec (IDF init) | 300x800 | 3 | 0.528 → 0.535 | 0.345 → 0.597 |
| Arts & Crafts | GI-SASRec (IDF init) | 1000x1200 | 3 | 0.502 → 0.479 | 0.095 → 0.268 |
| Arts & Crafts | GI-SASRec (IDF init) | 3000x2500 | 3 | 0.395 → 0.383 | 0.023 → 0.195 |
| Grocery | GI-SASRec (raw init) | 300x800 | 3 | 0.648 → 0.613 | 0.474 → 0.681 |
| Grocery | GI-SASRec (raw init) | 1000x1200 | 3 | 0.614 → 0.574 | 0.179 → 0.488 |
| Grocery | GI-SASRec (raw init) | 3000x2500 | 3 | 0.470 → 0.431 | 0.034 → 0.260 |
| Grocery | GI-SASRec (IDF init) | 300x800 | 3 | 0.552 → 0.535 | 0.671 → 0.835 |
| Grocery | GI-SASRec (IDF init) | 1000x1200 | 3 | 0.572 → 0.550 | 0.215 → 0.495 |
| Grocery | GI-SASRec (IDF init) | 3000x2500 | 2 | 0.448 → 0.414 | 0.064 → 0.262 |
| Office Products | GI-SASRec (raw init) | 300x800 | 3 | 0.541 → 0.489 | 0.074 → 0.319 |
| Office Products | GI-SASRec (raw init) | 1000x1200 | 3 | 0.448 → 0.446 | 0.022 → 0.022 |
| Office Products | GI-SASRec (raw init) | 3000x2500 | 3 | 0.336 → 0.335 | 0.009 → 0.009 |
| Office Products | GI-SASRec (IDF init) | 300x800 | 3 | 0.508 → 0.448 | 0.096 → 0.281 |
| Office Products | GI-SASRec (IDF init) | 1000x1200 | 3 | 0.417 → 0.349 | 0.012 → 0.120 |
| Office Products | GI-SASRec (IDF init) | 3000x2500 | 3 | 0.324 → 0.285 | 0.022 → 0.167 |

## H1 — coverage first

`h1_count = -1` in the raw CSVs means the computation was skipped for cost, not that there are no cycles. Runs with a real H1:

| scale | runs | geodesic H1 | latent H1 |
|---|---|---|---|
| 300x800 | 36 | 36/36 | 36/36 |
| 1000x1200 | 36 | 21/36 | 4/36 |
| 3000x2500 | 35 | 0/35 | 0/35 |

Only 300×800 is complete, so only it is tabulated below. At 3000×2500 no run has H1 at all.

### H1 at 300×800

| dataset | model | seeds | geodesic H1 count | geodesic H1 pers. | latent H1 count | latent H1 pers. |
|---|---|---|---|---|---|---|
| MovieLens-1M | GI-NCF (paper cfg) | 3 | 1185.333 → 1756.667 | 0.368 → 0.437 | 261.667 → 477.333 | 0.212 → 0.404 |
| MovieLens-1M | GI-SASRec (raw init) | 3 | 1178.667 → 1771.667 | 0.368 → 0.407 | 291.000 → 405.000 | 0.201 → 0.254 |
| MovieLens-1M | GI-SASRec (IDF init) | 3 | 1188.000 → 1636.000 | 0.353 → 0.361 | 225.333 → 237.667 | 0.232 → 0.263 |
| Amazon Beauty | GI-SASRec (raw init) | 3 | 412.000 → 1737.000 | 0.265 → 0.267 | 76.667 → 355.333 | 0.026 → 0.143 |
| Amazon Beauty | GI-SASRec (IDF init) | 3 | 315.667 → 1445.333 | 0.202 → 0.265 | 141.000 → 363.667 | 0.059 → 0.128 |
| Arts & Crafts | GI-SASRec (raw init) | 3 | 604.333 → 1327.333 | 0.250 → 0.261 | 51.667 → 133.000 | 0.031 → 0.074 |
| Arts & Crafts | GI-SASRec (IDF init) | 3 | 496.667 → 1394.333 | 0.217 → 0.259 | 258.333 → 337.000 | 0.062 → 0.132 |
| Grocery | GI-SASRec (raw init) | 3 | 942.667 → 2006.667 | 0.290 → 0.305 | 279.333 → 353.000 | 0.139 → 0.145 |
| Grocery | GI-SASRec (IDF init) | 3 | 1063.000 → 2052.000 | 0.248 → 0.264 | 391.333 → 670.000 | 0.118 → 0.154 |
| Office Products | GI-SASRec (raw init) | 3 | 245.000 → 607.667 | 0.192 → 0.199 | 20.000 → 117.667 | 0.010 → 0.077 |
| Office Products | GI-SASRec (IDF init) | 3 | 215.333 → 333.000 | 0.187 → 0.190 | 33.667 → 77.000 | 0.017 → 0.073 |

## H1 at large scales — subsampled separately

900 items drawn 3 times per snapshot, because the full computation is refused at these sizes. Mean ± std over the three draws. Comparable across epochs within a run, not across scales: the subsample is fixed while the catalogue grows.

| run | space | epoch | H1 count | H1 max persistence |
|---|---|---|---|---|
| amazon beauty frozenproj scaletest 3000x2500 0.001 | geodesic | 0 (first) | 183 ± 90 | 0.1851 ± 0.0077 |
| amazon beauty frozenproj scaletest 3000x2500 0.001 | geodesic | 99 (last) | 1069 ± 97 | 0.2173 ± 0.0056 |
| amazon beauty frozenproj scaletest 3000x2500 0.001 | latent | 0 (first) | 434 ± 14 | 0.0457 ± 0.0012 |
| amazon beauty frozenproj scaletest 3000x2500 0.001 | latent | 99 (last) | 738 ± 19 | 0.1018 ± 0.0145 |
| amazon beauty frozenproj scaletest 6000x3000 seed0 0.001 | geodesic | 0 (first) | 51 ± 61 | 0.0858 ± 0.0478 |
| amazon beauty frozenproj scaletest 6000x3000 seed0 0.001 | geodesic | 90 (last) | 432 ± 237 | 0.1236 ± 0.0311 |
| amazon beauty frozenproj scaletest 6000x3000 seed0 0.001 | latent | 0 (first) | 390 ± 16 | 0.0228 ± 0.0048 |
| amazon beauty frozenproj scaletest 6000x3000 seed0 0.001 | latent | 90 (last) | 544 ± 12 | 0.0425 ± 0.0029 |
| amazon beauty frozenproj scaletest 6000x3000 seed1 0.001 | geodesic | 0 (first) | 51 ± 61 | 0.0858 ± 0.0478 |
| amazon beauty frozenproj scaletest 6000x3000 seed1 0.001 | geodesic | 90 (last) | 384 ± 221 | 0.1200 ± 0.0293 |
| amazon beauty frozenproj scaletest 6000x3000 seed1 0.001 | latent | 0 (first) | 390 ± 16 | 0.0228 ± 0.0048 |
| amazon beauty frozenproj scaletest 6000x3000 seed1 0.001 | latent | 90 (last) | 616 ± 6 | 0.0509 ± 0.0038 |
| amazon beauty frozenproj scaletest 6000x3000 seed2 0.001 | geodesic | 0 (first) | 51 ± 61 | 0.0858 ± 0.0478 |
| amazon beauty frozenproj scaletest 6000x3000 seed2 0.001 | geodesic | 90 (last) | 405 ± 246 | 0.1255 ± 0.0252 |
| amazon beauty frozenproj scaletest 6000x3000 seed2 0.001 | latent | 0 (first) | 390 ± 16 | 0.0228 ± 0.0048 |
| amazon beauty frozenproj scaletest 6000x3000 seed2 0.001 | latent | 90 (last) | 534 ± 5 | 0.0435 ± 0.0030 |
