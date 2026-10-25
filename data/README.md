# Datasets

Five datasets are used. None of them is stored in this repository — download
them and place them exactly as shown below. Every loader resolves paths
through `repo_paths.DATA_DIR`, which defaults to this directory; set
`RM_DATA_DIR` to put the data on another disk instead.

## Expected layout

```
data/
├── ml-1m/
│   └── ratings.dat                              (and the optional movies.dat)
├── amazon_beauty/
│   └── Beauty_and_Personal_Care.csv
├── amazon_arts_crafts_sewing/
│   └── Arts_Crafts_and_Sewing.csv
├── amazon_grocery_gourmet_food/
│   └── Grocery_and_Gourmet_Food.csv
└── amazon_office_products/
    └── Office_Products.csv
```

The directory names are what the drivers pass as `dataset_dir_name`, and the
CSV basenames are what they pass as `amazon_category` — renaming either breaks
the arm scripts.

## 1. MovieLens-1M

Source: GroupLens, <https://grouplens.org/datasets/movielens/1m/>
Archive: <https://files.grouplens.org/datasets/movielens/ml-1m.zip>

Unzip and move the `ml-1m/` folder here. Only `ratings.dat` is required for
training and evaluation; `movies.dat` is additionally read by
`figures/plot_manifold_genre_territories_ml1m.py`, which colours the manifold
by primary genre.

`ratings.dat` is the classic `::`-separated
`UserID::MovieID::Rating::Timestamp` format, read by
`load_movielens_ratings()` in `ncf/ncf_experiment_pipeline.py`.

## 2-5. Amazon Reviews 2023 — four categories

Source: McAuley Lab, <https://amazon-reviews-2023.github.io/>
Variant used: **benchmark / 5-core / rating-only CSV** (columns
`user_id,parent_asin,rating,timestamp`), documented at
<https://amazon-reviews-2023.github.io/data_processing/5core.html>.

| category | target directory | file |
|---|---|---|
| Beauty and Personal Care | `data/amazon_beauty/` | `Beauty_and_Personal_Care.csv` |
| Arts, Crafts and Sewing | `data/amazon_arts_crafts_sewing/` | `Arts_Crafts_and_Sewing.csv` |
| Grocery and Gourmet Food | `data/amazon_grocery_gourmet_food/` | `Grocery_and_Gourmet_Food.csv` |
| Office Products | `data/amazon_office_products/` | `Office_Products.csv` |

Download template:

```
https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/benchmark/5core/rating_only/<CATEGORY>.csv.gz
```

for example:

```bash
cd data
for c in Beauty_and_Personal_Care:amazon_beauty \
         Arts_Crafts_and_Sewing:amazon_arts_crafts_sewing \
         Grocery_and_Gourmet_Food:amazon_grocery_gourmet_food \
         Office_Products:amazon_office_products ; do
  name=${c%%:*}; dir=${c##*:}
  mkdir -p "$dir"
  curl -L -o "$dir/$name.csv.gz" \
    "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/benchmark/5core/rating_only/$name.csv.gz"
  gunzip -k "$dir/$name.csv.gz"
done
```

The loaders read the **decompressed** `.csv`; keeping the `.gz` alongside it is
harmless. Amazon timestamps are integer milliseconds while MovieLens uses
seconds — only the relative order matters downstream, so the two are handled
by the same code path.

## Subsampling

No dataset is used at full size. Every run subsamples to a
`max_users × max_items` scale (300×800, 1000×1200, 3000×2500, 6000×3000) by
keeping the most active users and then the most frequent items, in
`subsample_users_items()` of `shared/data_preparation_interactions.py`. The
scale is a parameter of each arm script, so the same downloaded files serve
every scale in the paper.
