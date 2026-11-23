# SpaceX Falcon 9 First-Stage Landing Prediction

Capstone project for the **IBM Data Science Professional Certificate** (Coursera).

SpaceX advertises Falcon 9 launches at roughly $62M, against $165M+ for competitors, and most of that saving comes from reusing the first stage. If you can predict whether the first stage will land, you can estimate the cost of a launch. This project builds that prediction end to end: collecting the data, wrangling it, EDA, interactive visual analytics, and classification models.

## Pipeline

| # | Step | Notebook / file |
|---|------|-----------------|
| 1 | Data collection via the SpaceX REST API (Falcon 9 only, missing payloads imputed with the mean) | `01_data_collection_api.ipynb` |
| 2 | Data collection by scraping Wikipedia launch tables with BeautifulSoup | `02_data_collection_webscraping.ipynb` |
| 3 | Data wrangling: launches per site and orbit, plus a binary landing label (`Class`) built from the mission outcome | `03_data_wrangling.ipynb` |
| 4 | EDA with SQL (SQLite) | `04_eda_sql.ipynb` |
| 5 | EDA with visualisation and feature engineering (one-hot encoding) | `05_eda_visualization.ipynb` |
| 6 | Launch site geography with Folium: site markers, success/failure clusters, distances to the coast, railways, highways and cities | `06_launch_sites_folium.ipynb` |
| 7 | Interactive Plotly Dash dashboard: success pie chart by site, payload vs. outcome scatter with a payload range slider | `07_spacex_dash_app.py` |
| 8 | Classification models with GridSearchCV (10-fold) | `08_ml_prediction.ipynb` |

The final report is in `capstone_report.pdf`.

## Results

90 Falcon 9 launches, 80/20 train/test split (18 test launches).

| Model | Best params | CV accuracy | Test accuracy |
|-------|-------------|-------------|---------------|
| Logistic Regression | `C=0.01, l2, lbfgs` | 0.846 | 0.833 |
| SVM | `C=1.0, gamma=0.032, sigmoid` | 0.848 | 0.833 |
| Decision Tree | `entropy, max_depth=8, min_samples_leaf=4` | **0.875** | 0.833 |
| KNN | `k=10, p=1` | 0.848 | 0.833 |

- All four models tie on the test set: 15 of 18 correct. The confusion matrices show the errors are false positives, meaning landings predicted that didn't happen.
- The decision tree has the best cross-validation score. With only 18 test samples, though, the models can't really be told apart, and the tie is the honest headline.
- EDA findings:
  - The landing success rate rises steadily from 2013 to 2020.
  - Success rate by site: KSC LC-39A and VAFB SLC 4E reach 77%, CCAFS LC-40 60%.
  - VAFB SLC 4E has no heavy-payload launches (> 10,000 kg).
  - Heavy payloads land more often in Polar, LEO and ISS orbits. GTO shows no clear pattern.

## Data

| File | Contents |
|------|----------|
| `dataset_part_2.csv` | Cleaned launch records with the `Class` label |
| `dataset_part_3.csv` | One-hot encoded feature matrix used for modelling |
| `spacex_web_scraped.csv` | Wikipedia launch table |
| `spacex_launch_geo.csv` | Launch outcomes with site coordinates (Folium) |
| `spacex_launch_dash.csv` | Input for the Dash app |
| `Spacex.csv` | Table for the SQL EDA |

## Running

```bash
pip install pandas numpy scikit-learn matplotlib seaborn folium plotly dash requests beautifulsoup4
jupyter lab                 # notebooks
python 07_spacex_dash_app.py   # dashboard at http://127.0.0.1:8050
```

`07_spacex_dash_app.py` uses the pre-2.0 imports `dash_html_components` and `dash_core_components`. On Dash ≥ 2, change them to `from dash import html, dcc`.

## Credits

Lab templates © IBM Corporation (IBM Skills Network). Solutions by BastiaanMatt.
