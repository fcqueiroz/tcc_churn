# Project Structure

ref: https://drivendata.github.io/cookiecutter-data-science/

## Directory structure

├── CHANGELOG.md       <- Changelog for documenting all notable changes to this project  
├── Dockerfile         <- Dockerfile for building a docker image  
├── LICENSE  
├── Makefile           <- Makefile with commands like `make train_model` or `make test_model`  
├── README.md          <- The top-level README for developers using this project.  
├── data  
│   ├── external       <- Data from third party sources.  
│   ├── interim        <- Intermediate data that has been transformed.  
│   ├── processed      <- The final, canonical data sets for modeling.  
│   └── raw            <- The original, immutable data dump.  
│  
├── models             <- Trained and serialized models, model predictions, or model summaries  
│  
├── notebooks          <- Jupyter notebooks. Naming convention here is release (version) number,  
│                         story numbers, and a short `-` delimited description, e.g.  
│                         `01_st2_customer_recency`.  
│  
├── references         <- Data dictionaries, manuals, and all other explanatory materials.  
│  
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.  
│  
└── src                <- Source code for use in this project.  
    ├── data           <- Scripts to download or generate data  
    │   └── make_dataset.py  
    │  
    ├── features       <- Scripts to turn raw data into features for modeling  
    │   └── build_features.py  
    │  
    ├── models         <- Scripts to train models and then use trained models to make  
    │   │                 predictions  
    │   ├── predict_model.py  
    │   └── train_model.py  
    │  
    └── visualization  <- Scripts to create exploratory and results oriented visualizations  
        └── visualize.py  