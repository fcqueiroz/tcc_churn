# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## 0.2 - 2018-11-16
### Added
- It got easy to run Jupyter Notebooks inside a container using the same development environment. Check how in the [README.md](README.md)
- The transformation of raw data to usable data for the machine learning algorithm now happens inside a [sklean Pipeline](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html)
- Train/Test splits temporally, which means the subset reserved for fitting the model contain events that occur chronologically before the events being used in the subset for validating the model performance.
- Hyperparameters optimization is now provided through [Exhaustive Grid Search](https://scikit-learn.org/stable/modules/grid_search.html)
- The project runs as a installed package!
- The main script now provides 4 functions (train, test, save and load model) that can be run interactively or in batch mode.
- The script tells how much time was spent on each execution phase 

### Changed
- The source code provided to the docker container through a [volume](https://docs.docker.com/storage/volumes/) and no longer is included inside the docker image with the `COPY` command. 
- The docker image is built on top of a conda base image and configured with the same package versions in my development [environment](environment.yml)

### Fixed
- The target variable now considers that during the 30 days that follow the last customer order, the target class is unknown. So a 30 days censored data window was implemented between the moment the target is calculated and the latest data point available for training the model.  

### Removed
- Makefile
## 0.1 - 2018-11-10
### Added
- Project first public version!
- [README](README.md) with a brief explanation about this project and a ~~hopefully comprehensive~~ guide to run this project on your own machine
- Informations about the [project structure](references/project_structure.md)
- [Work plan](references/Work Plan pt-BR.pdf) for my Bachelor Final Project in portuguese (sorry!)
- GNU GPL3 [LICENSE](LICENSE)
- Analysis reproducibility through [Docker](https://www.docker.com/). If it works on my machine, then it works on yours :)
- Working code to produce two baseline machine learning results using as a (dummy) estimator the most frequent value and comparing with an unoptimized Decision Tree Classifier.
- [Makefile](Makefile) for training and testing the model with a single line of code and caching intermediate steps to speed the whole data pipeline.