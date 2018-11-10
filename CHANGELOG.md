# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

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