# Label Indeterminacy in AI & Law

This repository contains the code and datasets required to run the experiments detailed in the paper 'Label Indeterminacy in AI & Law', as accepted for the JURIX conference of 2025. 
Links to the full paper will be made available soon.

# Abstract:
Machine learning is increasingly used in the legal domain, where it typically operates retrospectively by treating past case outcomes as ground truth. However, legal outcomes are often shaped by human interventions that are not captured in most machine learning approaches. A final decision may result from a settlement, an appeal, or other procedural actions. This creates label indeterminacy: the outcome could have been different if the intervention had or had not taken place. We argue that legal machine learning applications need to account for label indeterminacy. Methods exist that can impute these indeterminate labels, but they are all grounded in unverifiable assumptions. In the context of classifying cases from the European Court of Human Rights, we show that the way that labels are constructed during training can significantly affect model behaviour. We therefore position label indeterminacy as a relevant concern in AI \& Law and demonstrate how it can shape model behaviour. 

# Repository:
## Notebooks:
- Balancing_data.ipynb: Details the data preprocessing required for the experiments and each of the label imputation methods.
- Experiment.ipynb: Used to run the experiment, includes training and testing a longformer for each method.
- Analysis.ipynb: Details the code used to create the figures and tables of the paper.

## Python files:
- echr.py: Functions to import and process the raw ECHR cases
- classifier.py: Functions used in the experiment
- analysis.py: Functions used in the analysis

## Sub-directories:
- datasets: The processed datasets and a link to the original dataset: https://echr-opendata.eu/download/
- results: The results of the experiment
- analysis: The plots generated from the analysis

