# Train Maintenance Data Mining

Group project for a data mining course in the BDMA master's, built on a real-world data challenge set by the Belgian railway operator SNCB-NMBS. Team: Jorge Ignacio del Río Sánchez, Kristóf Balázs, Sebastian Alberto Neri Pérez and Stefanos Kypritidis.
 
## Overview
 
Modern trains report the state of their subsystems (doors, pantographs, traction, HVAC, etc.) as information and fault codes to an on-board computer. SNCB provided 1,011 labelled technical incidents. Each one comes with the sequence of event codes reported by the vehicles involved, from about four hours before to one hour after the incident, along with the train speed and the state of the AC/DC power switches at each event. The task was to predict the incident type (12 classes) from these sequences and to find event patterns linked to specific incident types. Three things made this hard: the data is small, the classes are heavily imbalanced (the largest has around 300 incidents while several have only a handful), and the 917 distinct event codes are anonymised, which limits domain reasoning.
 
We tackled the problem in three ways. First, we replicated the method from SNCB's own [paper](https://arxiv.org/abs/2408.10288): filtering event codes by class specificity, mining frequent event patterns with FP-Growth, and classifying with Naive Bayes. Memory limits restricted us to one- and two-event patterns. Second, we treated each event sequence as a sentence of "words" and ran a brute-force benchmark: 3 representations (Count, TF-IDF, Word2Vec) × 7 oversampling options (SMOTE variants, ADASYN, random oversampling, none) × 10 scikit-learn classifiers gave 210 combinations, each evaluated with stratified 5-fold cross-validation. On top of these we built an ensemble whose weights were tuned with differential evolution. Third, we built a custom scikit-learn pipeline that adds features the text approach ignores. These are the last n events before the incident, the average speed over the last k events, the power-switch states just before the incident, and a flag for an emergency-braking code that we identified through a deceleration analysis in the exploratory phase. The pipeline also drops event codes that appear in almost every incident. Feature parameters, ADASYN oversampling and the gradient boosting classifier were tuned jointly with RandomizedSearchCV over 1,500 iterations.
 
The custom pipeline gave the best results: 0.71 weighted F1 in cross-validation, and 69.2% F1 / 70% accuracy on the held-out test set. That compares with 0.54 for the replicated paper method and 0.16 for a dummy baseline. Simple count features worked better than TF-IDF and Word2Vec, which suggests that which codes appear, and how often, carries most of the signal, more than their order. The engineered features added a modest gain over the best text model (0.69). The 210-model ensemble did not beat the best single model.
 
There is still a clear gap to SNCB's internal model, which they report at around 85% F1 on their own setup (a different time window and evaluation, so not directly comparable). The weighted F1 is driven by the larger classes. The smallest incident types remain essentially unpredictable with this little data. Exhaustive frequent-pattern mining or graph-based models would also need more memory and compute than we had available.
 
## Presentation and challenge description
 
- [Final presentation (PDF)](final_presentation.pdf)
- [Problem description by SNCB (PDF)](problem_description.pdf)
- [SNCB paper replicated in this project (arXiv)](https://arxiv.org/abs/2408.10288)
## Results at a glance
 
| Approach | CV weighted F1 | CV accuracy |
|---|---|---|
| Dummy classifier | 0.16 | 0.19 |
| Paper method (FP-Growth + Naive Bayes) | 0.54 | 0.56 |
| Best text model (Count vectorizer + gradient boosting) | 0.69 | 0.70 |
| Custom feature pipeline + gradient boosting | 0.71 | 0.73 |
 
On the held-out test set, the custom pipeline reaches 69.2% F1 and 70% accuracy.

## 📂 Project Structure

The repository contains the following key components:

### 🗂️ Notebooks
- **`exploratory_analysis.ipynb`**
  - Objective: Exploratory Data Analysis (EDA) of train incident dataset.
  - Tasks:
    - Overview of dataset structure (columns, data types, and missing values).
    - Visualizations of key features (e.g., incident types, events,  locations, time).
    - Emergency braking analysis

- **`research_paper_method.ipynb`**
  - Objective: Replicate the SNCB classification algorithms and steps described in the [paper](https://arxiv.org/pdf/2408.10288?)
  - Tasks:
    - Filter out event codes based on $r$ metric
    - Use FP-Growth to identify the most frequent sequences of events
    - Add binary columns for each frequent event sequence for each incident row
    - Fine-tune parameters of Naive Baynes and calculate f1-scores.

- **`text_classification.ipynb`**
  - Objective: Combine and test multiple embedding representations and predictive models to find the best ones.

- **`attribute_classification_randomcv_pipeline.ipynb`**
  - Objective: Improve the best text classification model by feature engineering
  - Tasks:
    - Create custom pipeline that creates new attributes from events sequence, train speed and pantograph states
    - Use randomsearchCV to find the best combination of parameters of those new attributes, oversampling technique and classifier
   
- **`visuals.ipynb`**
  - Objective: Compare and evaluate the cross-validation f1-scores and accuracies of the best models in a barplot
