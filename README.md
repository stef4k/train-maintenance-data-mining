# Train Maintenance Data Mining
Data mining application of real-world dataset to improve and automate train maintenance.

Check the [final presentation](https://github.com/stef4k/train-maintenance-data-mining/blob/main/final_presentation.pdf)

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
