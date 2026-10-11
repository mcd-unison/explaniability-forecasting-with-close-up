# Explainability for forecasting task with close-up option

This repository contains the implementation, notebooks, trained model artifacts, and result tables for a forecasting explainability study. The project focuses on a sequence-to-sequence LSTM forecasting model and a hierarchical decomposition framework used to explain how different input components contribute to predictions.

## Project structure

```text
.
├── README.md
├── .gitignore
├── requirements.txt                            # Python dependencies
├── tree_explainer.py                           # core explainability engine
├── exp_model_1.py                              # main decomposition definition
├── exp_model_1_raw.py                          # raw decomposition variant
├── exp_model_1_tri.py                          # tri decomposition variant
├── exp_model_1_wide.py                         # wide decomposition variant
├── evaluations.ipynb                           # evaluation notebook
├── example_general_use.ipynb                   # general usage example
├── example_SHAPE_load.ipynb                    # notebook for loading explainability outputs
├── tables_results.ipynb                        # notebook with result tables
├── data/
│   ├── calendar_weather_holiday_in_advance/
│   │   ├── model_calendar_wather_holiday_in_advance.h5
│   │   └── model_info.txt
│   ├── results/
│   │   └── paper/
│   │       ├── base/
│   │       ├── biased/
│   │       ├── filter/
│   │       ├── raw/
│   │       ├── tri/
│   │       ├── wide/
│   │       └── tables/
│   └── samples/
└── ...
```

## Main files

- tree_explainer.py: contains the core explainability implementation. It defines the hierarchical decomposition logic, local explanation flow, and utilities for model-based contribution analysis.
- exp_model_1.py: explainability model with closeups established with the subject matter experts.
- exp_model_1_raw.py: raw explainability model without closeups.
- exp_model_1_tri.py: explainability model with a branching factor b = 3.
- exp_model_1_wide.py: explainability model with a branching factor b = 5.
- evaluations.ipynb: notebook for model evaluation and analysis.
- example_general_use.ipynb: notebook showing the general usage flow of the explainability pipeline.
- example_SHAPE_load.ipynb: notebook for loading and using explainability output data and related example workflows.
- tables_results.ipynb: notebook for inspecting and summarizing results tables and output metrics.

## Data and model

The trained forecasting model currently available in the project is:

- data/calendar_weather_holiday_in_advance/model_calendar_wather_holiday_in_advance.h5

The metadata file at data/calendar_weather_holiday_in_advance/model_info.txt reports the following verified shapes:

- Training set: 2010-01-01 00:00:00 to 2023-08-06 23:00:00
- X_encoder_train tensor size: (118969, 180, 15)
- X_decoder_train tensor size: (118969, 36, 14)
- y_train tensor size: (118969, 36, 1)

The data/results/paper folder contains result artifacts organized by experiment or variant, including base, biased, filter, raw, tri, wide, and tables.

## Dependencies

The project dependencies are listed in requirements.txt and include:

- numpy
- pandas
- scikit-learn
- jupyter
- notebook
- tensorflow==2.15.1
- keras==2.15.0
- matplotlib
- plotly
- openpyxl
- kaleido
- tqdm

## Installation

To set up the environment and install the dependencies, run:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```



