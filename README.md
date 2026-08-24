# Explaniability for forecasting task with close-up option

This repository contains the code, data, and models for the paper "Explaniability for forecasting task with close-up option".

**Olivia Gutú** (olivia.gutu@unison.mx) and **Julio Waissman** (julio.waissman@unison.mx)

## Project Structure

```
├── README.md                                    # This file with project description
├── requirements.txt                             # Python package dependencies
├── time_explainer.py                            # Main module with model implementation
├── example_general_use.ipynb                    # Notebook with general usage examples
├── example_SHAPE_load.ipynb                     # Notebook with SHAPE loading and usage examples
└── data/
    └── calendar_weather_holiday_in_advance/
        ├── model_calendar_wather_holiday_in_advance.h5  # Trained model (HDF5 format)
        ├── model_info.txt                       # Model metadata and configuration
        ├── X_encoder_train.npy                  # Encoder input data for training
        ├── X_encoder_test.npy                   # Encoder input data for testing
        ├── X_decoder_train.npy                  # Decoder input data for training
        ├── X_decoder_test.npy                   # Decoder input data for testing
        ├── y_train.npy                          # Training targets/labels
        └── y_test.npy                           # Testing targets/labels
```

## Installation

To set up the environment and install the required dependencies, run:

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
pip3 install -r requirements.txt
```

