# From Single-Visit to Multi-Visit Image-Based Models: Single-Visit Models are Enough to Predict Obstructive Hydronephrosis

## Context

Most image-based models being developed today rely on 2D convolutional architectures (CNNs). These models are trained with the hope that one day, a patient can have a scan/image collected, pass it into the model, and produce a prediction related to some outcome.

However, hospitals/institutions may collect a patient's medical images across multiple hospital visits. It becomes a question whether or not data from previous hospital visits can be used in the model's prediction.

## Approach

We attempt to answer this question in the context of predicting renal obstruction using kidney ultrasounds.

We adapt the original 2D Siamese CNN to handle multi-visit (spatiotemporal) inference via:
* (Naive) Average Prediction
* Convolutional (Conv.) Pooling
* Long Short Term Memory (LSTM)
* Temporal Shift Module (TSM)

## Result

We evaluate models on 2 internal datasets collected at SickKids (test set and silent trial) and 2 external datasets.

We find no **significant difference** in AUROC/AUPRC between the single-visit baseline and the multi-visit models.

## Conclusion

This evidence suggests that the model is very flexible with respect to a patient's past and current data. Future deployment of the model in the hospital may include using data from any of a patient's past or current ultrasounds.

```
├── models/                    # Moduels for models
│   ├── baseline.py            # (Baseline) 2D Siamese CNN
│   ├── average_prediction.py  # Avg. Prediction
│   ├── conv_pooling.py        # Conv. Pooling
│   ├── lstm.py                # Long Short Term Memory
│   ├── tsm.py                 # Temporal Shift Module
│   └── tsm_blocks.py          # Contains external modules implementing Temporal Shift
├── op/
│   ├── model_training.py      # Used to train all models
│   └── grid_search.py         # Used to perform hyperparameter randomized grid search
└── utilities/
    ├── bootstrap_ci.py        # Used to bootstrap confidence interval on metrics, given model output
    ├── custom_logger.py       # Customized PyTorch Lightning CSVlogger
    ├── data_visualizer.py     # Used to view data
    ├── dataset_prep.py        # Main data loading/preprocessing module
    ├── kornia_augmentation.py # Custom module for image augmentations
    └── results.py             # [LEGACY CODE] for calculating metrics
```

