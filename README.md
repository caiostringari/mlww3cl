# Improving WW3 Results with Machine Learning

This repository contains the models used in the paper: *Improving WaveWatchIII Outputs with Machine Learning* currently under review in *Coastal Engineering*.

# Data

The  data is to heavy to be hosted by Github. Please use the links below to download it.

Google drive link here.


## 1. Model Architecture

Three models are architectures are available:

### a) `MLP_PAR_HTD`

Multilayer Perceptron (MLP) trained using integrated wave parameters (`Hm0`, `Tm01`, `Tm02`, `Dm`, `Spd`) and wind (`U10`, `V10`) as inputs.

### b) `MLP_SPC_HTD`

Multilayer Perceptron (MLP) trained using the flattened wave spectrum as inputs.

### c) `CNN_SPC_HTD`

Convolutional Neural Network (CNN) based on the `VGG16` architecture and using the wave spectrum in two-dimensional form.

The neural nets look something like this:

![](figures/Neural_Nets.png)


## 2. Training

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1JiNAzjf1RRQDTbYDpIfpez4g8rdZNYSv?usp=sharing) **\|** [![Jupyter Notebook](https://raw.githubusercontent.com/jupyter/design/master/logos/Badges/nbviewer_badge.svg)](notebook/train.ipynb)

Training the models is done using the same script: ```train.py```. For help, do:

```bash
python train.py --help
```

<details>
  <summary> Options are (click to expand): </summary>

  - `-i, --data`: Input data (.csv).

  - `-m, --model`:  Model name.

  - `-t, --type`: Model type. Possible choices are: `cnn_spectral`, `mlp_parametric` or `mlp_spectral`.

  - `--logdir`: Logging directory for `Tensorboard`.

  - `--random-state`: Random state. Used for reproducibility.

  - `--test-size`:  Test set size. Default is 0.3.

  - `--layers`: Number of layers for MLP models. Default is 3.

  - `-neurons`: Number of neurons per layer for MLP models. Default is 256.

  - `--learning-rate`: Learning rate for ADAM. Default is 10E-6.

  - `--dropout`: Dropout rate. Default is 0.25.

  - `--epochs`: Number of training epochs. Default is 128.

  - `--batch-size`: Batch size. Default is 2048.

  - `--stratify`:  Use class stratification (by location name). Default is True.

  - `--input-size`: 2d-spectrum size for CNN models. Default is 32x24.

</details>
<br/>

To obtain the results seen in the paper do:

Manually:

```bash
python train.py --type "mlp_parametric" --model "MLP_PAR_HTD" -i "data/wave_data.csv" --logdir  "logs/MLP_PAR" --epochs 2048 --layers 2 --neurons 512 --learning-rate 0.0001 --random-state 42 --test-size 0.25
```

```bash
python train.py --type "mlp_spectral" --model "MLP_SPC_HTD" -i "data/wave_data.csv" --logdir  "logs/MLP_SPC" --layers 3 --neurons 128 --epochs 1024 --learning-rate 0.0001 --random-state 42 --test-size 0.25
```

```bash
python train.py --type "cnn_spectral" --model "CNN_SPC_HTD" -i "data/wave_data.csv" --logdir "logs/CNN_SCP" --epochs 256 --batch-size 128
```

## 3. Evaluation


### 3.1. Training curves

#### a) `MLP_PAR_HTD`
```bash
tensorboard --logdir "logs/MLP_PAR/MLP_PAR_HTD/"
```

#### b) `MLP_SPC_HTD`
```bash
tensorboard --logdir "logs/MLP_SPC/MLP_SPC_HTD/"
```

#### c) `CNN_SPC_HTD`
```bash
tensorboard --logdir "logs/CNN_SCP/CNN_SPC_HTD/"
```


### 3.2. Metrics

TABLE HERE

To reproduce the results in the paper, do:

#### a) `MLP_PAR_HTD`
```bash
python metrics.py -i "data/predictions_mlp_par.csv" -o "data/metrics_mlp_par.csv"
```

#### b) `MLP_SPC_HTD`
```bash
python metrics.py -i "data/predictions_mlp_spc.csv" -o "data/metrics_mlp_spc.csv"
```

#### c) `CNN_SPC_HTD`
```bash
python metrics.py -i "data/predictions_mlp_par.csv" -o "data/metrics_mlp_spc.csv"
```


### 3.3. Timeseries

```bash
python timeseries.py -i "data/MLP_PAR/predictions.csv" -ID "BCurau" -o "metrics_mlp_par.png" --start "2013-03-01 00:00:00" --duration "120"
```


<!-- [Jupyter notebook](notebooks/01_MLP_with_integrated_parameters.ipynb) **|**
[Colab notebook](https://drive.google.com/open?id=1__yKUuyMvFgGfN9jrqbMtwZyCUg0neXh)

This models uses integrated parameters (e.g., Hs, Tp, Dp, Winds) to correct poorly predicted wave directions by WW3.

The model is a multilayer perceptron (MLP) with 2 hidden layers and 512 hidden units per layer.

![](plots/fig_mlp_wavepar.png)

[Tensorflow architecture](plots/arc_mlp_wavepar.png)

This model achieves a r-score of 0.67 after 1024 epochs on test data.

![](plots/results_mlp_wavepar.png)

## 2. MLP with Spectral Data

[Jupyter notebook](notebooks/02_MLP_with_spectral_data.ipynb) **|**
[Colab notebook](https://drive.google.com/open?id=1yjB3YypUCGyb0haz9OimRK07drYhf8Bf)

This model uses the flattened wave spectra to correct poorly predicted wave directions by WW3.

The model is a multilayer perceptron with 2 hidden layers and 64 hidden units per layer.

![](plots/fig_mlp_spc.png)

[Tensorflow architecture](plots/arc_mlp_spc.png)

This model achieves a r-score of 0.87 after 1024 epochs on test data.

![](plots/results_mlp_spc.png)

## 3. CNN with Spectral Data



This models uses the wave spectrum (in its 2d form) to correct poorly predicted wave directions by WW3.

The model is a convolutional neural network (CNN) build based upon the VGG network. There are 3 VVG8 blocks with increasing number of convolutional filters and two fully connected layers.

![](plots/fig_cnn_spc.png)

[Tensorflow architecture](plots/arc_cnn_spc.png)

This model achieves a r-score of 0.93 after 64 epochs on test data.

Note: this model is too heavy to run on CPU.

![](plots/results_cnn_spc.png)

## 4. Results:

### 4.1 Predictions on known timeseries data

[Jupyter notebook](notebooks/04_comparison_with_known_data.ipynb)

How do the models fare predicting data that they have seen? very well.

![](plots/comparison_known_data.png)

### 4.2 Predictions on unknown timeseries data

How do the models fare predicting data that they have never seen? Not too well.

![](plots/comparison_new_data.png)

### 4.3 Summary table (test+train data)

| Model     | r2   | RMSE  | MAE  | Bias  | Slope | Intercept |
|-----------|------|-------|------|-------|-------|-----------|
| WW3       | 0.36 | 17.52 | 0.36 | 9.80  | 0.62  | 84.29     |
| MLP (par) | 0.77 | 10.62 | 0.77 | 0.25  | 0.74  | 63.82     |
| MLP (spc) | 0.90 | 7.05  | 0.90 | -0.06 | 0.86  | 36.36     |
| CNN       | 0.93 | 5.68  | 0.93 | 0.48  | 0.94  | 14.95     |


## 5. Pre-trained models

1. [MLP with integrated parameters](pre-trained/MLP.h5)

2. [MLP with spectral data](pre-trained/SPC_MLP.h5)

3. [CNN with spectral data](pre-trained/CNN.h5) -->
