# TimeGAN

This repo contains the code for TimeGAN, a GAN to generate time-series data.

Tested with Python 3.7 and Tensorflow 2.3

### To Run TimeGAN

#### Example 
```
cd merge
python main_timegan.py --data_name prism --seq_len 10 --module gru --hidden_dim 3 --num_layer 3 --iteration 500 --batch_size 32 --loss bce
```

#### command inputs:
- data_name: prism
- seq_len: sequence length of time series
- module: gru, lstm
- hidden_dim: hidden dimensions
- num_layer: number of layers of model
- iteration: number of epochs to train for
- batch_size: batch size of data
- loss: bce, wgan_gp

#### Notes

`main_timegan.py` runs `timegan.py` which currently only works with fixed sequence length. For the prism dataset, patients visit length are capped and fixed at 10 visits. Patients with less than 10 visits are excluded.

`timegan_padding.py` is a version of timegan that runs with imputed prism data (dates where patients did not make a visit are imputed and these rows are masked when fed into the model). It works with `500_imputed_patients.csv`. `timegan_padding.py` currently cannot be run through `main_timegan.py`.

TODO: Add reference


