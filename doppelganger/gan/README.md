# DoppelGANger

This repo contains the code of DoppelGANger, a GAN to generate synthetic time series data. It is based on the [paper](http://arxiv.org/abs/1909.13403) "Generating High-fidelty, Synthetic Time Series Datasets with DoppelGANger". The original source code can be found [here](https://github.com/fjxmlzn/DoppelGANger)

The codes are tested with Python 3.7 and Tensorflow 2.3.0

---
## Dataset format
To train the DoppelGANger with your data, 3 files need to be prepared according to the format below:

* `data_train.npz`: A numpy `.npz` archive of the following three arrays:
	* `data_feature`: Training features (i.e. temporal data), in numpy float32 array format. The shape is [(number of training samples) x (maximum sequence length) x (total dimension of features)]. Categorical features are stored by one-hot encoding; for example, if a categorical feature has 3 possibilities, then it can take values between `[1., 0., 0.]`, `[0., 1., 0.]`, and `[0., 0., 1.]`. Each continuous feature should be **normalized** to `[0, 1]` or `[-1, 1]`before being fed into the model. The array is padded by zeros after the time series ends.
	* `data_attribute`: Training attributes (i.e. static data), in numpy float32 array format. The shape is [(number of training samples) x (total dimension of attributes)]. Categorical attributes are stored by one-hot encoding; for example, if a categorical attribute has 3 possibilities, then it can take values between `[1., 0., 0.]`, `[0., 1., 0.]`, and `[0., 0., 1.]`. Each continuous attribute should be **normalized** to `[0, 1]` or `[-1, 1]`before being fed into the model.
	* `data_gen_flag`: Flags indicating the activation of features, in numpy float32 array format. The shape is [(number of training samples) x (maximum length)]. 1 means the time series is activated at this time step, 0 means the time series is inactivated at this timestep.
* `data_feature_output.pkl`: A pickle dump of a list of `output.Output` objects, indicating the dimension, type, normalization of each feature.
* `data_attribute_output.pkl`: A pickle dump of a list of `output.Output` objects, indicating the dimension, type, normalization of each attribute.
 

Let's look at a concrete example. Assume that there are two features (a 1-dimension continuous feature normalized to [0,1] and a 2-dimension categorical feature) and two attributes (a 2-dimension continuous attribute normalized to [-1, 1] and a 3-dimension categorical attributes). Then `data_feature_output ` and `data_attribute_output ` should be:

```
data_feature_output = [
	Output(type_=CONTINUOUS, dim=1, normalization=ZERO_ONE, is_gen_flag=False),
	Output(type_=DISCRETE, dim=2, normalization=None, is_gen_flag=False)]
	
data_attribute_output = [
	Output(type_=CONTINUOUS, dim=2, normalization=MINUSONE_ONE, is_gen_flag=False),
	Output(type_=DISCRETE, dim=3, normalization=None, is_gen_flag=False)]
```

Note that `is_gen_flag` should always set to `False` (default).

Assume that there are two samples, whose lengths are 2 and 4, and assume that the maximum length is set to 4. Then `data_feature `, `data_attribute `, and `data_gen_flag ` could be:

```
data_feature = [
	[[0.2, 1.0, 0.0], [0.4, 0.0, 1.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
	[[0.9, 0.0, 1.0], [0.3, 0.0, 1.0], [0.2, 0.0, 1.0], [0.8, 1.0, 0.0]]]
	
data_attribute = [
	[-0.2, 0.3, 1.0, 0.0, 0.0],
	[0.2, 0.3, 0.0, 1.0, 0.0]]
	
data_gen_flag = [
	[1.0, 1.0, 0.0, 0.0],
	[1.0, 1.0, 1.0, 1.0]]
```

## Generating Synthetic Data  

A walkthrough of how synthetic data is generated is shown in the notebook `walkthrough_prism.ipynb`. It shows the complete steps starting from data cleaning, saving the data in the required format to loading in the generated data and converting it back to a `csv` format.

### Sample Data
Sets of sampled data are shown in `data` and `data_attr`. The original dataset `isaFull.tsv` is first cleaned to `ori_prism_cleaned.csv`.

### Data preparation
Code for data cleaning and preparation can be found in the `prism_prep` folder. `ori_prism_cleaned.csv` is transformed to the `pkl` and `npz` files.

### Train and Generate Data with DoppelGANger
Run `main.py` to generate synthetic data. There are a few optional command line arguments, which is the time-series sequence length, batch size, number of epochs, total samples to be generated, path to data and output path.

Example command: 
```
cd gan
python main.py --seq_len 130 --batch_size 64 --epochs 100 --total_generate_num_sample 1347 --path_to_data "data" --output_path "generated_data.npz"
```
To view command usage help, type in `python main.py --help`
The GAN then outputs a `npz` file.

### Evaluating Synthetic Data
The `evaluations` folder contains the code for evaluating synthetic data generated. A sample of how the evaluation methods are used to assess the distribution, fidelity and usefulness of the synthetic data are shown in `prism_evaltest.ipynb`. The output from the GAN is converted to `gen_orism_int_xx.csv` which is used for evaluation purposes.

### Sample Output
The intermediate csv from above is then converted to `gen_prism_final_xx.csv` as the final form of output. This csv has the same column names (for the columns that are selected) and format as the original dataset given initially.

TODO: Add license