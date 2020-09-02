from output import OutputType, Output, Normalization
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def renormalize_per_sample(data_feature, data_attribute, data_feature_outputs,
                           data_attribute_outputs, gen_flags,
                           num_real_attribute):
    """
    -gets correct dimension for generated data attribute (num_ori_attributes) 
    from num_ori_attributes + num_created_attributes
    -renormalizes data_feature and replaces part of the time-series that has ended with zeros

    Args:
    data_feature: generated data features (num_samples, seq_len, num_features)
    data_attribute: generated data attributes (num_samples, num_ori_attributes + num_created_attributes)
    data_feature_outputs: list that describes generated feature output type, dimension, normalization
    data_attribute_outputs: list that describes generated attribute output type, dimension, normalization
    gen_flags: generation flag for generated data
    num_real_attribute: number of original attributes

    Returns:
    data_feature: generated data feature taking into account gen flags
    data_attribute: generated data attribute of the original dimension
    """

    attr_dim = 0
    for i in range(num_real_attribute):
        attr_dim += data_attribute_outputs[i].dim
    attr_dim_cp = attr_dim

    fea_dim = 0
    for output in data_feature_outputs:
        # for each continous feature
        if output.type_ == OutputType.CONTINUOUS:
            for _ in range(output.dim):
                max_plus_min_d_2 = data_attribute[:, attr_dim]
                max_minus_min_d_2 = data_attribute[:, attr_dim + 1]
                attr_dim += 2

                max_ = max_plus_min_d_2 + max_minus_min_d_2
                min_ = max_plus_min_d_2 - max_minus_min_d_2

                max_ = np.expand_dims(max_, axis=1)
                min_ = np.expand_dims(min_, axis=1)

                if output.normalization == Normalization.MINUSONE_ONE:
                    data_feature[:, :, fea_dim] = \
                        (data_feature[:, :, fea_dim] + 1.0) / 2.0

                data_feature[:, :, fea_dim] = \
                    data_feature[:, :, fea_dim] * (max_ - min_) + min_

                fea_dim += 1
        #if feature is discrete, add feature dim
        else:
            fea_dim += output.dim

    tmp_gen_flags = np.expand_dims(gen_flags, axis=2) # (num_sample, seq_len, 1)
    data_feature = data_feature * tmp_gen_flags # (num_sample, seq_len, num_features)

    # get back only the original attributes
    data_attribute = data_attribute[:, 0: attr_dim_cp] # (num_sample, 1)

    return data_feature, data_attribute


def normalize_per_sample(data_feature, data_attribute, data_feature_outputs,
                         data_attribute_outputs):
    """
    -adds 2 extra attributes ((max +- min)/2) for each feature for each sample 
    to the original attribute
    -normalizes data feature
    Args:
    data_feature: original data feature
    data_attribute: original data attribute
    data_feature_outputs: list that describes original feature output type, dimension, normalization
    data_attribute_outputs: list that describes original attribute output type, dimension, normalization

    Returns:
    data_feature: normalized data feature
    data_attribute: original data attribute + newly created attributes
    data_attribute_outputs: list that describes original + created attribute output type, dimension, normalization
    real_attribute_mask: boolean list specifying if attributes are orginal or newly created
    """

    # assume all samples have maximum length
    # get max and min for each feature for each sample
    data_feature_min = np.nanmin(data_feature, axis=1) # (total_sample, num_features)
    data_feature_max = np.nanmax(data_feature, axis=1)

    additional_attribute = []
    additional_attribute_outputs = []

    dim = 0
    for output in data_feature_outputs:
        #for each feature, we create 2 extra attributes with the min & max
        if output.type_ == OutputType.CONTINUOUS:
            for _ in range(output.dim):
                max_ = data_feature_max[:, dim] # (total_sample, )
                min_ = data_feature_min[:, dim]

                additional_attribute.append((max_ + min_) / 2.0)
                additional_attribute.append((max_ - min_) / 2.0)
                additional_attribute_outputs.append(Output(
                    type_=OutputType.CONTINUOUS,
                    dim=1,
                    normalization=output.normalization,
                    is_gen_flag=False))
                additional_attribute_outputs.append(Output(
                    type_=OutputType.CONTINUOUS,
                    dim=1,
                    normalization=Normalization.ZERO_ONE,
                    is_gen_flag=False))

                max_ = np.expand_dims(max_, axis=1)
                min_ = np.expand_dims(min_, axis=1)

                data_feature[:, :, dim] = \
                    (data_feature[:, :, dim] - min_) / (max_ - min_ + 1e-7)
                if output.normalization == Normalization.MINUSONE_ONE:
                    data_feature[:, :, dim] = \
                        data_feature[:, :, dim] * 2.0 - 1.0

                dim += 1
        else:
            dim += output.dim

    # create a mask for original attribute and attributed we just created
    real_attribute_mask = ([True] * len(data_attribute_outputs) +
                           [False] * len(additional_attribute_outputs))

    additional_attribute = np.stack(additional_attribute, axis=1) # (num_sample, num_continuous_features * 2)
    data_attribute = np.concatenate(
        [data_attribute, additional_attribute], axis=1) #(num_sample, num_continuous_feature * 2 + num_ori_attribute)
    data_attribute_outputs.extend(additional_attribute_outputs)

    return data_feature, data_attribute, data_attribute_outputs, \
        real_attribute_mask


def add_gen_flag(data_feature, data_gen_flag, data_feature_outputs,
                 sample_len):

    """
    -adds generation flags to the end of original data features
    -adds an additional output to the data_feature_outputs list
    Args:
    data_feature: original data feature
    data_gen_flag: original data gen flag
    data_feature_outputs: list that describes original feature output type, dimension, normalization
    sample_len: max sequence length of time series

    Returns:
    data_feature: original data feature + gen flag
    data_feature_outputs: original data_feature_output + output type, dimension and normalization of gen flag
    """

    for output in data_feature_outputs:
        if output.is_gen_flag:
            raise Exception("is_gen_flag should be False for all"
                            "feature_outputs")

    if (data_feature.shape[2] !=
            np.sum([t.dim for t in data_feature_outputs])):
        raise Exception("feature dimension does not match feature_outputs")

    if len(data_gen_flag.shape) != 2:
        raise Exception("data_gen_flag should be 2 dimension")

    num_sample, length = data_gen_flag.shape

    data_gen_flag = np.expand_dims(data_gen_flag, 2) # (num_sample, seq_len, 1)

    data_feature_outputs.append(Output(
        type_=OutputType.DISCRETE,
        dim=2,
        is_gen_flag=True))

    shift_gen_flag = np.concatenate(
        [data_gen_flag[:, 1:, :],
         np.zeros((data_gen_flag.shape[0], 1, 1))],
        axis=1)  # (num_samples, seq_len, 1)
    if length % sample_len != 0:
        raise Exception("length must be a multiple of sample_len")
    data_gen_flag_t = np.reshape(
        data_gen_flag,
        [num_sample, int(length / sample_len), sample_len]) # (num_sample, 1, seq_len)
    data_gen_flag_t = np.sum(data_gen_flag_t, 2)
    data_gen_flag_t = data_gen_flag_t > 0.5
    data_gen_flag_t = np.repeat(data_gen_flag_t, sample_len, axis=1)
    data_gen_flag_t = np.expand_dims(data_gen_flag_t, 2)
    # add the gen_flag and inverse of gen_flag to data_feature
    data_feature = np.concatenate(
        [data_feature,
         shift_gen_flag,
         (1 - shift_gen_flag) * data_gen_flag_t],
        axis=2) # (num_sample, seq_len, num_features + 2)

    return data_feature, data_feature_outputs
