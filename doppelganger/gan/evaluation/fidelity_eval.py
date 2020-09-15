import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from statsmodels.tsa.stattools import acf
import datetime as dt
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA


def visualization (ori_data, generated_data, analysis, syn_name):
    """Using PCA or tSNE for generated and original data visualization.
    (Originally from tGAN code, updated by Aisha to allow ori_data and generated_data to have different time_sequence_length)

    Args:
    - ori_data: original data
    - generated_data: generated synthetic data
    - analysis: tsne or pca
    - syn_name: the name of the synthetic generator
    """  
    # Analysis sample size (for faster computation)
    anal_sample_no = min([1000, len(ori_data),len(generated_data)])
    idx = np.random.permutation(len(ori_data))[:anal_sample_no]

    # Data preprocessing
    ori_data = np.asarray(ori_data)
    generated_data = np.asarray(generated_data)  

    ori_data = ori_data[idx]
    generated_data = generated_data[idx]

    def SameTimeSeq(ori_data, gen_data):
        #make time_seq_o and time_seq_g of a pair of ori_df and gen_df the same to get tSNE and PCA work; we achieve this
        #by padding zeros to the shorter time_seq df.
        no_o, time_seq_o, d = ori_data.shape
        no_g, time_seq_g, d = gen_data.shape
        time_seq = max(time_seq_o,time_seq_g)
        if time_seq_o != time_seq_g:
            if time_seq_g < time_seq_o:
                padding = np.zeros((no_g,time_seq_o-time_seq_g,d))
                gen_data = np.concatenate((gen_data,padding),axis = 1)
            else:
                padding = np.zeros((no_o,time_seq_g-time_seq_o,d))
                ori_data = np.concatenate((ori_data,padding),axis = 1)
        return ori_data,gen_data,time_seq

    ori_data, generated_data, seq_len = SameTimeSeq(ori_data, generated_data)

    for i in range(anal_sample_no):
        if (i == 0):
          prep_data = np.reshape(np.mean(ori_data[0,:,:], 1), [1,seq_len])
          prep_data_hat = np.reshape(np.mean(generated_data[0,:,:],1), [1,seq_len])
        else:
          prep_data = np.concatenate((prep_data, 
                                      np.reshape(np.mean(ori_data[i,:,:],1), [1,seq_len])))
          prep_data_hat = np.concatenate((prep_data_hat, 
                                          np.reshape(np.mean(generated_data[i,:,:],1), [1,seq_len])))

    # Visualization parameter        
    colors = ["red" for i in range(anal_sample_no)] + ["blue" for i in range(anal_sample_no)]    

    if analysis == 'pca':
        # PCA Analysis
        pca = PCA(n_components=2)
        pca.fit(prep_data)
        pca_results = pca.transform(prep_data) #PCA coordinates for the real data
        pca_hat_results = pca.transform(prep_data_hat) #PCA coordinates for the synthetic data

        # Plotting
        f, ax = plt.subplots(1)    
        plt.scatter(pca_results[:,0], pca_results[:,1],
                    c=colors[:anal_sample_no], alpha=0.2, label="Original")
        plt.scatter(pca_hat_results[:,0], pca_hat_results[:,1], 
                    c=colors[anal_sample_no:], alpha=0.2, label="Generated")

        ax.legend()  
        plt.title(syn_name+' '+'PCA plot')
        plt.xlabel('x-pca')
        plt.ylabel('y-pca')
        plt.show()
        
        return pca_results, pca_hat_results

    elif analysis == 'tsne':

        # Do t-SNE Analysis together       
        prep_data_final = np.concatenate((prep_data, prep_data_hat), axis=0)

        # TSNE anlaysis
        tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
        tsne_results = tsne.fit_transform(prep_data_final) #tSNE coordinates, 
        #for [:anal_sample_no, : ] are the coordinates for the original,
        #for [anal_sample_no: , : ] are the coordinates for the generated

        # Plotting
        f, ax = plt.subplots(1)

        plt.scatter(tsne_results[:anal_sample_no,0], tsne_results[:anal_sample_no,1], 
                    c=colors[:anal_sample_no], alpha=0.2, label="Original")
        plt.scatter(tsne_results[anal_sample_no:,0], tsne_results[anal_sample_no:,1], 
                    c=colors[anal_sample_no:], alpha=0.2, label="Generated")

        ax.legend()

        plt.title(syn_name+' '+'t-SNE plot')
        plt.xlabel('x-tsne')
        plt.ylabel('y-tsne')
        plt.show()
        
        return tsne_results[:anal_sample_no, :], tsne_results[anal_sample_no:,:]



def train_val_test_split(ori_data, gen_data, frac=(0.65, 0.2, 0.15)): 
    """
    Splits data into train, validation and test set according to fraction specified
    Args:
    ori_data: 3D np array of ori numerical data to be split
    gen data: 3D np array of gen numerical data to be split
    frac: fraction of split, should add up to 1
    """

    data = np.concatenate([ori_data,gen_data], axis=0)
    labels = np.concatenate([np.ones(len(ori_data)), np.zeros(len(gen_data))], axis=0)

    train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=frac[2])
    train_data, val_data, train_labels, val_labels = train_test_split(train_data, train_labels, train_size=frac[0]/(frac[0]+frac[1]))
    return train_data, val_data, test_data, train_labels, val_labels, test_labels


def discriminative_model(input_size, hidden_dim): 
    """
    define discriminative model
    """

    inputs = tf.keras.Input(shape=input_size)
    GRU_output_sequence, GRU_last_state = tf.keras.layers.GRU(hidden_dim, return_sequences=True, return_state=True)(inputs)
    Dense1 = tf.keras.layers.Dense(hidden_dim)(GRU_last_state)
    Dense2 = tf.keras.layers.Dense(1, activation='sigmoid')(Dense1)
    
    model = tf.keras.Model(inputs=inputs, outputs=[Dense2])
    model.compile(optimizer="adam", loss=tf.keras.losses.BinaryCrossentropy())
    
    return model


def subset (df, col):
    """
    gets the rows in the dataframe that has values "yes" in specified column

    Args:
    df: dataframe 
    col: column name in the dataframe where filtering should be done

    Returns:
    new_df: dataframe containing rows that have values "yes" in specified column
    """
    new_df= df[df[col].isin(["Yes","yes"])]
    
    return new_df


def auto_prep (df):
    """
    accepts a dataframe that contains date of events as input
    and computes the number of events that happen per week

    Args:
    df: dataframe with visit date

    Returns:
    df: dataframe with visit date 
    table_freq: dataframe with number of events per week
    """

    # Remove empty days and transform visit date to date time data type
    df = df.dropna(subset=['Visit date [EUPATH_0000091]'])
    df['Visit date [EUPATH_0000091]'] = pd.to_datetime(df['Visit date [EUPATH_0000091]'])

    # Create visit week column containing date on MONDAY of the week the visit took place in:
    df['visit_week'] = df['Visit date [EUPATH_0000091]'].apply(lambda d: d-dt.timedelta(days=d.weekday()))
    
    # Create frequency table of event
    table_freq = pd.value_counts(df['visit_week']).to_frame().reset_index()
    table_freq = table_freq.set_index("index").sort_index()
    
    return df, table_freq


def autocorrelation (df_ori, df_gen, table_freq_ori, table_freq_gen, syn_name, description):
    """
    plots autocorrelation for original and generate data

    Args:
    df_ori: dataframe of real data with dates
    df_gen: dataframe of gen data with dates
    table_freq_ori: dataframe with number of events per week for ori data, usually from auto_prep function
    table_freq_gen: dataframe with number of events per week for gen data, usually from auto_prep function
    syn_name: string format, version of synthetic data (for labelling)
    description: string format, any extra information for title of plot (for labelling)

    Returns:
    acf_df: dataframe with autocorrelation values for real and gen data
    fig: figure to be plotted
    """
    
    #making sure the start and finish are the same and that weeks with no events are 0
    weeks_df = pd.date_range(start=min(df_ori['visit_week'].min(), df_gen['visit_week'].min()),
                            end=max(df_ori['visit_week'].max(),df_gen['visit_week'].max()), freq='W-MON')
    weeks_df = pd.DataFrame(weeks_df)
    weeks_df.columns = ['index']
    weeks_df = weeks_df.set_index('index').sort_index()

    table_freq_ori = pd.merge(weeks_df, table_freq_ori, on="index", how='outer').fillna(0)
    table_freq_gen = pd.merge(weeks_df, table_freq_gen, on="index", how='outer').fillna(0)

    #autocorrelation from statsmodels.tsa.stattools
    acf_ori = acf(table_freq_ori['visit_week'], fft=False)
    acf_gen = acf(table_freq_gen['visit_week'], fft=False)
    
    acf_ori_df = pd.DataFrame(acf_ori)
    acf_gen_df = pd.DataFrame(acf_gen)
    
    acf_df = pd.concat([acf_ori_df, acf_gen_df], axis=1)
    acf_df.columns = ["original","generated"]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
                    x=acf_df.index, 
                    y=acf_df["original"],
                    mode='lines+markers',
                    name='original data'))
    fig.add_trace(go.Scatter(
                    x=acf_df.index, 
                    y=acf_df["generated"],
                    mode='lines+markers',
                    name='generated data'))
    fig.update_layout(xaxis_title="lag",
                      yaxis_title="autocorrelation",
                      title=syn_name + description)
    
    return acf_df, fig