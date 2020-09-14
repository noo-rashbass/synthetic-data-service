import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import LinearSVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error


class Prediction:
    def __init__(self, df, df2=None):
        if type(df2) != pd.core.frame.DataFrame: 
            #so the class is suitable for both TSTR and TSTS&TRTR purpose depending on how many dataframs are initialised. 
            #If there is only one 'df', then it will carry on training on this df and testing on this df;
            #If there is another 'df2', then it will train on df and test on df2, so this is for TSTR purporse
            self.df_train, self.df_test = train_test_split(df, test_size=0.25)
        else:
            self.df_train, _1= train_test_split(df, test_size=0.25)
            _2, self.df_test = train_test_split(df2, test_size=0.25)
        self.keys = (df.keys()).tolist()

    def ColumnPrepare(self, column_name):
        #We want to predict column 'column_name' from other columns, so X is values from other columns,
        #y is values from column 'column_name'; _train variables are used to train models (supervised learning),
        #_test variables are used to test models
        keys = self.keys.copy()
        keys.remove(column_name)
        keys_removed = keys
        
        X_train = self.df_train[keys_removed].values
        y_train = self.df_train[column_name].values
        X_test = self.df_test[keys_removed].values
        y_test = self.df_test[column_name].values
        return X_train, y_train, X_test, y_test

    def PredictionColumnWise(self, method):
        '''
        Train predictive models specified by 'method' arg to predict each column from other columns
        Args:
        method: ='LR','DTR','SVM','RFR', refering to different predictive models
        
        Returns: dataframe consisting of y_test (label) and y_predict (predicted results) for each column
        '''
        name_index_1 = []
        name_index_2 = ['y_test','y_pred']*len(self.keys)
        first = True
        for column_name in self.keys:
            X_train, y_train, X_test, y_test = self.ColumnPrepare(column_name)
            
            if method == 'LR':
                reg = LinearRegression()
            if method == 'DTR':
                reg = DecisionTreeRegressor()
            if method == 'SVM':   
                reg = LinearSVR()
            if method == 'RFR':
                reg = RandomForestRegressor()
            if method == 'KNR1':
                reg = KNeighborsRegressor(radius=1)
            if method == 'KNR5':
                reg = KNeighborsRegressor(radius=5)
            reg.fit(X_train, y_train)    
            y_pred = reg.predict(X_test)
            value_temp = np.concatenate((y_test[:,np.newaxis], y_pred[:,np.newaxis]), axis = 1)
            value_temp = value_temp.T

            if first:
                value = value_temp
                first = False
            else:
                value = np.concatenate((value,value_temp), axis=0)
            name_index_1 += [column_name]*2
        index = pd.MultiIndex.from_arrays([name_index_1,name_index_2])
        return pd.DataFrame(value.T, columns=index )
    
    def Evaluation_MSE(self, method, aver=False):
        '''
        Args:
        method: ='LR','DTR','SVM','RFR', refering to different predictive models
        
        Returns:
        The User can choose to return a series of MSEs for each column by leaving aver == False, 
        if aver == True, then the return is the 'E' i.e. average of MSEs for each column
        '''
        
        Presult = self.PredictionColumnWise(method)
        n = len(self.keys)
        MSE = np.array([0.5]*n)
        k=0
        for column_name in self.keys:
            y_pred = Presult[column_name]['y_pred']
            y_test = Presult[column_name]['y_test']
            MSE[k] = mean_squared_error(y_pred, y_test)
            k+=1
        MSE_series = pd.Series(MSE, index=self.keys) # The e_i's for every column
        E = np.sum(MSE_series) / len(MSE_series) # The 'E' for this model
        if aver:
            return E
        else:
            return pd.Series(MSE, index=self.keys)


def Comparison_Table(df_dic, aver, models=['LR','DTR','SVM','RFR','KNR1','KNR5']):
    '''
    Takes in a dictionary with key: data name, value: dataframe of ori numerical data, dataframe of gen numerical data
    If aver = True, it returns a dataframe which containes the E values for the model which is trained and tested on 
                    the dataset in that column
    If aver = False, it returns a dataframe which contains the MSE for each numerical column

    Args:
    df_dic: dictionary for real and fake numerical data
    aver: whether the MSE should be averaged
    models: models to run on

    Returns:
    d: dataframe with averaged or non-averaged MSE values
    '''
    # multi index for aver = True
    keys = list(df_dic.keys())
    keys_index = []
    v_np = np.zeros((len(models),len(keys)*2))
    for k in keys:
        keys_index += [k] * 2
    type_index = ['ori MSE','gen MSE']*len(keys)

    # multi index for aver = False
    #assuming all dfs in dict have same columns
    ori_cols = df_dic[keys[0]][0].columns.to_list()
    ori_cols_index = ori_cols * len(keys) * 2
    type_index_n = (['ori']*len(ori_cols) + ['gen']*len(ori_cols)) * len(keys)
    keys_index_n = []
    for k in keys:
        keys_index_n += [k] * len(ori_cols) * 2
    v_np_n = np.zeros((len(models), len(ori_cols_index)))

    # multi index for aver = True
    if aver:
        column_multi_avg = np.array([keys_index,type_index])
        column_multi_avg = pd.MultiIndex.from_arrays(column_multi_avg)
    
    # multi index for aver = False
    else:
        column_multi_navg = np.array([keys_index_n, type_index_n, ori_cols_index])
        column_multi_navg = pd.MultiIndex.from_arrays(column_multi_navg)
    
    for i in range(len(keys)):
        key = keys[i]
        ori = df_dic[key][0]
        gen = df_dic[key][1]
        PO = Prediction(ori)
        PG = Prediction(gen)

        #loop over for each model
        for j in range(len(models)):
            model = models[j]
            o = PO.Evaluation_MSE(model,aver)
            g = PG.Evaluation_MSE(model,aver)
            if aver:
                v_np[j][i*2] = PO.Evaluation_MSE(model,aver = True)
                v_np[j][i*2+1] = PG.Evaluation_MSE(model,aver = True)
                d = pd.DataFrame(v_np,index = models, columns = column_multi_avg)
            else:
                v_np_n[j][i*2*len(ori_cols) : i*2*len(ori_cols)+len(ori_cols)] = o
                v_np_n[j][i*2*len(ori_cols)+len(ori_cols) : i*2*len(ori_cols)+2*len(ori_cols)] = g
                d = pd.DataFrame(v_np_n,index = models, columns = column_multi_navg)
    return d


def plot_colmse(colmse_t):
    """
    Plots the MSE of each column of the ori data against gen data. Ideally points should lie on the diagonal

    Args:
    colmse_t: dataframe with MSE for each column for ori and gen data, usually from Comparison_Table func (aver=False)
    """

    versions = colmse_t.columns.unique(level=0).to_list()
    models = colmse_t.index.to_list()
    for i in range(len(versions)):
        for j in range(len(models)):
            orid = colmse_t.iloc[j][versions[i], 'ori'].values
            gend = colmse_t.iloc[j][versions[i], 'gen'].values
            # plotting log values as there is a big diff in magnitude
            plt.scatter(np.log(orid), np.log(gend), c='r')
            # plotting the diagonal
            plt.plot((np.log(min(orid)), np.log(max(orid))), (np.log(min(orid)), np.log(max(orid))))
            plt.xlabel('log of ori data mse')
            plt.ylabel('log of gen data mse')
            plt.show()


def get_origen_mse(com_t):
    """
    computes MSE between column-averaged mse for ori and gen data

    Args:
    com_t: dataframe with column-averaged mse for ori/ gen data for each predictive model, usually output from Comparison_Table func (with aver=True)

    Returns:
    df: dataframe with MSE values of ori and gen data
    """

    versions = com_t.columns.unique(level=0).to_list()

    mse_list = []
    for i in range(len(versions)):
        mse = mean_squared_error(com_t[versions[i], 'ori MSE'], com_t[versions[i], 'gen MSE'])
        mse_list.append(mse)
    mse_list = np.array(mse_list).reshape((-1, len(mse_list)))
    df = pd.DataFrame(mse_list, index=['avg ori gen MSE'], columns=versions)
    return df


def get_avg_mse_per_model(colmse_t):
    """
    computes the average mean squared error (over models) for each numerical column

    Args:
    colmse_t: dataframe with has MSE for each numerical column for each model, usually from Comparison_Table function
    
    Returns:
    df: dataframe with average MSE for each model
    """
    # get the number of versions of data available
    versions = colmse_t.columns.unique(level=0).to_list()
    models = colmse_t.index.to_list()

    np_arr = np.zeros((len(models), len(versions)))

    #get the MSEs
    mse = []
    for i in range(len(versions)):
        for j in range(len(models)):
            error_per_model = mean_squared_error(colmse_t.iloc[j][versions[i], 'ori'], colmse_t.iloc[j][versions[i], 'gen'])
            np_arr[j, i] = error_per_model
    
    mse_df = pd.DataFrame(np_arr)
    mse_df.columns = versions
    mse_df.index = models

    return mse_df


def get_mean_avg_modelmse(avg_modelmse_t):
    """
    get the mean of the average MSEs per model
    """
    #mean per version of data
    df = pd.DataFrame(avg_modelmse_t.mean())

    #for the sake of consistency
    df = df.T

    df.index = ['mean model-averaged mse']
    return df


def SRA(R, S):
    '''Calculate the SRA of lists R and S
    
    Args:
    - R: A list of performance metrics of different predictive models from TSTS
    - S: A list of performance metrics of different predictive models from TRTR, len(S)=len(R)
    
    Returns:
    - SRA: SRA value
    
    '''
    def identity_function(statement):
        v = 0
        if statement:
            v = 1
        return v
            
    k = min(len(R), len(S)) #technically should be same
    sum_ = 0
    for i in range(k):
        for j in range(k):
            if i != j:
                if (R[i]-R[j])==0:
                    if (S[i]-S[j])==0:
                        agree = True
                    else:
                        agree = False
                else:
                    agree = (R[i]-R[j])*(S[i]-S[j])>0
                sum_ += identity_function(agree)
    SRA = sum_ / (k*(k-1))
    return SRA
    

def get_SRA_per_col(colmse_t, num_cols_name):
    """
    computes the SRA ranking for each numerical column

    Args:
    colmse_t: dataframe which consists of the MSE for each column of ori and gen data, usually output from Comparison_Table function
    num_cols_name: list of names of the numerical columns

    Returns:
    sra_df: dataframe that shows the SRA ranking of each column
    """
    versions = colmse_t.columns.unique(level=0).to_list()

    # for multi indexing
    v_index = []
    for v in versions:
        v_index += [v] * len(num_cols_name)

    col_multi_index = np.array([v_index, num_cols_name*len(versions)])
    col_multi_index = pd.MultiIndex.from_arrays(col_multi_index)

    # get the SRAs
    SRA_list = []
    for i in range(len(versions)):
        for c in num_cols_name:
            sra_col = SRA(colmse_t[versions[i],'ori', c], colmse_t[versions[i], 'gen', c])
            SRA_list.append(sra_col)
    sra_df = pd.DataFrame(np.array(SRA_list).reshape(-1,len(SRA_list)))
    sra_df.columns = col_multi_index
    sra_df.index = ['SRA']
    return sra_df


def Scatter_TSTS_TRTR(E_table):
    """
    Plots the E values for training + testing on real data against training + testing on synthetic data
    Ideally, the points should lie on the diagonal y=x
    Args:
    E_table: df containing E values (usually generated from Comparison_Table function, with aver=True)
    """
    df_dic = list(set(E_table.keys().get_level_values(0).tolist()))
    models = (E_table.index).tolist()
    l = len(df_dic)
    for n in range(l):
        key = df_dic[n]
        plt.scatter(E_table[key,'ori MSE'], E_table[key,'gen MSE'], color='r')
        max_v = np.max(E_table[key,'ori MSE'])
        plt.plot([0,max_v],[0,max_v])
        plt.xlabel('MSE of original data')
        plt.ylabel('MSE of generated data')
        plt.title(key)
        #plt.savefig('png_files/3.1 DWP/scatter from IIIE Table '+ key+'.png')
        plt.show()


def SRA_TSTS_TRTR(E_table):
    """
    SRA for table of E values
    """
    df_dic = list(set(E_table.keys().get_level_values(0).tolist()))
    models = (E_table.index).tolist()
    l = len(df_dic)
    v_np = np.zeros(l)
    for n in range(l):
        key = df_dic[n]
        v_np[n] = SRA(E_table[key,'ori MSE'],E_table[key,'gen MSE'])
    display('SRA: Ranking MSE for each model between real and generated data')
    return pd.Series(v_np, index=df_dic)


# TIME PREDICTIVE

def make_time_windows(dataset, w):
    """replaces each participant's 130 time-steps long timeseries with 
    all possible time series chunks of length w
    Args:
    dataset: 3d numpy array (num sample, max time series length, numerical column dimension) 
            containing only numerical columns from the dataset
    w: window length of time series

    """
    num_participants, full_length, _ = np.shape(dataset)
    time_windows = []

    for i in list(range(num_participants)): # i = participant's position in dataset

        for j in list(range(full_length-w+1)): # j = row number of first row in window
            time_windows.append(dataset[i,j:j+w,:])

    return np.stack(time_windows)


def make_x_y(dataset, y_index, w):
    """make inputs for model. Note the returned data's sequence length = w-1
        
        Args:
        dataset: 3d numpy array (num sample, max time series length, numerical column dimension) 
                containing only numerical columns from the dataset
        y_index: index of column to be treated as the 'label'
        w: window length of time series, should be less than time series length

        Returns:
        x: train data
        y: train label
    """
    dataset = make_time_windows(dataset, w)
    x = np.delete(dataset, obj=y_index, axis=2) # remove y column from all time series
    x = np.delete(x, obj=-1, axis=1) # remove last row from all time series
    y = dataset[:,1:,[y_index]] # take only y column and remove its earliest cell
    return x, y


def last_time_step_mae(Y_true, Y_pred):
    """
    calculates the MAE of the last time step
    """
    return tf.keras.metrics.MAE(Y_true[:, -1], Y_pred[:, -1])


def make_predictive_model(num_cols):
    """
    define architecture of time predictive model
    Args:
    num_cols: int value- number of columns
    """
    input_size = [None,num_cols-1] 
    hidden_dim = num_cols//2 #

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.InputLayer(input_shape=input_size)) # (#timesteps -1, #features=dim-1) 
    model.add(tf.keras.layers.GRU(hidden_dim, return_sequences=True)) # (#timesteps -1, hidden_dim)
    model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(hidden_dim, activation = "sigmoid")))  # (#timesteps -1, hidden_dim)
    model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(1, activation = "linear")))  # (#timesteps -1, 1)

    model.compile(optimizer = "adam", loss = tf.keras.losses.MeanAbsoluteError(), metrics=[last_time_step_mae])

    return model


def predictive_train_test(train_set, test_set, y_cols_name, y_cols='all', windows='max', val_ratio=0.2, batch_size=64, epochs=15, show_training=True):
    """ does the T_T_ portion of the scheme

    Args:
    - train_set, test_set: ori_data, gen_data in order of T_T_ (labels NOT yet split off), usually output from train_test_split
    Make sure train and test are disjoint!
    If doing TSTSvsTRTR then use this function twice
    - y_cols: list of columns to use as target
    - y_cols_name: the names of ALL the numerical columns
    - windows: list contains lengths of windows to be used in order of y columns
            NOTE length of input sequences is actually window_length - 1 because of make_x_y
    - val_ratio: validation set ratio split from train_set for use during training
    - batch_size
    - epochs

    Returns:
    - results_df: dataframe containing training results
    - histories
    """

    results = []
    histories = []
    _, max_window, num_columns = np.shape(train_set)
    
    #### for convenience when not tuning parameters
    if y_cols=='all':
        y_cols = list(range(num_columns))

    if windows=='max':
        windows=[max_window]*len(y_cols)
    elif isinstance(windows, int):
        windows=[windows]*len(y_cols)

    for k in range(len(y_cols)): # step 5 (repeating steps 2-4 for each column)
        print('\nPredictive model running for column ', str(y_cols_name[k]), ' with window length ', str(windows[k]))
        train_x, train_y = make_x_y(train_set, y_cols[k], windows[k]) # step 2
        test_x, test_y = make_x_y(test_set, y_cols[k], windows[k])
        train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, test_size=val_ratio)

        model = make_predictive_model(num_columns)
        hist = model.fit(np.asarray(train_x), np.asarray(train_y), batch_size=batch_size, epochs=epochs, validation_data=(val_x, val_y), verbose=0) # step 4
        print('Getting results for column ', str(y_cols_name[k]), ' with window length ', str(windows[k]))

        results.append([y_cols_name[k], windows[k], batch_size, epochs]+model.evaluate(test_x, test_y, batch_size=batch_size))
        histories.append(hist)

        if show_training:
            plt.plot(hist.history['loss'])
            plt.plot(hist.history['val_loss'])
            plt.title('model loss')
            plt.ylabel('loss')
            plt.xlabel('epoch')
            plt.legend(['train loss', 'val loss'], loc='upper right')
            plt.show()

    results_df = pd.DataFrame(results, columns=['y_col', 'window_length', 'batch_size', 'epochs', 'test_loss', 'test_metric'])

    return results_df, histories
        

def get_TxTx_tpred_results(ori_3d_num, gen_3d_num, y_cols_name, y_cols='all', windows='max', val_ratio=0.2, batch_size=64, epochs=15, show_training=True):
    """
    computes TRTR TSTS test loss and metrics loss for each column specified
    works for only one version of data generated as it expects numpy arrays for ori_3d_num and gen_3d_num

    Args:
    ori_3d_num: 3d numpy array of numerical columns in original data
    gen_3d_num: 3d numpy array of numerical columns in generated data

    Returns:
    TRTR_TSTS_combine: dataframe with test loss and test metrics for TRTR and TSTS 
    """
    train_ori_set, test_ori_set = train_test_split(ori_3d_num, train_size=0.5)
    TRTR_results, TRTR_histories = predictive_train_test(train_ori_set, test_ori_set, y_cols_name, y_cols, windows, epochs=epochs, show_training=show_training)

    train_gen_set, test_gen_set = train_test_split(gen_3d_num, train_size=0.5)
    TSTS_results, TSTS_histories = predictive_train_test(train_gen_set, test_gen_set, y_cols_name, y_cols, windows, epochs=epochs, show_training=show_training)

    TRTR_TSTS_combine = pd.merge(TRTR_results[['y_col', 'test_loss', 'test_metric']], TSTS_results[['y_col','test_loss', 'test_metric']], on='y_col')
    TRTR_TSTS_combine.columns = ['y_col', 'test_loss_trtr', 'test_metric_trtr', 'test_loss_tsts', 'test_metric_tsts']

    return TRTR_TSTS_combine


def TxTx_tpred_multiple(df_dic, y_cols_name, y_cols='all', windows='max', val_ratio=0.2, batch_size=64, epochs=15, show_training=True):  
    """
    computes get_TxTx_tpred_results for multiple version of generated data since it expects a dictionary of data as input
    Concatenates results of multiple versions of data into a dataframe

    Args:
    df_dic: dictionary with key as data version and value as (ori_3d_num, gen_3d_num)
    """ 
    l = len(df_dic)
    keys = list(df_dic.keys())

    #for multi indexing
    keys_index = []
    for k in keys:
        keys_index += [k]* 5
    d = None

    for i in range(l):
            k = df_dic[keys[i]]
            ori_3d = k[0]
            gen_3d = k[1]
            d_txtx = get_TxTx_tpred_results(ori_3d, gen_3d, y_cols_name, y_cols, windows, epochs=epochs, show_training=show_training)
            if d is None:
                d = d_txtx
            else:
                d = pd.concat([d, d_txtx], axis=1)
    #multi indexing for columns
    cols = d_txtx.columns.to_list()
    multi_array = np.array([keys_index,cols * l])
    col_multi = pd.MultiIndex.from_arrays(multi_array)
    d.columns = col_multi

    return d


def plot_TxTx_tpred(TxTx_tpred):

    """
    Plots TRTR test loss/metric against TSTS. Ideally the points should lie on the diagonal

    Args:
    TxTx_tpred: dataframe containing the test loss/ metric for each numerical column, usually output from TxTx_tpred_multiple func
    """

    versions = TxTx_tpred.columns.unique(level=0).to_list()

    for i in range(len(versions)):
            orid = TxTx_tpred[versions[i], 'test_loss_trtr']
            gend = TxTx_tpred[versions[i], 'test_loss_tsts']
            plt.scatter(np.log(orid), np.log(gend), c='r')
            plt.plot([np.log(min(orid)), np.log(max(orid))], [np.log(min(orid)), np.log(max(orid))])
            plt.xlabel('log (TRTR test loss)')
            plt.ylabel('log (TSTS test loss)')
            plt.show()

            orid = TxTx_tpred[versions[i], 'test_metric_trtr']
            gend = TxTx_tpred[versions[i], 'test_metric_tsts']
            plt.scatter(np.log(orid), np.log(gend), c='r')
            plt.plot([np.log(min(orid)), np.log(max(orid))], [np.log(min(orid)), np.log(max(orid))])
            plt.xlabel('log (TRTR test metric)')
            plt.ylabel('log (TSTS test metric)')
            plt.show()


def get_TxTx_mse(TxTx_tpred):
    """
    gets the MSE over the columns for TRTR TSTS

    Args:
    TxTx_tpred: dataframe containing the test loss/ metric for each numerical column, usually output from TxTx_tpred_multiple func

    Returns:
    TxTx_tpred_mse: dataframe with TRTR TSTS mse
    """
    
    versions = TxTx_tpred.columns.unique(level=0).to_list()
    
    v_index = []
    for v in versions:
        v_index += [v] * 2
    t_index = ['test_loss', 'metric_loss'] * len(versions)
    col_multi_index = np.array([v_index, t_index])
    col_multi_index = pd.MultiIndex.from_arrays(col_multi_index)

    mse = []
    for i in range(len(versions)):
        test_mse = mean_squared_error(TxTx_tpred[versions[i], 'test_loss_trtr'], TxTx_tpred[versions[i], 'test_loss_tsts'])
        metric_mse = mean_squared_error(TxTx_tpred[versions[i], 'test_metric_trtr'], TxTx_tpred[versions[i], 'test_metric_tsts'])
        mse.append(test_mse)
        mse.append(metric_mse)
    mse = np.array(mse).reshape((-1, len(mse)))
    TxTx_tpred_mse = pd.DataFrame(mse, columns=col_multi_index, index=['trtr tsts MSE'])
    return TxTx_tpred_mse


def get_TxTx_SRA(TxTx_tpred):
    """
    compares numerical column between TRTR and TSTS to get SRA value

    Args: 
    TxTx_tpred: dataframe containing the test loss/ metric for each numerical column, usually output from TxTx_tpred_multiple func

    Returns:
    TxTx_tpred_sra: dataframe with SRA values for test loss/metric
    """

    versions = TxTx_tpred.columns.unique(level=0).to_list()
    
    v_index = []
    for v in versions:
        v_index += [v] * 2
    t_index = ['test_loss', 'metric_loss'] * len(versions)
    col_multi_index = np.array([v_index, t_index])
    col_multi_index = pd.MultiIndex.from_arrays(col_multi_index)

    sra = []
    for i in range(len(versions)):
        test_SRA = SRA(TxTx_tpred[versions[i], 'test_loss_trtr'], TxTx_tpred[versions[i], 'test_loss_tsts'])
        metric_SRA = SRA(TxTx_tpred[versions[i], 'test_metric_trtr'], TxTx_tpred[versions[i], 'test_metric_tsts'])
        sra.append(test_SRA)
        sra.append(metric_SRA)
    sra = np.array(sra).reshape((-1, len(sra)))
    TxTx_tpred_sra = pd.DataFrame(sra, columns=col_multi_index, index=['SRA'])

    return TxTx_tpred_sra