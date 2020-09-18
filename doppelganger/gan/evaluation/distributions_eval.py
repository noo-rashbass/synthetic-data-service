import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import pearsonr
from math import sqrt
from sklearn.metrics import mean_squared_error
import plotly.graph_objects as go

def Histogram_KSTest(ori_nozero, gen_nozero, syn_name, size=100):
    """
    draws the pdf and cdf of numerical columns in original and generated data
    computes the KS Test value for ori and gen data
    K-S test result returns stats, p_value
    Ideally p_value should be as large as possible

    Args:
    ori_nonzero: original data with rows containing zeroes removed in dataframe format
    gen_nonzero: generated data with rows containing zeroes removed in dataframe format
    syn_name: version of synthetic data (used for labelling) e.g. 'doppelganger1'

    Returns:
    p_series: p_value for each numerical column in dataframe
    """
    keys = (ori_nozero.keys()).tolist()
    n = len(keys)
    l = min(len(ori_nozero),len(gen_nozero))
    
    p_value = np.zeros(n)
    for num in range(n):
        name = keys[num]
        ser_ori = ori_nozero[name].values[:l]
        ser_gen = gen_nozero[name].values[:l]
        df= pd.DataFrame({'ori':ser_ori, 'gen':ser_gen})
        df.plot.hist(bins=100, alpha=0.5, cumulative=False)
        plt.title(syn_name+' '+name+'_pdf')
        df.plot.hist(bins=100, alpha=0.5, cumulative=True, histtype='step')
        plt.title(syn_name+' '+name+'_cdf')
        plt.show()

        value = [0,0]
        for k in range(10):
        #Randomly take 100 samples from the generated and real data, since the total sample size is 40,000+, which is too
        #large and the null hypothesis can get easily rejected, which actually doesn't make statistical sense.
            idx = np.random.permutation(l)
            idx = idx[:size]
            name = keys[num]
            ser_ori = ori_nozero[name].values[idx]
            ser_gen = gen_nozero[name].values[idx]
            (t,p) = stats.ks_2samp(ser_ori, ser_gen)
            value[0] = value[0] + t
            value[1] = value[1] + p
        value = np.array(value)/10
        p_value[num] = value[1]
        print('K-S test result:',value) #the displayed array is the average (statistic, p_value), the closer to 0 the p is
        #the null hypothesis is more likely to be rejected.
    p_series = pd.Series(p_value, index = keys)
    return p_series

def Scatter_Distance(ori_data, gen_data, syn_name):
    """
    Plots the proportion of 1s and 0s in each categorical column in the synthetic data against the real data.
    An ideal synthetic data should lie on the diagonal y=x.
    Calculates the mse of original and generated data

    Args:
    ori_data: original data with only categorical columns in dataframe format
    gen_data: generated data with only categorical columns in dataframe format
    syn_name: version of synthetic data (used for labelling) e.g. 'doppelganger

    Returns:
    distance: MSE between ori and gen values
    """
    
    def CatProportion(series):
        return series.value_counts()/len(series)

    cat_keys = (gen_data.keys()).tolist()
    first = True
    name_index = []
    cat_index = []
    for name in cat_keys:
        # use 1 - value_for_0 to get True value (since value for 1 might be NaN)
        try:
            df_1=pd.DataFrame({'gen':1 - CatProportion(gen_data[name])[0],'ori': 1 - CatProportion(ori_data[name])[0]}, index=["True"])
        # actually value for 0 might be NaN as well sometimes 
        except:
            df_1=pd.DataFrame({'gen':CatProportion(gen_data[name])[1],'ori': CatProportion(ori_data[name])[1]}, index=["True"])
        if first:
            v = df_1.values
            first = False
        else:
            v = np.concatenate((v,df_1.values),axis = 0)
        df1_index = df_1.index.tolist()
        name_index += [name]*len(df1_index)
        cat_index += df1_index
    
    df_cat = pd.DataFrame(v, index = [name_index,cat_index], columns = [syn_name+ ' '+'gen probability',syn_name+' '+'ori probability'])
    df_cat = df_cat.fillna(0)
    display(df_cat)

    fig = go.Figure(data=go.Scatter(
                                x=df_cat[syn_name+' '+'gen probability'],
                                y=df_cat[syn_name+' '+'ori probability'],
                                mode='markers',
                                text=df_cat.index.get_level_values(0),
                                name='data'))
    fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines', name='diagonal'))
    fig.update_layout(xaxis_title="Generated Categorical values",
                      yaxis_title="Original Categorical values",
                      title=syn_name)
    fig.show()
    
    distance = mean_squared_error(df_cat.values[:, 0], df_cat.values[:, 1])
    
    return distance, df_cat


def r_corr_test(df, PTable=False, CoefficientandPtable=False, lower=True ):
    '''Returns a table of Pearson's r correlation coefficients between every pair of columns in the dataframe
    
    Args:
    df: The input dataframe
    PTable: False (default) or True, if True, then the return is a table containing the p(probavility)-value of correlation test.
    CoefficientandPtable: False(default) or True, if true, then the return is a table containing 
    tuples (p-value, r coefficient) from the correlation test.
    lower: True(default) or False. If True, the lower triangle part of the table is filled with 
    the transpose of the upper triangle part rather than leaved with None.
    
    Returns:
    The requested table as specified in the args. If PTable and CoefficientandPtable are all False, 
    then the return table consists of coefficient values only.
    '''

    df_index = (df.keys()).tolist()
    n = len(df_index)
    ini = [ [ None for y in range( n ) ] 
                 for x in range( n ) ]

    #pearsonr returns two values: the correlation coefficient and significance test probability p
    #so we create two empty dataframes to store them
    coefficient_table = pd.DataFrame(ini,index = df_index,columns = df_index)
    p_table = coefficient_table.copy()
    coe_and_p_table = coefficient_table.copy()

    for i in range(n):
        for j in range(i+1,n):
            name1 = df_index[i]
            name2 = df_index[j]
            obs_1 = df[name1].dropna()
            obs_2 = df[name2].dropna()
            dataframe = pd.DataFrame({name1: obs_1, name2: obs_2})

            values = dataframe.dropna().values
            (coe,p) = pearsonr(values[:,0],values[:,1])
            coefficient_table.loc[name1,name2]=coe
            p_table.loc[name1,name2]=p
            coe_and_p_table.loc[name1,name2]=(coe,p)
    
    if lower:
        #A function that can fill the lower part of the dataframe, because coe_table and p_table has their lower triangles empty
        #But for comparison reasons you may want them to be filled
        def fill_lower(df):
            n = df.values.shape[0]
            for j in range(n):
                for i in range(j+1,n):
                    df.iloc[i,j]=df.iloc[j,i]
            return df
        
        coefficient_table = fill_lower(coefficient_table)
        p_table = fill_lower(p_table)
        coe_and_p_table = fill_lower(coe_and_p_table)
    
    if PTable:
        return p_table
    elif CoefficientandPtable:
        return coe_and_p_table
    else:
        return coefficient_table


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

def CorrelationSRA(ori_correlation_df, gen_correlation_df, ColumnWise = False):
    '''Returns the value of SRA for the absolute Pearsons correlation coefficients for each column between 
    all other columns. SRA is between 0 and 1, the closer the SRA is to 1, the more the agreement between the ranking,
    the more similar the synthetic data and the real data are.
    
    Args:
    ori_correlation_df: the correlation coefficient dataframe for the real data, usually generated from the function 
                        r_corr_test.
    gen_correlation_df: the correlation coefficient dataframe for the synthetic data, usually generated from the function 
                        r_corr_test. 
    ColumnWise: False(default) or True. If True, the return is a Series containing the SRA value for each column and the average.
                Otherwise, the return is the average of SRA values for all columns
    
    Returns:
    s: It is either a column-wise SRA series or the average SRA values of them, determined by the arg ColumnWise.
    '''
    
    columns = (ori_correlation_df.keys()).tolist()
    n = len(columns)
    ini = np.ones(n)
    
    for i in range(n):
        ori_values = ori_correlation_df.iloc[i,:].fillna(-1)
        gen_values = gen_correlation_df.iloc[i,:].fillna(-1) #quick fix when len(R) != len(S) due to a particular column having all zeroes, 
        # hence r_corr_test returns NAN values, the fillna above was originally dropna(), 
        # resulting that column to be dropped and creating trouble in CorrelationSRA as len(R) != len(S) anymore, hence index out of bound when looping
        ini[i] = SRA(abs(ori_values), abs(gen_values))
    
    if ColumnWise:
        s = pd.Series(ini,index = columns)
        s['AVERAGE'] = sum(ini)/n
    else:
        s = sum(ini)/n
    return s


def MSE(r_table_ori, r_table_gen):
    '''
    Returns the MSE for each position between two dataframes and an average value.
    
    Args:
    r_table_ori: dataframe output from r_corr_test func for original data
    r_table_gen: dataframe output from r_corr_test func for generated data

    Returns:
    df: dataframe with MSE between original and generated data for correlation values
    score: float value representing MSE of all correlation values
    '''
    import pandas as pd
    import numpy as np
    ori = r_table_ori.fillna(0).values
    gen = r_table_gen.fillna(0).values
    columns = (r_table_gen.keys()).tolist()
    matrix = (ori-gen)**2
    df = pd.DataFrame(matrix, index = columns, columns = columns)
    score = np.sum(matrix)/(len(ori)*(len(ori)-1)) #The diagonal is always zero so we don't count them
    return df, score

# NOT USED NOW
def PtToDiagnalDist(coordinates, aver=True):
        '''
        For a series of points(2D), calculate the distance between each points and the diagonal y=x
        
        Args: 
        coordinates: array in shape (n,2), n is the number of points
        aver: True(default) or False. If aver == True, the distances are averaged, otherwise a list of distances is returned
        
        Returns:
        a: a list of distances, returned when aver==False
        np.sum(a)/n: the average of distances, returned when aver == True
        '''
        n = len(coordinates)
        a = np.zeros(n)
        for i in range(n):
            x = coordinates[i][0]
            y = coordinates[i][1]
            d = abs(x-y)/sqrt(2) #formula to calculate the distance between point (X,Y) and line ax+by+c = 0 is
            # abs(aX+bY+c)/abs(a^2+b^2)
            a[i]=d
        if aver:
            return np.sum(a)/n
        else:
            return a
