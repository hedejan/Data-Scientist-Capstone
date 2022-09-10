import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
import pickle
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    StandardScaler, PowerTransformer, OneHotEncoder, FunctionTransformer,)
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from kneed import KneeLocator
from sklearn.pipeline import Pipeline
from functions.clean_transform import (
    dump_pickle, load_pickle, )



def scree_plot(pca, target):
    '''
    Creates a scree plot associated with the principal components 
    
    INPUT: pca - the result of instantian of PCA in scikit learn
            
    OUTPUT:
            None
    '''
    num_components = len(pca.explained_variance_ratio_)
    ind = np.arange(0,num_components)
    vals = pca.explained_variance_ratio_*100
    cumvals = np.cumsum(vals)
    n_components = np.where(cumvals>=target+1)[0][0]
    print(f'Using {n_components}/{num_components} components, at least {target}% of the variance explained.\n\n')
    
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    
    ax1.bar(ind, vals, color='r')
    ax2.plot(ind, cumvals, color='b')
    for i in range(0,num_components, int(num_components/15)):
        ax1.annotate(r'%s%%' % ((str(vals[i])[:4])), (ind[i]+0.2, vals[i]), va='bottom', ha='center', fontsize=12)
    ax1.xaxis.set_tick_params(width=0)
    ax1.yaxis.set_tick_params(width=2, length=12)
 
    ax1.set_xlabel('Number of Principal Components', fontsize=14)
    ax1.set_ylabel('Variance Explained (%)', color='r', fontsize=14)
    ax2.set_ylabel('Cumulative Explained Variance (%)', color='b', fontsize=14)
    ax2.set_yticks([20, 40, 60, 80, 90, 95, 100])
    plt.title('Explained Variance Per Principal Component', fontsize=18)
    plt.grid()
    fig.set_size_inches(16, 9)
    plt.show()
    
    return n_components


def do_pca(df:pd.DataFrame) -> pd.DataFrame:
    """"
    This func instantiate a PCA, finds n_components that explains pre-defined percentage of variance
    amd fit_transform df to reduce its dimensionality

    Args:
        df (dataframe): input dataset

    Returns:
        dataframe: reduced dataset
    """
    
    pca = PCA(n_components=None)
    pca.fit(df)
    n_components = scree_plot(pca, 95)
    pca = PCA(n_components=n_components)
    df_pca = pca.fit_transform(df)

    return pca, df_pca


def find_pca_components(pca, pca_in_features):
    
    pca_components = pd.DataFrame(
        np.round(pca.components_, 4), columns=pca_in_features,)

    return pca_components


def pca_results(df, pca):
    '''
    Create a DataFrame of the PCA results
    Includes dimension feature weights and explained variance
    Visualizes the PCA results
    '''

    # Dimension indexing
    dimensions = ['Dimension {}'.format(i) for i in range(1,len(pca.components_)+1)]

    # PCA components
    components = pd.DataFrame(np.round(pca.components_, 4), columns = df.keys())
    components.index = dimensions

    # PCA explained variance
    ratios = pca.explained_variance_ratio_.reshape(len(pca.components_), 1)
    variance_ratios = pd.DataFrame(np.round(ratios, 4), columns = ['Explained Variance'])
    variance_ratios.index = dimensions

    # Return a concatenated DataFrame
    return pd.concat([variance_ratios, components], axis = 1)

def pca_explained_var(df_clean, n_components):
    pca = PCA(n_components=n_components)
    df_pca = pca.fit_transform(df_clean)
    comp_check = pca_results(df_clean, pca)
    explained_var = comp_check['Explained Variance'].sum()
    print(f'for n_components={n_components}, the cumulative explained variance percentage = {explained_var*100:.2f}%')
    return None


def pca_weights(pca_in_features, pca, component_no, show_plot=False):
    
    pca_weights = pd.DataFrame(np.round(pca.components_, 4), columns=pca_in_features).iloc[component_no].sort_values(ascending=False)
    pca_weights = pd.concat([pca_weights.head(5), pca_weights.tail(5)])
    
    if show_plot:
        # Plot the result
        plt.figure(figsize=(16,9))
        pca_weights.plot(kind='bar')
        plt.xticks(fontsize=14)
        plt.title(label='Most 10 weighted features for PCA component %s'%component_no, 
                  fontsize=20)
        plt.grid()
        plt.show()
    print(pca_weights)
    return pca_weights


def get_kmeans_score(data, center):
    '''
    returns the kmeans score regarding SSE for points to centers
    INPUT:
        data - the dataset you want to fit kmeans to
        center - the number of centers you want (the k value)
    OUTPUT:
        score - the SSE score for the kmeans model fit to the data
    '''
    #instantiate kmeans
    kmeans = KMeans(n_clusters=center)

    # Then fit the model to your data using the fit method
    model = kmeans.fit(data)

    # Obtain a score related to the model fit
    score = np.abs(model.score(data))

    return score