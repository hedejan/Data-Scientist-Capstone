import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    StandardScaler, PowerTransformer, 
    OneHotEncoder, FunctionTransformer,)
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from tqdm import tqdm
import warnings



datasets_dict = {'azdias':'./arvato_data/Udacity_AZDIAS_052018.csv', 
               'customers':'./arvato_data/Udacity_CUSTOMERS_052018.csv',
               'train':'./arvato_data/Udacity_MAILOUT_052018_TRAIN.csv',
               'test':'./arvato_data/Udacity_MAILOUT_052018_TEST.csv',
              }

def dump_pickle(variable, save_as, folder):
    '''saves variable as pickle file in folder
    '''
    file_path = './' + folder + '/' + f'{save_as}.pkl'
    with open(file_path, 'wb') as f:
        pickle.dump(variable, f)
    
    return None

def load_pickle(saved_as, folder):
    '''loads variable from saved pickle file in folder
    '''
    file_path = './' + folder + '/' + f'{saved_as}.pkl'
    with open(file_path, 'rb') as f:
        
        return pickle.load(f)
    
    
def load_dataset(dataset=None):
    
    if not dataset:
        dataset = input('select a dataset (azdias, customers, train, test): ')
    
    file_path = datasets_dict[dataset]
    df = pd.read_csv(file_path, sep=';',parse_dates=['EINGEFUEGT_AM',], index_col='LNR',
                     converters={'GEBURTSJAHR': lambda x:x if x != '0' else np.nan},
                     na_values={
                        'CAMEO_DEUG_2015':'X',
                        'CAMEO_DEU_2015':'XX',
                        'CAMEO_INTL_2015':'XX',},
                        )
    print(f'{dataset} shape: {df.shape}')

    return df

def clean_RZ(df):
    mask = df['Attribute'].str.endswith('_RZ')
    df.loc[mask,'Attribute'] = df.loc[mask,'Attribute'].str.replace('_RZ','')
    return df

def create_ref_tables():
    
    attr = pd.read_excel('./arvato_data/DIAS Attributes - Values 2017.xlsx', 
                     skiprows=1, 
                     usecols='B:E',)\
                    .ffill()

    info = pd.read_excel('./arvato_data/DIAS Information Levels - Attributes 2017.xlsx',
                        skiprows=1, 
                        usecols='B:E',)\
                    .ffill().bfill()

    features_dtypes = pd.read_csv('./data/features_type.csv')

    # remove '_RZ' from end of Attribute column values
    attr, info = clean_RZ(attr), clean_RZ(info)
    
    # fix double entry issue
    info.Attribute = info.Attribute.str.replace('_ ','_')
    info.Attribute = info.Attribute.str.replace(' +',';', regex=True)
    info.Attribute = info.Attribute.str.split(';')
    info = info.explode('Attribute')

    # outer merge attr and info tables
    attr_info = pd.merge(left=attr, 
                         right=info, 
                         left_on=attr.Attribute, 
                         right_on=info.Attribute,
                         how='outer')\
    .drop(columns=['Attribute_x','Attribute_y','Description_x'])\
    .rename(columns={'key_0':'Attribute',
                     'Description_y':'Description',})

    # create ref table
    ref = pd.merge(
        left=attr_info, right=features_dtypes,left_on='Attribute',
        right_on='Attribute', how='outer',
    ).sort_values(by=['Attribute','Value'])
    
    # create ref table for values with meaning of unknown to be used to update pop and cust
    attr_unknown = attr_info[attr_info['Meaning'].str.contains('unknown', na=False)].iloc[:,0:2]\
    .assign(Value=lambda df:df['Value'].astype(str).str.replace(', ',' ').str.split())\
    .assign(Value=lambda df:df['Value'].apply(lambda x:[int(v) for v in x]))\
    .reset_index(drop=True)
    
    return attr_info, ref, attr_unknown

attr_info, ref, attr_unknown = create_ref_tables()


def map_nan(df, attr_unknown):
    # replace values representing unknowns to np.nan reference to attr_unknown table
    for attribute in attr_unknown.Attribute:
        if attribute in df.columns:
            to_replace = attr_unknown.query('Attribute == @attribute ')['Value'].values[0]
            # clean df
            df[attribute] = df[attribute].replace(to_replace=to_replace, value=np.nan)
            
    return df

def nan_hist_plots(df, df_name):
    ''' docstring '''
    
    print(f'{df_name} shape: ', df.shape)
    d0 = (df.isnull().sum(axis=0)/df.shape[0]).values
    d1 = (df.isnull().sum(axis=1)/df.shape[1]).values
    data = np.array([d0, d1], dtype=object)

    xaxes = ['Percentage of Missing values'] * 2
    yaxes = ['Count'] * 2
    titles = [
        f'''\n{df_name} Dataset
        Distrbution of Missing Values across Features\n''',
        f'''\n{df_name} Dataset
        Distrbution of Missing Values across Examples\n'''] 

    fig, axes = plt.subplots(1,2, figsize=(16, 8))
    plt.setp(axes, xticks=list(np.arange(0.0, 1.1, 0.1)))
    axes = axes.ravel()
    for idx, ax in enumerate(axes):
        ax.hist(data[idx])
        ax.set_title(titles[idx], fontsize=14)
        ax.set_xlabel(xaxes[idx])
        ax.set_ylabel(yaxes[idx])
        ax.grid(True)
    plt.tight_layout()
    plt.show()
    
    return None
    
def nan_boxplot(df, df_name):
    '''
    This func generates a boxplot for a dataset's statistics of missing values
    by rows and columns
    Input: dataset as Pandas dataframe
    Output: boxplot
    '''
    nan_rows_perc = df.isna().sum(axis=0)/df.shape[0]
    nan_cols_perc = df.isna().sum(axis=1)/df.shape[1]

    fig, ax = plt.subplots(figsize=(15,6),nrows=1, ncols=2)
    ax = ax.flatten()

    nan_rows_perc.plot(kind='box', ax=ax[0], grid=True, vert=False,
                 title='Ratio of Missing Values per Feature',)
    nan_cols_perc.plot(kind='box', ax=ax[1], grid=True, vert=False,
                  title='Ratio of Missing Values per Example',)
    
    fig.suptitle(f'NaN\'s Statistics for {df_name} Dataset\n', 
                 fontsize=16)
    fig.set_facecolor(color='lightgray', )
    fig.tight_layout()
    plt.show()
    
    return None

def upper_whisker(s):
    '''
    This func returns the upper whisker  as defined by matplotlib.pyplot.boxplot 
    documentation
    Input: 
        s: Series, list or Array-like
    Output: Upper whisker
    '''
    iqr = np.quantile(s, .75) - np.quantile(s, .25)
    # The default value of whis = 1.5 corresponds to Tukey's original definition of boxplots.
    
    return np.quantile(s, .75) + 1.3 * iqr

def clean_nan(df, row_thresh, col_thresh):
    '''
    This func cleans columns with amount of missing values above upper whiskers 
    or a threshold (whichever is higher) as per Tukey's definition
    Input: 
        df: data as pandas dataframe
        thresh: min perc of non-NAN values per columns
    Output: clean df
    '''
    
    nans_per_row_perc = df.isna().sum(axis=1)/df.shape[1]
    rows_to_retain = nans_per_row_perc <= max(row_thresh, upper_whisker(nans_per_row_perc))    
    
    nans_per_col_perc = df.isna().sum(axis=0)/df.shape[0]
    cols_to_retain = nans_per_col_perc <= max(col_thresh, upper_whisker(nans_per_col_perc))
    
    df = df.loc[rows_to_retain, cols_to_retain]
    
    return df

def fix_data(df):
    
    # rename df columns to match reference tables
    replace = {
        'CAMEO_INTL_2015': 'CAMEO_DEUINTL_2015',
        'D19_BUCH_CD':'D19_BUCH_RZ',
        'SOHO_KZ':'SOHO_FLAG',
        'KBA13_CCM_1401_2500':'KBA13_CCM_1400_2500',
        'KK_KUNDENTYP':'D19_KK_KUNDENTYP',
    }
    df.rename(columns=replace, inplace=True)
    
    # fix mixed dtypes issue
    df['CAMEO_DEUG_2015'] = df['CAMEO_DEUG_2015'].astype(float)
    
    return df


def classify_features(df, plot=False):

    # categorical columns
    bin_cols = [c for c in ref.query('Type == "binary"')['Attribute'].unique() if c in df.columns]
    
    # categorical columns
    cat_cols = [c for c in ref.query('Type == "categorical"')['Attribute'].unique() if c in df.columns]
    
    # ordinal columns
    ord_cols = [c for c in ref.query('Type in ["ordinal","interval"]')['Attribute'].unique() if c in df.columns]
    ord_cols.extend(['DECADE', 'MAINSTREAM', 'WEALTH', 'LIFE_STAGE', ])
    
    # numeric and skewed columns
    num_cols = [c for c in ref.query('Type == "numeric"')['Attribute'].unique() if c in df.columns]
    
    # mixed columns
    mixed_cols = [c for c in ref.query('Type == "mixed"')['Attribute'].unique() if c in df.columns]
    
    # date / datetime columns
    date_cols = [c for c in ref.query('Type == "date"')['Attribute'].unique() if c in df.columns]
    
    # create a key:value pair dictionary
    feat_types_dict =  {
        'bin_cols':bin_cols,
        'cat_cols':cat_cols,
        'ord_cols':ord_cols,
        'num_cols':num_cols,
        'mixed_cols':mixed_cols,
        'date_cols':date_cols,}

    # remove keys with empty list
    feat_types_dict = {k: v for k, v in feat_types_dict.items() if v != []}

    if plot:
        to_plot = pd.Series({k:len(feat_types_dict[k]) for k in feat_types_dict.keys()})
        fig, ax = plt.subplots(figsize=(12,5))
        ax.bar(x=to_plot.index, height=to_plot.values)
        for p in ax.patches:
            ax.annotate(str(p.get_height()), (p.get_x() * 1.005+.35, p.get_height() * 1.005))
        plt.show()
    
    return feat_types_dict


def feat_eng_data(df, feat_types_dict):
    
    to_drop = list(set(df.columns) - set(ref.Attribute.unique()))
    
    # feature engineer mixed features
    try:
        df['DECADE'] = df['PRAEGENDE_JUGENDJAHRE'].replace(
            {1:1,2:1,3:2,4:2,5:3,6:3,7:3,8:4,9:4,10:5,11:5,12:5,13:5,14:6,15:6})
        df['MAINSTREAM']=df['PRAEGENDE_JUGENDJAHRE'].replace(
        {1:1,10:1,11:0,12:1,13:0,14:1,15:0,2:0,3:1,4:0,5:1,6:0,7:0,8:1,9:0})
        to_drop.append('PRAEGENDE_JUGENDJAHRE')
        feat_types_dict['mixed_cols'].remove('PRAEGENDE_JUGENDJAHRE')
    except:
        pass
    
    try:
        df['WEALTH'] = df['CAMEO_DEUINTL_2015'].replace(
            {'11':5,'12':5,'13':5,'14':5,'15':5,'21':4,'22':4,'23':4,'24':4,
             '25':4,'31':3,'32':3,'33':3,'34':3,'35':3,'41':2,'42':2,'43':2,
             '44':2,'45':2,'51':1,'52':1,'53':1,'54':1,'55':1,})
        df['LIFE_STAGE']=df['CAMEO_DEUINTL_2015'].replace(
            {'11':1,'12':2,'13':3,'14':4,'15':5,'21':1,'22':2,'23':3,'24':4,
             '25':5,'31':1,'32':2,'33':3,'34':4,'35':5,'41':1,'42':2,'43':3,
             '44':4,'45':5,'51':1,'52':2,'53':3,'54':4,'55':5,})
        to_drop.append('CAMEO_DEUINTL_2015')
        feat_types_dict['mixed_cols'].remove('CAMEO_DEUINTL_2015')
    except:
        pass

    try:
        df['AGE'] = 2017 - df['GEBURTSJAHR'].astype(float)
    except:
        pass

    try:
        df['OST_WEST_KZ'].replace({'W': 1, 'O': 0}, inplace=True)
    except:
        pass
    try:
        df['VERS_TYP'].replace({2: 0}, inplace=True)
    except:
        pass
    try:
        df['ANREDE_KZ'].replace({2: 0}, inplace=True)
    except:
        pass

    # drop encode and other mixed features in addition to date columns
    to_drop = list(set(to_drop + feat_types_dict['date_cols']))
    df.drop(columns=to_drop, inplace=True)
    
    return df

def clean_data(df, row_thresh, col_thresh, test=False):
    
    print('data cleaning ...')
    df_clean = df.copy()
    if test:
        df = df.drop_duplicates()
    df_clean = df_clean.sample(frac=1) # randomize
    attr_info, ref, attr_unknown = create_ref_tables()
    df_clean = map_nan(df_clean, attr_unknown)
    df_clean = clean_nan(df_clean, row_thresh, col_thresh)
    df_clean = fix_data(df_clean)
    feat_types_dict = classify_features(df_clean)
    df_clean = feat_eng_data(df_clean, feat_types_dict)
    
    return df_clean


def create_power_transformer(df):
    '''
    To fix the skeweness issue in numeric features_res
    '''
    feat_types_dict = classify_features(df)
    num_cols = feat_types_dict['num_cols']
    skewed_cols = (df[num_cols].abs().skew() > 1.5).where(lambda x:x==True).dropna().index.tolist()
    pt = PowerTransformer()
    pt.fit(df[skewed_cols])

    return skewed_cols, pt


def create_column_transformer(df):

    feat_types_dict = classify_features(df)

    bin_pipeline = Pipeline([
        ('impute', SimpleImputer(missing_values=np.nan, strategy='most_frequent')),
    ])    

    cat_pipeline = Pipeline([
        ('impute', SimpleImputer(missing_values=np.nan, strategy='most_frequent')),
        ('encode', OneHotEncoder(handle_unknown='ignore'))
    ])

    ord_pipeline = Pipeline([
        ('impute', SimpleImputer(missing_values=np.nan, strategy='most_frequent')),
        ('scale', StandardScaler()),
    ])

    num_pipeline = Pipeline([
        ('impute', SimpleImputer(missing_values=np.nan, strategy='mean')),
        ('scale', StandardScaler()),
    ])

    # build data transformer pipeline
    transformers = [
        ('numerical', num_pipeline, feat_types_dict['num_cols']), # 0
        ('ordinal', ord_pipeline, feat_types_dict['ord_cols']), # 1
        ('binary', bin_pipeline, feat_types_dict['bin_cols']), # 2
        ('categorical', cat_pipeline, feat_types_dict['cat_cols']), # 3
        ('mixed', cat_pipeline, feat_types_dict['mixed_cols']), # 4
    ]

    # instantiate a column transformer
    column_transformer = ColumnTransformer(transformers = transformers)

    return column_transformer


def transform_data(df, column_transformer, fit=False):
    
    print('column transformation ...')
    index = df.index
    feat_types_dict = classify_features(df)

    if fit:
        df_trans = column_transformer.fit_transform(df)
    else:
        df_trans = column_transformer.transform(df)
    
    # extract dummy column names
    dummy_cat_cols = list(column_transformer.transformers_[3][1].named_steps['encode'].get_feature_names_out(feat_types_dict['cat_cols']))
    dummy_mixed_cols = list(column_transformer.transformers_[4][1].named_steps['encode'].get_feature_names_out(feat_types_dict['mixed_cols']))

    col_names = (
        # feat_types_dict['skewed_cols'] + 
        feat_types_dict['num_cols'] + 
        feat_types_dict['ord_cols'] + 
        feat_types_dict['bin_cols'] + 
        dummy_cat_cols + 
        dummy_mixed_cols
    )
    df_trans = pd.DataFrame(data=df_trans, columns=col_names, index=index)
    
    return df_trans


def match_features(df, pca_in_features):

    df_features, features_to_match = set(df.columns), set(pca_in_features)
    add_features, drop_features = list(features_to_match - df_features), list(df_features - features_to_match)
    df[add_features] = 0
    df.drop(columns=drop_features, inplace=True)
    
    return df

def pca_prep_check(df, pca_in_features):

    print('features not in pca_in_features:')
    print(list(set(df.columns) - set(pca_in_features)))
    print()
    print('features only in pca_in_features:')
    print(list(set(pca_in_features) - set(df.columns)))
    
    return None