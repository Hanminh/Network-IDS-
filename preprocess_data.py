import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer, OneHotEncoder, LabelEncoder
import seaborn as sns
import numpy as np
from env import *

def add_header(df):
    df = df.drop(df.columns[42], axis=1)
    df.columns = ['duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes', 'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations', 'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login', 'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate', 'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'class_label']
    return df

def multi_class_label(df):
    df.class_label.replace(['apache2','back','land','neptune','mailbomb','pod','processtable','smurf','teardrop','udpstorm','worm'],'Dos',inplace=True)
    df.class_label.replace(['ftp_write','guess_passwd','httptunnel','imap','multihop','named','phf','sendmail',
       'snmpgetattack','snmpguess','spy','warezclient','warezmaster','xlock','xsnoop'],'R2L',inplace=True)
    df.class_label.replace(['ipsweep','mscan','nmap','portsweep','saint','satan'],'Probe',inplace=True)
    df.class_label.replace(['buffer_overflow','loadmodule','perl','ps','rootkit','sqlattack','xterm'],'U2R',inplace=True)
    return df

def standardize_numeric_feature(df):
    scaler = StandardScaler()
    numeric_features = df.select_dtypes(include=[np.float64, np.int64]).columns
    for i in numeric_features:
        df[i] = scaler.fit_transform(df[i].values.reshape(-1,1))
    return df

def process_categorical_feature(df, all_columns=None):
    categorical = df[['protocol_type', 'service', 'flag']]
    dummies = pd.get_dummies(categorical)
    if all_columns is not None:
        # Align test data with train's dummy columns
        dummies = dummies.reindex(columns=all_columns, fill_value=0)
    return dummies, dummies.columns  # Return the columns for alignment

def process_multi_class_label(df):
    label_encoder = LabelEncoder()
    multi_label = pd.DataFrame(df.class_label)
    enc = multi_label.apply(label_encoder.fit_transform)
    df['intrusion_type'] = enc
    df= pd.get_dummies(df, columns= ['class_label'], prefix= '', prefix_sep= '')
    df['class_label'] = multi_label
    return df, label_encoder.classes_
    
def feature_extraction(df):
    numeric_features_names = df.select_dtypes(include='number').columns
    numeric_features = df[numeric_features_names]
    correlation = numeric_features.corr()
    corr_y = abs(correlation['intrusion_type'])
    highest_corr = corr_y[corr_y > 0.5]
    return highest_corr, highest_corr.index

def preprocess_data(path=PATH_TRAIN_FULL, type='train', all_columns=None, feature_names=None):
    df = pd.read_csv(path, header=None)
    df = add_header(df)
    df = multi_class_label(df)
    df = standardize_numeric_feature(df)
    
    # Process categorical features
    if type == 'train':
        categorical, all_columns = process_categorical_feature(df)
    else:
        categorical, _ = process_categorical_feature(df, all_columns)

    df, label_encoder = process_multi_class_label(df)
    if type == 'train':
        corr, feature_names = feature_extraction(df)
    else :
        corr = None
        feature_names = feature_names
    df0 = df[feature_names]
    df0 = df0.join(categorical)
    df0 = df0.join(df[['Dos', 'R2L', 'U2R', 'Probe', 'normal', 'class_label']])
    df0.to_csv(f'DataProcessed/{type}.csv', index=False)
    return df0, corr, label_encoder, all_columns, feature_names

# Process train data
df, corr, label, categorical_columns, feature_names = preprocess_data(PATH_TRAIN_FULL, 'train')

# Save the label encoder and column names
np.save("le_train.npy", label, allow_pickle=True)
np.save("categorical_columns.npy", categorical_columns, allow_pickle=True)

# Process test data using the same categorical columns
df1, corr1, label1, _, _ = preprocess_data(PATH_TEST_FULL, 'test', categorical_columns, feature_names)

# Save the label encoder for test data
np.save("le_test.npy", label1, allow_pickle=True)



