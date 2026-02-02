import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import os
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
import pickle
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

DIR_PATH = os.path.abspath(os.path.dirname(__file__))

skp_en = pd.read_pickle(os.path.join(DIR_PATH, '../../data/raw/en_SKP3.pcl'))
gender_en = pd.read_csv(os.path.join(DIR_PATH, '../../data/raw/en_IDSpola.csv'))
stanjazn_en = pd.read_csv(os.path.join(DIR_PATH, '../../data/raw/en_IDStanjaZN.csv'))
csd_en = pd.read_csv(os.path.join(DIR_PATH, '../../data/raw/en_PrejemnikCSD.csv'))
dndp_en = pd.read_csv(os.path.join(DIR_PATH, '../../data/raw/en_PrejemnikDNDP.csv'))
level_en = pd.read_pickle(os.path.join(DIR_PATH, '../../data/raw/en_Nivo.pcl'))
p16_en = pd.read_pickle(os.path.join(DIR_PATH, '../../data/raw/en_P16_level3.pcl'))
prenehanja_en = pd.read_csv(os.path.join(DIR_PATH, '../../data/raw/en_IDPrenehanjaDR.csv'))
skp2_en = pd.read_excel(os.path.join(DIR_PATH, '../../data/raw/SKP-08, V1 - Tabela.xlsx'), converters={'Šifra kategorije':str})
p16_level2_en = pd.read_excel(os.path.join(DIR_PATH, '../../data/raw/KLASIUS-P-16, V1 - Tabela.xlsx'), converters={'Šifra kategorije':str})

skp = dict(list(zip(skp_en.SKP3,skp_en.Naziv)))
gender = dict(list(zip(gender_en.IDspola,gender_en.Naziv)))
stanjazn = dict(list(zip(stanjazn_en.IdStanjaZN,stanjazn_en.Naziv)))
csd = dict(list(zip(csd_en.PrejemnikCSD,csd_en.Naziv)))
dndp = dict(list(zip(dndp_en.PrejemnikDNDP, dndp_en.Naziv)))
level = dict(list(zip(level_en.Nivo, level_en.Naziv)))
p16 = dict(list(zip(p16_en.p16_level3, p16_en.Naziv)))
prenehanja = dict(list(zip(prenehanja_en.IDprenehanjaDR, prenehanja_en.Naziv)))
skp2 = dict(list(zip(skp2_en['Šifra kategorije'].astype('category'), skp2_en['Angleški deskriptor'])))
p16_level2 = dict(list(zip(p16_level2_en['Šifra kategorije'].astype('category'), p16_level2_en['Angleški deskriptor'])))

class SurvivalDataset:
  def __init__(self, fname, path='', random_state = 11):
    if fname.endswith('.rda'): import pyreadr; self.dataset = pyreadr.read_r(os.path.join(path, fname))[fname.split('.')[0].split('/')[-1]]
    if fname.endswith('.csv'): self.dataset = pd.read_csv(os.path.join(path, fname))
    if fname.endswith('.xlsx'): self.dataset = pd.read_excel(os.path.join(path, fname))
    self.preprocessed = None
    self.random_state = random_state

  def preprocess(self, time = 'time', status = 'status'):
    if self.preprocessed is not None: return self.preprocessed
    df = self.dataset.copy()
    df = df.drop(columns=['id']) if 'id' in df.columns else df
    object_cols = df.select_dtypes(include=['object', 'category']).columns
    df = pd.get_dummies(df, columns=object_cols)
    df = df.fillna(df.mean())
    numeric_cols = df.select_dtypes(include=['number']).drop(columns=[time, status]).columns
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    if 2 in df[status].tolist(): df[status] = df[status].map({1: 0, 2:1, 0:0})
    df[df.select_dtypes(include=['bool']).columns] = df.select_dtypes(include=['bool']).astype('int')
    self.preprocessed = df
    return df
  
  def get_survival_vector(self, time = 'time', status = 'status'):
    if self.preprocessed is None: self.preprocess(time, status)
    survival_time = self.preprocessed[time].values
    survival_status = self.preprocessed[status].values
    time_upper_limit = int(survival_time.max())
    time_tensor = torch.empty((survival_time.shape[0], time_upper_limit))
    print(time_tensor.shape[1])
    for i,el in tqdm(enumerate(survival_time)):
      for j in range(time_tensor.shape[1]):
        if (survival_status[i] == 1) and (j>=el): time_tensor[i][j] = np.nan
        elif j<el: time_tensor[i][j] = 1
        else: time_tensor[i][j] = 0
    print(f'Succesfully created time tensor of shape: {time_tensor.shape}')
    return time_tensor
    
  def split_xy(self, time = 'time', status = 'status', test = 0.2):
    if self.preprocessed is None: self.preprocess(time, status)
    X = self.preprocessed.drop(columns=[time, status])
    y = self.get_survival_vector(time, status)
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test, random_state = self.random_state)
    print(X_train.shape, y_train.shape)
    return (X_train, X_test, y_train, y_test)

  def get_tensors(self, time = 'time', status = 'status'):
    if self.preprocessed is None: self.preprocess(time, status)
    X_train, X_test, y_train, y_test = self.split_xy(time, status)
    return (torch.tensor(np.vstack(X_train.values).astype('float32')),
            torch.tensor(np.vstack(X_test.values).astype('float32')),
            y_train,
            y_test)

  def pysurvival_split(self, time = 'time', status = 'status', test = 0.2):
    if self.preprocessed is None: self.preprocess(time, status)
    df = self.preprocessed.copy()
    from sklearn.model_selection import train_test_split
    index_train, index_test = train_test_split(range(df.shape[0]), test_size = test, random_state = self.random_state)
    data_train = df.iloc[index_train].reset_index( drop = True )
    data_test  = df.iloc[index_test].reset_index( drop = True )
    # Creating the X, T and E input
    X_train, X_test = data_train.drop([time, status], axis = 1) , data_test.drop([time, status], axis = 1)
    T_train, T_test = data_train[time].values, data_test[time].values
    E_train, E_test = data_train[status].values, data_test[status].values
    return (X_train, T_train, E_train, X_test, T_test, E_test)


class AutoEncoder: pass

class TabularDataset(Dataset):
    def __init__(self, data, cat_cols=None, output_col=None, event_col=None):
        self.n = data.shape[0]
        if event_col:
            self.event_X = data[event_col].astype(np.float32).values.reshape(-1,1)
        else:
            self.event_X = np.zeros((self.n, 1))
        if output_col:
            self.y = data[output_col].astype(np.float32).values.reshape(-1, 1)
        else:
            self.y =  np.zeros((self.n, 1))
        self.cat_cols = cat_cols if cat_cols else []
#         print(cat_cols)
        self.cont_cols = [col for col in data.columns
                          if col not in self.cat_cols + [output_col]]
        if self.cont_cols:
            self.cont_X = data[self.cont_cols].astype(np.float32).values
        else:
            self.cont_X = np.zeros((self.n, 1))
        if self.cat_cols:
            self.cat_X = data[self.cat_cols].astype(np.int64).values
        else:
            self.cat_X =  np.zeros((self.n, 1))
        print(self.cat_X.shape)

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return [self.y[idx], self.cont_X[idx], self.cat_X[idx], self.event_X[idx]]


class HecatDataset:
    def __init__(self, path, filename):
        self.dataset = pd.read_pickle(os.path.join(path, filename))
    
    def transform_entry_date(self, hecat_df):
        if 'Entry_date' in hecat_df.columns:
            hecat_df['Entry_month'] = hecat_df['Entry_date'].dt.month
            hecat_df['Entry_day'] = hecat_df['Entry_date'].dt.day
            hecat_df['Entry_month_sin'] = np.sin(2*np.pi*hecat_df['Entry_month']/max(hecat_df['Entry_month']))
            hecat_df['Entry_day_sin'] = np.sin(2*np.pi*hecat_df['Entry_day']/max(hecat_df['Entry_day']))
            hecat_df['Entry_month_cos'] = np.cos(2*np.pi*hecat_df['Entry_day']/max(hecat_df['Entry_day']))
            hecat_df['Entry_day_cos'] = np.cos(2*np.pi*hecat_df['Entry_day']/max(hecat_df['Entry_day']))
            hecat_df = hecat_df.drop(columns = ['Entry_date', 'Entry_month', 'Entry_day'])
        return hecat_df

    def rename_columns(self, df):
        df = df.rename(columns={"idosebe": "id", "StarostLeta": "Age", "MeseciDelDobe": "Months_of_work_experience", "IDSpola": "Gender", "IDObcine": "Municipality", "IDDrzave": "Country", "IDpoklicaSKP08": "Profession (ESCO)", "IdInvalidnosti": "Dissabilities", "DatumVpisaBO": "Entry_date", "IDVpisaBO": "Reason_for_PES_entry", "ePrijava": "eApplication", "IDKlasiusProgram": "Profession_program", "IDKlasiusP": "Specific_profession_category", "IDklasiusSRV": "Education_category", "IDStanjaZN": "Employment_plan_status", "IDZaposljivosti": "Employability_assessment", "IDPrenehanjaDR": "Employment_plan_ready"})
        return df
    
    def drop_unnecessary_columns(self, hecat_df):
        hecat_df = self.dataset
        hecat_df = hecat_df.drop(columns = ['DatumObdobja', 'MeseciBrezpos', 'DatumIzpisaBO', 'IDizpisaBO', 'PrejemnikDNDP', 'PrejemnikCSD', 'IdIndikatorPrometa', 'OEN', 'IDUpEnote', 'IzdelanZN', 'mso_from', 'mso_to', 'Unnamed: 0'])
        return hecat_df
    

    def rsf_dataset(self, to_pcl = False):
        columns_to_keep = ['truncated', 'Age', 'Months_of_work_experience', 'Entry_month_sin', 'Entry_day_sin',
       'Entry_month_cos', 'Entry_day_cos', 'Gender', 'Education_category', 'Profession_program', 'Dissabilities',
        'Reason_for_PES_entry', 'eApplication', 'Employment_plan_status', 'duration']
        continuous = ['Age', 'Months_of_work_experience', 'Entry_month_sin', 'Entry_day_sin',
               'Entry_month_cos', 'Entry_day_cos']
        cols = ['eApplication', 'Gender', 'Profession_program']
        for_scaling = ['Age', 'Months_of_work_experience']
        df = self.dataset
        df = self.drop_unnecessary_columns(df)
        df = self.rename_columns(df)
        df = self.transform_entry_date(df)
        df = df[df['duration']>0]
        df = df[columns_to_keep]
        df['Profession_program'] = df['Profession_program'].astype('str').str.zfill(4).str.slice(stop=2)
        df = df.loc[df['Profession_program']!='00']
    
        col_list = ['Education_category', 'Profession_program', 'Gender',
                                            'Age', 'Reason_for_PES_entry', 'Dissabilities']
        df = df.loc[~np.array(df[col_list].duplicated())]
        print(df.Profession_program.unique().tolist())
        le = preprocessing.LabelEncoder()
        for col in cols:
            df[col] = le.fit_transform(df[col].values)
        scaler = MinMaxScaler() 
        scaled_values = scaler.fit_transform(df[[col for col in df.columns if col in for_scaling]]) 
        df.loc[:,[col for col in df.columns if col in for_scaling]] = scaled_values
        colz = [col for col in df.columns if col not in continuous + ['duration', 'truncated']]
#        print(colz)
#        df[colz] = OneHotEncoder().fit_transform(df[colz])
        df = pd.get_dummies(df, columns = colz)
        if to_pcl is True:
            with open('../../data/processed/rsf_dataset_program.pcl', 'wb') as f:
                pickle.dump(df, f)
        return df
    
    def mtlr_dataset(self):
        df = self.dataset
        df = self.drop_unnecessary_columns(df)
        df = self.rename_columns(df)
        df = self.transform_entry_date(df)
        df = df[df['duration']>0]
        df = df[columns_to_keep]
        le = preprocessing.LabelEncoder()
        continuous = ['Age', 'Months_of_work_experience', 'Entry_month_sin', 'Entry_day_sin',
       'Entry_month_cos', 'Entry_day_cos']
        categorical = [col for col in df.columns if col not in continuous +['duration']]
        for catt in categorical:
            df[catt] = le.fit_transform(df[catt].values)
        df_test = df.sample(frac=0.2)
        df_train = df.drop(df_test.index)
        df_val = df_train.sample(frac=0.2)
        df_train = df_train.drop(df_val.index)
        train_dataset = TabularDataset(data=df_train, cat_cols=categorical,
                             output_col='duration')
        valid_dataset = TabularDataset(data=df_val, cat_cols=categorical,
                                     output_col='duration')
        test_dataset = TabularDataset(data=df_test, cat_cols=categorical,
                                     output_col='duration')
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        valid_loader = DataLoader(valid_dataset, batch_size=64, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
        
        return (train_loader, valid_loader, test_loader)


    def rsf_split(self, test = 0.2, to_pcl = True):
        df = self.rsf_dataset()
        # Building training and testing sets #
        index_train, index_test = train_test_split(range(df.shape[0]), test_size = test, random_state = 11)
        data_train = df.iloc[index_train].reset_index( drop = True )
        data_test  = df.iloc[index_test].reset_index( drop = True )

        # Creating the X, T and E input
        X_train, X_test = data_train.drop(['truncated', 'duration'], axis = 1) , data_test.drop(['truncated', 'duration'], axis = 1)
        T_train, T_test = data_train['duration'].values, data_test['duration'].values
        E_train, E_test = data_train['truncated'].values, data_test['truncated'].values
        if to_pcl is True:
            save = (X_train, T_train, E_train, X_test, T_test, E_test)
            with open('../../data/processed/rsf_split_program_wo_ea.pcl', 'wb') as f:
                pickle.dump(save, f)
        return (X_train, T_train, E_train, X_test, T_test, E_test)

    def mtlr_split(self, test = 0.2):
        pass
    
    def __str__(self):
        return str(self.dataset.head())

class ReducedDataset:

    def __init__(self, path, filename, random_state=11):
        self.dataset = pd.read_pickle(os.path.join(path, filename))
        self.dataset = self.dataset.sample(frac=0.1)
        self.random_state = random_state
        self.skp = pd.read_csv(os.path.join(DIR_PATH, '../../data/raw/dimSKP08.csv'))[['IDpoklicaSKP','SFpoklicaSKP']]
        self.skp.SFpoklicaSKP = self.skp.SFpoklicaSKP.astype(int).astype(str).str.zfill(4)
        self.p16_nivo = pd.read_pickle(os.path.join(DIR_PATH,'../../data/raw/p16_nivo_skp_surs.pcl'))
        self.skp_to_p16 = self.p16_nivo.stack().sort_values().groupby(level=[0]).tail(1).reset_index(level=1).droplevel(2).drop(columns=[0]).reset_index()
        self.dataset = pd.merge(self.dataset, self.skp, left_on = 'IDpoklicaSKP08', right_on='IDpoklicaSKP', how='left')
        self.dataset = pd.merge(self.dataset, self.skp_to_p16, left_on = 'SFpoklicaSKP', right_on='SKP4', how='left')
        self.dataset.loc[:,'SKP3'] = self.dataset.SKP4.str[:3].astype('category')
        self.dataset.loc[:,'SKP2'] = self.dataset.SKP4.str[:2].astype('category')
        self.dataset.loc[:,'P16_level3'] = self.dataset.P16.str[:3].astype('category')
        self.dataset.loc[:,'P16_level2'] = self.dataset.P16.str[:2].astype('category')
        self.dataset.IDPrenehanjaDR = self.dataset.IDPrenehanjaDR.apply(lambda x: prenehanja[x])
        self.dataset.P16_level3 = self.dataset.P16_level3.apply(lambda x: p16[x])
        self.dataset.Nivo = self.dataset.Nivo.fillna(self.dataset.Nivo.mode().iloc[0])
        self.dataset.Nivo = self.dataset.Nivo.apply(lambda x: level[x])
        self.dataset.PrejemnikDNDP = self.dataset.PrejemnikDNDP.apply(lambda x: dndp[x])
        self.dataset.PrejemnikCSD = self.dataset.PrejemnikCSD.apply(lambda x: csd[x])
        self.dataset.IDStanjaZN = self.dataset.IDStanjaZN.apply(lambda x: stanjazn[x])
        self.dataset.IDSpola = self.dataset.IDSpola.apply(lambda x: gender[x])
        self.dataset.SKP3 = self.dataset.SKP3.apply(lambda x: skp[x])
        self.dataset.SKP2 = self.dataset.SKP2.apply(lambda x: skp2[x])
        self.dataset.P16_level2 = self.dataset.P16_level2.apply(lambda x: p16_level2[x])
    
    def transform_entry_date(self, hecat_df):
        if 'Entry_date' in hecat_df.columns:
            hecat_df['Entry_month'] = hecat_df['Entry_date'].dt.month
            hecat_df['Entry_day'] = hecat_df['Entry_date'].dt.day
            hecat_df['Entry_month_sin'] = np.sin(2*np.pi*hecat_df['Entry_month']/max(hecat_df['Entry_month']))
            hecat_df['Entry_day_sin'] = np.sin(2*np.pi*hecat_df['Entry_day']/max(hecat_df['Entry_day']))
            hecat_df['Entry_month_cos'] = np.cos(2*np.pi*hecat_df['Entry_day']/max(hecat_df['Entry_day']))
            hecat_df['Entry_day_cos'] = np.cos(2*np.pi*hecat_df['Entry_day']/max(hecat_df['Entry_day']))
            hecat_df = hecat_df.drop(columns = ['Entry_date', 'Entry_month', 'Entry_day'])
        return hecat_df

    def rename_columns(self, df):
        # # Use the following line for ISCO3
        # df = df.rename(columns={"idosebe": "id", "StarostLeta": "Age", "MeseciDelDobe": "Months_of_work_experience", "IDSpola": "Gender", "IDObcine": "Municipality", "IDDrzave": "Country", "IDpoklicaSKP08": "Profession (ISCO)", "IdInvalidnosti": "Dissabilities", "DatumVpisaBO": "Entry_date", "IDVpisaBO": "Reason_for_PES_entry", "ePrijava": "eApplication", "IDKlasiusProgram": "Profession_program", "IDKlasiusP": "Specific_profession_category", "IDklasiusSRV": "Education_category", "IDStanjaZN": "Employment_plan_status", "IDZaposljivosti": "Employability_assessment", "IDPrenehanjaDR": "Reason for termination", "PrejemnikCSD": "Social_benefits", "PrejemnikDNDP": "Unemployment_benefits", "P16_level3": "P16_level", "Nivo": "Level", "SKP3": "ISCO"})
        #Use the following line for ISCO2
        df = df.rename(columns={"idosebe": "id", "StarostLeta": "Age", "MeseciDelDobe": "Months_of_work_experience", "IDSpola": "Gender", "IDObcine": "Municipality", "IDDrzave": "Country", "IDpoklicaSKP08": "Profession (ISCO)", "IdInvalidnosti": "Dissabilities", "DatumVpisaBO": "Entry_date", "IDVpisaBO": "Reason_for_PES_entry", "ePrijava": "eApplication", "IDKlasiusProgram": "Profession_program", "IDKlasiusP": "Specific_profession_category", "IDklasiusSRV": "Education_category", "IDStanjaZN": "Employment_plan_status", "IDZaposljivosti": "Employability_assessment", "IDPrenehanjaDR": "Reason for termination", "PrejemnikCSD": "Social_benefits", "PrejemnikDNDP": "Unemployment_benefits", "P16_level2": "P16_level", "Nivo": "Level", "SKP2": "ISCO"})
        return df
    
    def drop_unnecessary_columns(self, hecat_df):
        hecat_df = self.dataset
        hecat_df = hecat_df.drop(columns = ['DatumObdobja', 'MeseciBrezpos', 'DatumIzpisaBO', 'IDizpisaBO', 'IdIndikatorPrometa', 'OEN', 'IDUpEnote', 'IzdelanZN', 'mso_from', 'mso_to', 'Unnamed: 0'])
        return hecat_df
    
    def rsf_dataset_toy(self, to_pcl = False):
        columns_to_keep = ['truncated', 'Age', 'Months_of_work_experience', 'Entry_month_sin', 'Entry_day_sin',
       'Entry_month_cos', 'Entry_day_cos', 'Gender', 'Social_benefits', 'Unemployment_benefits', 'P16_level', 'ISCO', 'Level',
        'Reason_for_PES_entry', 'eApplication', 'Reason for termination', 'duration', 'Razvrstitev ZRSZ']
        continuous = ['Age', 'Months_of_work_experience', 'Entry_month_sin', 'Entry_day_sin',
               'Entry_month_cos', 'Entry_day_cos']

        for_scaling = ['Age', 'Months_of_work_experience']
        df = self.dataset
        df = self.drop_unnecessary_columns(df)
        df = self.rename_columns(df)
        df = self.transform_entry_date(df)
        df = df[df['duration']>0]
        df = df[columns_to_keep]

        for col in df.columns:
            print(col, ": ", df[col].nunique())

        scaler = MinMaxScaler() 
        scaled_values = scaler.fit_transform(df[[col for col in df.columns if col in for_scaling]]) 
        df.loc[:,[col for col in df.columns if col in for_scaling]] = scaled_values
        colz = [col for col in df.columns if col not in continuous + ['duration', 'truncated', 'Razvrstitev ZRSZ']]
#        print(colz)
#        df[colz] = OneHotEncoder().fit_transform(df[colz])
        df = pd.get_dummies(df, columns = colz)
        if to_pcl is True:
            with open('../../data/processed/rsf_dataset_reduced_no_eps_isco2.pcl', 'wb') as f:
                pickle.dump(df, f)
        return df

    def rsf_dataset(self, to_pcl = False):
        columns_to_keep = ['truncated', 'Age', 'Months_of_work_experience', 'Entry_month_sin', 'Entry_day_sin',
       'Entry_month_cos', 'Entry_day_cos', 'Gender', 'Social_benefits', 'Unemployment_benefits', 'P16_level', 'ISCO', 'Level',
        'Reason_for_PES_entry', 'eApplication', 'Reason for termination', 'duration']
        continuous = ['Age', 'Months_of_work_experience', 'Entry_month_sin', 'Entry_day_sin',
               'Entry_month_cos', 'Entry_day_cos']

        for_scaling = ['Age', 'Months_of_work_experience']
        df = self.dataset.copy()
        df = self.drop_unnecessary_columns(df)
        df = self.rename_columns(df)
        df = self.transform_entry_date(df)
        df = df[df['duration']>0]
        df = df[columns_to_keep]

        scaler = MinMaxScaler() 
        scaled_values = scaler.fit_transform(df[[col for col in df.columns if col in for_scaling]]) 
        df.loc[:,[col for col in df.columns if col in for_scaling]] = scaled_values
        colz = [col for col in df.columns if col not in continuous + ['duration', 'truncated']]
#        print(colz)
#        df[colz] = OneHotEncoder().fit_transform(df[colz])
        df = pd.get_dummies(df, columns = colz)
        if to_pcl is True:
            with open(os.path.join(DIR_PATH, '../../data/processed/rsf_dataset_reduced_no_eps_isco2.pcl', 'wb')) as f:
                pickle.dump(df, f)
        return df
    
    def mtlr_dataset(self):
        df = self.dataset
        df = self.drop_unnecessary_columns(df)
        df = self.rename_columns(df)
        df = self.transform_entry_date(df)
        df = df[df['duration']>0]
        df = df[columns_to_keep]
        le = preprocessing.LabelEncoder()
        continuous = ['Age', 'Months_of_work_experience', 'Entry_month_sin', 'Entry_day_sin',
       'Entry_month_cos', 'Entry_day_cos']
        categorical = [col for col in df.columns if col not in continuous +['duration']]
        for catt in categorical:
            df[catt] = le.fit_transform(df[catt].values)
        df_test = df.sample(frac=0.2)
        df_train = df.drop(df_test.index)
        df_val = df_train.sample(frac=0.2)
        df_train = df_train.drop(df_val.index)
        train_dataset = TabularDataset(data=df_train, cat_cols=categorical,
                             output_col='duration')
        valid_dataset = TabularDataset(data=df_val, cat_cols=categorical,
                                     output_col='duration')
        test_dataset = TabularDataset(data=df_test, cat_cols=categorical,
                                     output_col='duration')
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        valid_loader = DataLoader(valid_dataset, batch_size=64, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
        
        return (train_loader, valid_loader, test_loader)

    def rsf_split(self, test = 0.2, to_pcl = False):
        X = self.rsf_dataset().drop(columns=['duration', 'truncated'])
        y = self.get_survival_vector()
        # y = torch.tensor(self.rsf_dataset().duration.fillna(self.rsf_dataset().duration.mean()).values)
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test, random_state = self.random_state)
        print(X_train.shape, y_train.shape)
        if to_pcl is True:
            save = (X_train, X_test, y_train, y_test)
            with open(os.path.join(DIR_PATH, '../../data/processed/rsf_split_reduced_no_eps_isco2.pcl', 'wb')) as f: pickle.dump(save, f)
        return (X_train, X_test, y_train, y_test)
        # df = self.rsf_dataset()
        # # Building training and testing sets #
        # index_train, index_test = train_test_split(range(df.shape[0]), test_size = test, random_state = 11)
        # data_train = df.iloc[index_train].reset_index( drop = True )
        # data_test  = df.iloc[index_test].reset_index( drop = True )

        # # Creating the X, T and E input
        # X_train, X_test = data_train.drop(['truncated', 'duration'], axis = 1) , data_test.drop(['truncated', 'duration'], axis = 1)
        # T_train, T_test = data_train['duration'].values, data_test['duration'].values
        # E_train, E_test = data_train['truncated'].values, data_test['truncated'].values
        # return (X_train, T_train, E_train, X_test, T_test, E_test)

    def get_tensors(self):
        X_train, X_test, y_train, y_test = self.rsf_split()
        return (torch.tensor(np.vstack(X_train.values).astype('float32')),
                torch.tensor(np.vstack(X_test.values).astype('float32')),
                y_train,
                y_test)

    def get_survival_vector(self):
        survival_time = self.rsf_dataset().duration.values
        survival_status = self.rsf_dataset().truncated.values
        time_upper_limit = int(survival_time.max())
        time_tensor = torch.empty((survival_time.shape[0], time_upper_limit))
        for i,el in enumerate(survival_time):
            for j in range(time_tensor.shape[1]):
                if (survival_status[i] == 1) and (j>=el): time_tensor[i][j] = 0 #np.nan
                elif j<el: time_tensor[i][j] = 1
                else: time_tensor[i][j] = 0
        return time_tensor

    def mtlr_split(self, test = 0.2):
        pass
    
    def __str__(self): return str(self.dataset.head())

if __name__=='__main__':
    path = os.getcwd() + '/../../data/raw'
    print(path)
    filename = 'BO_truncated_mso_2018.pcl'
    data = ReducedDataset(path, filename)
    rsf = data.rsf_dataset(to_pcl = True)
    print(rsf.head())
    print(rsf.shape)
    xtr, ttr, etr, xts, tts, ets = data.rsf_split(to_pcl = True)