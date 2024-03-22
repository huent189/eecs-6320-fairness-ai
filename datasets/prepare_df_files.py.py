from cxr_foundation.mimic import parse_embedding_file_pattern
from cxr_foundation import embeddings_data
import pandas as pd
import numpy as np
import random as python_random
from sklearn.utils import shuffle

seed=19
np.random.seed(seed)
python_random.seed(seed)
base_path="./"
extraction_folder = base_path
df_embeddings = pd.read_csv(base_path+"generalized-image-embeddings-for-the-mimic-chest-x-ray-dataset-1.0/SHA256SUMS.txt",
                            delimiter=" ", header=None,
                            skiprows=[0])
SOURCE_COL_NAME = "embeddings_file"
# Create additional columns from file path components
df_embeddings = df_embeddings[[1]]
df_embeddings.rename(columns={1: "embeddings_file"}, inplace=True)
df_embeddings[["subject_id","study_id", "dicom_id"]] = df_embeddings.apply(
    lambda x: parse_embedding_file_pattern(x[SOURCE_COL_NAME]), axis=1, result_type="expand")
df_embeddings.embeddings_file=base_path+"generalized-image-embeddings-for-the-mimic-chest-x-ray-dataset-1.0/"+df_embeddings.embeddings_file
df_metadata = pd.read_csv(base_path+"/mimic-cxr-2.0.0-metadata.csv.gz", compression="gzip")
MIMIC_CXR_Labels_df = pd.read_csv(base_path+"/mimic-cxr-2.0.0-chexpert.csv.gz", compression="gzip")
demographic_df  = pd.read_csv(base_path+"/admissions.csv.gz", compression="gzip")
patients_df  = pd.read_csv(base_path+"/patients.csv.gz", compression="gzip")

MIMIC_CXR_Labels_df.replace(np.nan,0,inplace=True)
MIMIC_CXR_Labels_df.replace(-1,np.NAN,inplace=True)
# Remove rows with NaN values
MIMIC_CXR_Labels_df.dropna(inplace=True)

MIMIC_CXR_Labels_df = MIMIC_CXR_Labels_df.rename(columns={"Enlarged Cardiomediastinum": "Enlarged_Cardiomediastinum","Lung Lesion":"Lung_Lesion","Lung Opacity":"Lung_Opacity",
                                                        "No Finding":"No_Finding","Pleural Effusion":"Pleural_Effusion","Pleural Other":"Pleural_Other",
                                                        "Support Devices":"Support_Devices"})

demographic_df = demographic_df.drop_duplicates(subset='subject_id')

# remove patients who have inconsistent documented race information
# credit to github.com/robintibor
ethnicity_df = demographic_df.loc[:,['subject_id', 'ethnicity']].drop_duplicates()

v = ethnicity_df.subject_id.value_counts()
subject_id_more_than_once = v.index[v.gt(1)]

ambiguous_ethnicity_df = ethnicity_df[ethnicity_df.subject_id.isin(subject_id_more_than_once)]
inconsistent_race = ambiguous_ethnicity_df.subject_id.unique()

# Each study contains one or more DICOMs
data_df = df_metadata.merge(patients_df, on=['subject_id'])
data_df = data_df.drop(columns=['anchor_year','anchor_year_group','dod'])

#Merg with demographic info data frame
data_df = data_df.merge(demographic_df, on='subject_id')
# Select only the columns you want to include
demographic_columns_to_drop = ['hadm_id', 'admittime','dischtime','deathtime',
                               'admission_type','admission_location','discharge_location'
                               ,'language','marital_status','edregtime','edouttime',
                               'hospital_expire_flag']

# Merge the DataFrames and select only the desired columns
data_df = data_df.drop(columns=demographic_columns_to_drop)
data_df = data_df[~data_df.subject_id.isin(inconsistent_race)]
data_df = data_df.rename(columns={"ethnicity": "race"})
data_df = data_df[data_df.race.isin(['ASIAN','BLACK/AFRICAN AMERICAN','WHITE','OTHER',
                                     'HISPANIC/LATINO','AMERICAN INDIAN/ALASKA NATIVE'])]

# Chexpert labels df does not contain DICOM ID. Must join on (subject_id + study_id)
data_df = data_df.merge(MIMIC_CXR_Labels_df, on=['study_id','subject_id'])
data_df = df_embeddings.merge(data_df, on=['dicom_id'], how='left')
data_df=data_df.loc[:, ~data_df.columns.duplicated()]
data_df.dropna(inplace=True)
data_df.rename(columns={'subject_id_x': 'subject_id','study_id_x': 'study_id'}, inplace=True)

data_df=data_df[['embeddings_file','subject_id','study_id','dicom_id','gender','insurance',
                 'anchor_age','race','Atelectasis','Cardiomegaly','Consolidation',
                 'Edema','Enlarged_Cardiomediastinum','Fracture','Lung_Lesion','Lung_Opacity'
                 ,'No_Finding','Pleural_Effusion','Pleural_Other','Pneumonia','Pneumothorax'
                 ,'Support_Devices']]
data_df.insert(4, "split","none", True)

data_df.rename(columns={'embeddings_file': 'path'},inplace=True)

data_df['anchor_age'] = data_df['anchor_age'].astype('int64')

data_df.insert(data_df.columns.get_loc('anchor_age') + 1, 'age_decile', None)

# Define custom bin edges and labels
bin_edges = [0, 20, 40, 60, 80, float('inf')]
bin_labels = ['0-20', '20-40', '40-60', '60-80', '80+']

# Use cut to create age deciles based on custom bins
data_df['age_decile'] = pd.cut(data_df['anchor_age'],
                                    bins=bin_edges, labels=bin_labels,
                                    right=False)
unique_sub_id = data_df.subject_id.unique()

train_percent, valid_percent, test_percent = 0.80, 0.10, 0.10

unique_sub_id = shuffle(unique_sub_id)
value1 = (round(len(unique_sub_id)*train_percent))
value2 = (round(len(unique_sub_id)*valid_percent))
value3 = value1 + value2
value4 = (round(len(unique_sub_id)*test_percent))

data_df = shuffle(data_df)

train_sub_id = unique_sub_id[:value1]
validate_sub_id = unique_sub_id[value1:value3]
test_sub_id = unique_sub_id[value3:]


data_df.loc[data_df.subject_id.isin(train_sub_id), "split"]="train"
data_df.loc[data_df.subject_id.isin(validate_sub_id), "split"]="validate"
data_df.loc[data_df.subject_id.isin(test_sub_id), "split"]="test"


data_df.to_csv(base_path+"/processed_mimic_df.csv")