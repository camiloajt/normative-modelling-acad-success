"""
1. Dividir el conjunto de datos en un conjunto de entrenamiento y de prueba.
train dataset: 85% de los datos pertenecientes al grupo no graduado a tiempo (0).
test dataset: 15% de los datos pertenecientes al grupo no graduado a tiempo, 100% de los datos graduados a tiempo (1).
"""
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path.cwd()

OUTPUT_DATA_DIR = PROJECT_ROOT / 'data' / 'datasets'
ORIGINAL_DATASET_NAME = 'original_dataset.csv'

TRAIN_DATASET_NAME = 'train_dataset.csv'
TEST_DATASET_NAME = 'test_dataset.csv'

def main():
    output_dir = PROJECT_ROOT / 'data' / 'datasets'
    output_dir.mkdir(exist_ok=True)

    data_dir = OUTPUT_DATA_DIR / ORIGINAL_DATASET_NAME
    dataset_df = pd.read_csv(data_dir)

    #-------------------------------------------------
    # Getting dataframes by group(diagnostic)
    VAR_TARGET_NAME = 'GRADUADO_A_TIEMPO'
    normal_dataset_df = dataset_df[dataset_df[VAR_TARGET_NAME] == 0] #0: No graduados a tiempo
    abnormal_dataset_df = dataset_df[dataset_df[VAR_TARGET_NAME] == 1] #1: Graduados a tiempo

    # ----------------------------------------------------------------------------------------
    # Get a mini dataset train for each age
    # Con esto nos aseguramos de que en el dataset de entrenamiento haya un sujeto por cada edad disponible en el dataset
    unique_ages_df = normal_dataset_df.copy()
    unique_ages_df.drop_duplicates(subset=['RANGO_EDAD'], keep='first', inplace=True)
    
    normal_dataset_df = normal_dataset_df[~normal_dataset_df['IDX_STUDENT'].isin(unique_ages_df['IDX_STUDENT'])]

    #-------------------------------------------------
    # Split dataset in train data and test data
    test_size=0.1
    train_data, normal_test_data = train_test_split(normal_dataset_df, test_size=test_size, random_state=42)

    train_data = pd.concat([train_data, unique_ages_df], axis=0) #Uniendo los dataframe de entrenamiento
    print("Train normal data size: {}".format(train_data.shape[0]))
    print("Test normal data size: {}".format(normal_test_data.shape[0]))

    #-------------------------------------------------
    # Concat abnormal data with test dataset
    test_data = pd.concat([normal_test_data, abnormal_dataset_df], axis=0)
    
    #-------------------------------------------------
    print(train_data.shape)
    print(test_data.shape)

    print("Saving new dataset {dataset}".format(dataset=TRAIN_DATASET_NAME))
    print("Saving new dataset {dataset}".format(dataset=TEST_DATASET_NAME))

    # Exporting datasets
    #train_data.to_csv(output_dir / TRAIN_DATASET_NAME, index=False)
    #test_data.to_csv(output_dir / TEST_DATASET_NAME, index=False)


if __name__== "__main__":
    main()