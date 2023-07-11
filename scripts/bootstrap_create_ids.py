"""
2. Boostraping: Crear datasets aleatorios del conjunto de entrenamiento.
"""
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm


PROJECT_ROOT = Path.cwd()

OUTPUT_DATA_DIR = PROJECT_ROOT / 'data'
DATASET_NAME = 'train_dataset.csv'

def main():
    """Creates the csv files with the ids of the subjects used to train the normative model."""
    # ----------------------------------------------------------------------------------------
    n_bootstrap = 10
    ids_path = OUTPUT_DATA_DIR / 'datasets' / DATASET_NAME
    # ----------------------------------------------------------------------------------------
    # Create experiment's output directory
    bootstrap_dir = OUTPUT_DATA_DIR / 'bootstrap_ids'
    bootstrap_dir.mkdir(exist_ok=True)

    version = 'v1'
    ids_dir = bootstrap_dir / version
    ids_dir.mkdir(exist_ok=True)

    # Set random seed for random sampling of subjects
    np.random.seed(42)

    dataset_df = pd.read_csv(ids_path)
    dataset_df = dataset_df[dataset_df['GRADUADO_A_TIEMPO'] == 0] # Only normal samples (not graduate on time)

    # ----------------------------------------------------------------------------------------
    # Get a mini dataset for each age
    # Con esto aseguramos de que en cada bootstrap haya un sujeto por cada edad disponible en el dataset
    # Esto con el fin de evitar conflictos en el one hot encoding de entrenamiento
    unique_ages_df = dataset_df.copy()
    unique_ages_df.drop_duplicates(subset=['RANGO_EDAD'], keep='first', inplace=True)
    
    dataset_df = dataset_df[~dataset_df['IDX_STUDENT'].isin(unique_ages_df['IDX_STUDENT'])]
    # ----------------------------------------------------------------------------------------
    
    n_sub = len(dataset_df)
    #train_size = 0.80

    for i_bootstrap in tqdm(range(n_bootstrap)):
        bootstrap_ids = dataset_df.sample(n_sub, replace=True)
        #bootstrap_ids = dataset_df.sample(frac=train_size, replace=True)
        ids_filename = 'cleaned_bootstrap_{:03d}.csv'.format(i_bootstrap)

        final_bootstrap_ids = pd.concat([bootstrap_ids, unique_ages_df])
        final_bootstrap_ids.to_csv(ids_dir / ids_filename, index=False)

    print("IDs boostrap saved...")


if __name__ == "__main__":
    main()