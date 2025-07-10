import os
import configparser
import random
import pandas as pd
from tqdm import tqdm

from bake_sample import Baker

CONFIG_NAME = '../params.conf'
CONFIG_PATH = os.path.join(os.path.dirname(__file__), CONFIG_NAME)

def process_files(files_list, dataset_path, csv_name, baker: Baker):

    rows = []

    for file in tqdm(files_list):

        full_path = os.path.join(dataset_path, file)

        try:
            rows.append(baker.bake_file(full_path))
        except:
            pass
    
    df = pd.DataFrame(rows)
    df.to_csv(csv_name, index=False)

if __name__ == '__main__':

    config = configparser.ConfigParser()
    config.read(CONFIG_PATH)

    training_split = config.getfloat('Split', 'training')
    validation_split = config.getfloat('Split', 'validation')
    test_split = 1 - training_split - validation_split

    dataset_path = config.get('Params', 'output_path')
    csvs_path = config.get('Params', 'csvs_path')

    os.makedirs(csvs_path, exist_ok = True)

    file_baker = Baker()

    all_files = os.listdir(dataset_path)

    #n_samples = config.getint('Params', 'amount')
    n_samples = len(all_files)

    n_training_split = int(n_samples * training_split)
    n_validation_split = int(n_samples * validation_split)
    n_test_split = n_samples - n_training_split - n_validation_split

    # Shuffle files.
    random.shuffle(all_files)

    training_files   = all_files[:n_training_split]
    validation_files = all_files[n_training_split: (n_training_split + n_validation_split)]
    test_files       = all_files[(n_training_split + n_validation_split):]

    process_files(training_files, dataset_path, os.path.join(csvs_path, 'train.csv'), file_baker)
    process_files(validation_files, dataset_path, os.path.join(csvs_path, 'val.csv'), file_baker)
    process_files(test_files, dataset_path, os.path.join(csvs_path, 'test.csv'), file_baker)