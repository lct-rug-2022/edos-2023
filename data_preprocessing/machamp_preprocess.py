import os
import csv
from pathlib import Path

import pandas as pd


ROOT_FOLDER = Path(__file__).parent.parent  # repo root folder

DATA_DIR = str(ROOT_FOLDER / 'multitask_data' / 'processed')
RES_DIR = str(ROOT_FOLDER / 'multitask_data' / 'machamp')
os.makedirs(RES_DIR, exist_ok=True)


for task_type in os.listdir(DATA_DIR):
    if task_type.startswith('.'):
        continue

    os.makedirs(os.path.join(RES_DIR, task_type), exist_ok=True)

    for filename in os.listdir(os.path.join(DATA_DIR, task_type)):
        if filename.startswith('.'):
            continue

        df = pd.read_csv(os.path.join(DATA_DIR, task_type, filename))
        df.drop(['id', 'dataset', 'orig_id', 'text'], axis=1, inplace=True)  # taking only text_preprocessed

        if 'hate_target_groups' in filename:
            df['gender'] = df['gender'].apply(round)

        # write multilingual data as is (shuffled)
        df.to_csv(
            os.path.join(RES_DIR, task_type, filename),
            sep='\t', index=False, header=False, quoting=csv.QUOTE_NONE
        )

        # write data by language
        for lang, lang_df in df.groupby('language'):
            filename_clean, ext = filename.rsplit('.', 1)
            write_path = os.path.join(RES_DIR, task_type, f'{filename_clean}_{lang}.{ext}')
            lang_df.to_csv(write_path, sep='\t', index=False, header=False, quoting=csv.QUOTE_NONE)
