import os
import sys
from pathlib import Path
from zipfile import ZipFile

import typer
import torch
import pandas as pd

ROOT_FOLDER = Path(__file__).parent.parent.parent  # repo root folder

sys.path.append(str(ROOT_FOLDER))
sys.path.append(str(ROOT_FOLDER / 'machamp_repo'))

from machamp_repo.machamp.predictor.predict import predict_with_paths


IS_CUDA_AVAILABLE = torch.cuda.is_available()
print('IS_CUDA_AVAILABLE', IS_CUDA_AVAILABLE)

DEVICE = 'cuda:0' if IS_CUDA_AVAILABLE else 'cpu'

RESULTS_TO_LABEL = {
    'A': {
        0: 'not sexist',
        1: 'sexist'
    }
}

app = typer.Typer(add_completion=False)

DEV_TEST_FILES_BY_TASK = {
    'A_dev': ROOT_FOLDER / 'edos_data' / 'processed' / 'dev_task_a_entries.csv',
    'A_test': ROOT_FOLDER / 'edos_data' / 'processed' / 'test_task_a_entries.csv',
    'B_dev': ROOT_FOLDER / 'edos_data' / 'processed' / 'dev_task_b_entries.csv',
    'B_test': ROOT_FOLDER / 'edos_data' / 'processed' / 'test_task_b_entries.csv',
    'C_dev': ROOT_FOLDER / 'edos_data' / 'processed' / 'dev_task_c_entries.csv',
    'C_test': ROOT_FOLDER / 'edos_data' / 'processed' / 'test_task_c_entries.csv',
}


def run_prediction(serialization_dir, model_name, dataset_name, run_id, items):
    model = torch.load(os.path.join(serialization_dir, model_name), map_location=DEVICE)
    in_path = os.path.join(serialization_dir, f'{dataset_name}_{run_id}.inp')
    item_str = ''.join('\n\t\t' + item for item in items).strip('\n')  # don't even ask
    with open(in_path, 'w') as f:
        f.write(item_str)

    cls_idx = 1  # hardcoded
    output_path = os.path.join(serialization_dir, f'{dataset_name}_{run_id}.out')

    predict_with_paths(
        model,
        input_path=in_path,
        output_path=output_path,
        dataset=dataset_name,
        batch_size=32,
        calc_metrics=False,
        raw_text=False,
        device=DEVICE
    )

    r = []
    with open(output_path) as f:
        for line in f.readlines():
            cls = line.split('\t')[cls_idx].strip()
            if cls.isdigit():
                cls = int(cls)
            r.append(cls)

    return r


def predict_to_file(
        task,
        dataset_ids,
        results,
        folder,
        filename_postfix='',
):
    print('Predicting to file:', filename_postfix)

    predict_labels = [RESULTS_TO_LABEL.get(task, {}).get(i, i) for i in results]

    pd.DataFrame(
        zip(dataset_ids, predict_labels),
        columns=['rewire_id', 'label_pred']
    ).to_csv(folder / f'prediction_{filename_postfix}.csv', index=None)

    with ZipFile(folder / f'prediction_{filename_postfix}.zip', 'w') as myzip:
        myzip.write(folder / f'prediction_{filename_postfix}.csv', arcname=f'prediction_{filename_postfix}.csv')


@app.command()
def main(
        task: str = typer.Option('A', help='EDOS task to predict'),
        model: str = typer.Option('none', help='Pretrained model Path'),
        model_file: str = typer.Option('model.pt', help='Pretrained model file like model.pt'),
):
    print(f'-- {model} --')

    model_path = Path(model) if '/' in model else ROOT_FOLDER / 'models' / 'machamp' / model

    for part in ['dev', 'test']:
        df = pd.read_csv(DEV_TEST_FILES_BY_TASK[f'{task}_{part}'])
        print(f'df_{part}', len(df))

        res = run_prediction(
            serialization_dir=str(model_path),
            model_name=model_file,
            dataset_name=f'edos_{task}',
            run_id='testing',
            items=list(df['text_preprocessed']),
        )  # res = [1, 0]
        print(f'res {part}', len(res))

        predict_to_file(
            task=task,
            dataset_ids=list(df['rewire_id']),
            results=res,
            folder=model_path,
            filename_postfix=f'{task}-{model_file}-{part}'
        )


if __name__ == '__main__':
    app()
