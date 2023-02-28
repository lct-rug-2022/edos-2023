import re
import os
import sys
import shutil
from copy import deepcopy
from pathlib import Path

import torch
import typer
import evaluate
import pandas as pd
import neptune.new as neptune

ROOT_FOLDER = Path(__file__).parent.parent.parent  # repo root folder

sys.path.append(str(ROOT_FOLDER))
sys.path.append(str(ROOT_FOLDER / 'machamp_repo'))

from machamp_repo.machamp.model import trainer
from machamp_repo.machamp.predictor.predict import predict_with_paths
from machamp_repo.machamp.utils.myutils import load_json


app = typer.Typer(add_completion=False)

CURR_DIR = Path(__file__).parent
AVAILABLE_DATASETS = load_json(str(CURR_DIR / 'datasets.json'))

FILE_LANG_RE = re.compile(r'_(\w\w)\.csv')

DEVICE = 'cuda:0'
MACHAMP_SENT_IDXS = [2]
MACHAMP_TASK = {
    "task_type": "classification",
    "metric": "f1_macro",
    "column_idx": 1
}


def create_dataset_card(dataset_name, path_to_type, train_filename):
    dev_filename = train_filename.replace('_train', '_val')
    return {
        "train_data_path": str(path_to_type / train_filename),
        "dev_data_path": str(path_to_type / dev_filename),
        "sent_idxs": deepcopy(MACHAMP_SENT_IDXS),
        "tasks": {dataset_name: deepcopy(MACHAMP_TASK)}
    }


def collect_lang_datasets(dataset_name, dataset_file, path_to_type, languages):
    dataset_train_files = {}
    for filename in os.listdir(path_to_type):
        if not filename.startswith(f'{dataset_file}_train'):
            continue

        file_lang = FILE_LANG_RE.search(filename)
        key = file_lang.group(1) if file_lang is not None else 'all'
        dataset_train_files[key] = filename

    lang_datasets = {}
    if languages == 'all':
        lang_datasets[dataset_name] = create_dataset_card(
            dataset_name, path_to_type, dataset_train_files['all']
        )
        return lang_datasets

    if languages == 'all_sep':
        languages = [k for k in dataset_train_files.keys() if k != 'all']

    # if not `all` or `all_sep`, languages must be a list
    assert isinstance(languages, list)

    for lang in languages:
        if lang in dataset_train_files:
            lang_dataset_name = dataset_name
            if 'edos' not in dataset_name.lower():  # keep consistent name for EDOS which is in en only
                lang_dataset_name += f'_{lang}'

            lang_datasets[lang_dataset_name] = create_dataset_card(
                lang_dataset_name, path_to_type, dataset_train_files[lang]
            )

    return lang_datasets


def compile_dataset_config(datasets, languages):
    dataset_config = {}

    for dataset_type, dataset_dict in AVAILABLE_DATASETS.items():
        path_to_type = ROOT_FOLDER / 'multitask_data' / 'machamp' / dataset_type

        for dataset_name, dataset_file in dataset_dict.items():
            take_all = datasets == 'all'
            take_all_from_type = isinstance(datasets, str) and dataset_type in datasets
            take_this_dataset = dataset_name in datasets
            if take_all or take_all_from_type or take_this_dataset:
                lang_datasets = collect_lang_datasets(dataset_name, dataset_file, path_to_type, languages)
                dataset_config.update(lang_datasets)

    return dataset_config


def create_neptune_run(edos_task, params, datasets, languages, project=None, tags=None):
    run = neptune.init_run(project=project)
    run['base_model'] = params['transformer_model']
    run['parameters/learning_rate'] = params['training']['optimizer']['lr']
    run['parameters/batch_size'] = params['batching']['batch_size']
    run['parameters/max_epochs'] = params['training']['num_epochs']
    run['datasets'] = datasets
    run['languages'] = languages

    run["sys/tags"].add([edos_task, params['transformer_model']])
    if tags is not None:
        run["sys/tags"].add(tags)

    return run


def get_true_labels(dataset_name, dataset_conf):
    cls_idx = dataset_conf['tasks'][dataset_name]['column_idx']  # column idx of a **task**
    true_df = pd.read_csv(dataset_conf['dev_data_path'], sep='\t', header=None)
    return true_df[true_df.columns[cls_idx]].tolist()


def run_prediction(model, serialization_dir, dataset_name, dataset_conf):
    cls_idx = dataset_conf['tasks'][dataset_name]['column_idx']  # column idx of a **task**
    output_path = os.path.join(serialization_dir, dataset_name + '.out')

    predict_with_paths(
        model,
        input_path=dataset_conf['dev_data_path'],
        output_path=output_path,
        dataset=dataset_name,
        batch_size=32,
        calc_metrics=False,
        raw_text=False,
        device=DEVICE
    )

    pred_df = pd.read_csv(output_path, sep='\t', header=None)
    return pred_df[pred_df.columns[cls_idx]].tolist()


def compute_f1(metric_f1, str_true, str_pred):
    labels2id = {label: i for i, label in enumerate(sorted(set(str_true)))}
    labels = [labels2id[label] for label in str_true]
    predictions = [labels2id[pred] for pred in str_pred]
    return metric_f1.compute(predictions=predictions, references=labels, average='macro')


@app.command()
def main(
        edos_task: str = typer.Option('A', help='EDOS task to train'),
        datasets: str = typer.Option('all', help='Datasets to use for multitasking'),  # all, all_hate_speech, all_misogyny, list of datasets
        languages: str = typer.Option('en', help='Languages to consider for multilingual setup'),  # all, all_sep, list of languages
        finetune_after: bool = typer.Option(False, is_flag=True, help='Whether to finetune on EDOS afterwards'),
        finetune_only: bool = typer.Option(False, is_flag=True, help='Whether to only finetune corresponding MaChamp model on EDOS'),
        base_model: str = typer.Option(None, help='ModelHub pretrained model to finetune'),
        learning_rate: float = typer.Option(None, help='Learning Rate'),
        batch_size: int = typer.Option(None, help='Batch Size'),
        max_epochs: int = typer.Option(None, help='Number of Epochs'),
        max_finetune_epochs: int = typer.Option(None, help='Number of Epochs to finetune on EDOS'),
        finetune_learning_rate: float = typer.Option(None, help='Learning Rate to finetune on EDOS'),
        # put here any other parameters you want to check
        save_folder: Path = typer.Option(ROOT_FOLDER / 'models' / 'machamp', dir_okay=True, writable=True, help='Folder to save trained model'),
):
    target_dataset = f'edos_{edos_task}'
    params = load_json(str(CURR_DIR / f'params_{edos_task}.json'))
    metric_f1 = evaluate.load('f1')
    run = None

    if base_model is not None:
        params['transformer_model'] = base_model

    if learning_rate is not None:
        params['training']['optimizer']['lr'] = learning_rate

    if batch_size is not None:
        params['batching']['batch_size'] = batch_size

    if max_epochs is not None:
        params['training']['num_epochs'] = max_epochs

    # add more params here if needed

    if 'all' not in languages:
        languages = [x.strip() for x in languages.split(',') if x.strip()]

    if 'all' not in datasets:
        datasets = [x.strip() for x in datasets.split(',') if x.strip()]

    if isinstance(datasets, list) and target_dataset not in datasets:
        datasets.append(target_dataset)

    dataset_config = compile_dataset_config(datasets, languages)

    dataset_str = ', '.join(datasets) if isinstance(datasets, list) else datasets
    language_str = ', '.join(languages) if isinstance(languages, list) else languages
    base_model_str = params['transformer_model'].rsplit('/', 1)[-1]
    params_str = (
        f"{params['training']['num_epochs']}epochs_"
        f"{params['training']['optimizer']['lr']}lr_"
        f"{params['batching']['batch_size']}bs"
    )

    model_name = (
        f"multitask_{edos_task}_{base_model_str}"
        f"_{dataset_str.replace(', ', '-').replace('_', '')}"
        f"_{language_str.replace(', ', '-')}"
        f"_{params_str}"
    )

    serialization_dir = str(save_folder / model_name)
    model_dir = serialization_dir

    if not finetune_only:
        run = create_neptune_run(
            edos_task=edos_task,
            params=params,
            datasets=dataset_str,
            languages=language_str
        )

        os.makedirs(str(save_folder), exist_ok=True)
        if os.path.exists(serialization_dir):
            shutil.rmtree(serialization_dir)
        os.makedirs(serialization_dir)

        trainer.train(
            name=model_name,
            parameters_config=params,
            dataset_configs=[dataset_config],
            serialization_dir=serialization_dir,
            to_predict=False,
            seed=params['random_seed'],
            device=DEVICE,
            neptune_run=run
        )

    if finetune_only or finetune_after:
        if run is not None:
            run.stop()  # there will be in a new run

        if max_finetune_epochs is not None:
            params['training']['num_epochs'] = max_finetune_epochs

        if finetune_learning_rate is not None:
            params['training']['optimizer']['lr'] = finetune_learning_rate

        dataset_config = compile_dataset_config([target_dataset], ['en'])

        run = create_neptune_run(
            edos_task=edos_task,
            params=params,
            datasets=dataset_str,
            languages=language_str,
            tags=['finetune']
        )

        param_str = (
            f"{params['training']['num_epochs']}"
            f"_{params['training']['optimizer']['lr']}"
        )

        model_name += f'_finetune_{param_str}'
        finetune_serialization_dir = str(save_folder / model_name)
        model_dir = finetune_serialization_dir

        if os.path.exists(finetune_serialization_dir):
            shutil.rmtree(finetune_serialization_dir)
        os.makedirs(finetune_serialization_dir)

        trainer.train(
            name=model_name,
            parameters_config=params,
            dataset_configs=[dataset_config],
            serialization_dir=finetune_serialization_dir,
            retrain=os.path.join(serialization_dir, 'model.pt'),
            to_predict=False,
            seed=params['random_seed'],
            device=DEVICE,
            neptune_run=run
        )

    model = torch.load(os.path.join(model_dir, 'model.pt'), map_location=DEVICE)

    scores = []
    for dataset, dataset_info in dataset_config.items():
        str_true = get_true_labels(dataset, dataset_info)
        str_pred = run_prediction(
            model,
            serialization_dir=model_dir,
            dataset_name=dataset,
            dataset_conf=dataset_info
        )
        score = compute_f1(metric_f1, str_true, str_pred)
        run[f"final_f1/{dataset}"] = score['f1']
        scores.append(score['f1'])
    run[f"final_f1/avg"] = sum(scores) / len(scores)

    run.stop()


if __name__ == '__main__':
    app()
