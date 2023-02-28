import json
import random
from pathlib import Path
from zipfile import ZipFile

import pandas as pd
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer, EarlyStoppingCallback, AutoConfig,
    TrainingArguments, DataCollatorWithPadding
)
from datasets import load_dataset, ClassLabel
from transformers.integrations import NeptuneCallback
import numpy as np
import evaluate
import typer
import torch
from neptune.new.types import File


SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True


ROOT_FOLDER = Path(__file__).parent.parent.parent  # repo root folder

with open(Path(__file__).parent / 'edos_eval_params.json') as f:
    EDOS_EVAL_PARAMS = json.load(f)

EDOS_LABEL_BY_TASK = {
    'A': 'label_sexist',
    'B': 'label_category',
    'C': 'label_vector',
}
EDOS_TRAIN_FILE = ROOT_FOLDER / 'edos_data' / 'processed' / 'edos_2023_train.csv'
EDOS_VAL_FILE = ROOT_FOLDER / 'edos_data' / 'processed' / 'edos_2023_val.csv'


DEV_TEST_FILES_BY_TASK = {
    'A_dev': ROOT_FOLDER / 'edos_data' / 'processed' / 'dev_task_a_entries.csv',
    'A_test': ROOT_FOLDER / 'edos_data' / 'processed' / 'test_task_a_entries.csv',
    'B_dev': ROOT_FOLDER / 'edos_data' / 'processed' / 'dev_task_b_entries.csv',
    'B_test': ROOT_FOLDER / 'edos_data' / 'processed' / 'test_task_b_entries.csv',
    'C_dev': ROOT_FOLDER / 'edos_data' / 'processed' / 'dev_task_c_entries.csv',
    'C_test': ROOT_FOLDER / 'edos_data' / 'processed' / 'test_task_c_entries.csv',
}


IS_CUDA_AVAILABLE = torch.cuda.is_available()
print('IS_CUDA_AVAILABLE', IS_CUDA_AVAILABLE)


app = typer.Typer(add_completion=False)


def create_edos_dataset(tokenizer, task):
    def tokenize(examples):
        try:
            examples = tokenizer(examples['text_preprocessed'], truncation=True, padding='do_not_pad')
        except:
            examples = tokenizer([i or '' for i in examples['text_preprocessed']], truncation=True, padding='do_not_pad')
        return examples

    ds = load_dataset('csv', data_files={
        'train': str(EDOS_TRAIN_FILE),
        'test': str(EDOS_VAL_FILE),
    })
    ds = ds.rename_column(EDOS_LABEL_BY_TASK[task], 'label')

    # remove other columns to supress warnings
    ds.remove_columns([v for k, v in EDOS_LABEL_BY_TASK.items() if k != task])
    ds.remove_columns(['id', 'orig_id', 'text_preprocessed', 'text'])

    ds = ds.filter(lambda x: str(x['label']).lower() != 'none')  # filter 'none' in second task dataset

    classes = sorted(list(ds['train'].unique('label')))
    cl = ClassLabel(names=classes)
    ds = ds.cast_column('label', cl)
    label2id, id2label = {n: i for i, n in enumerate(cl.names)}, {i: n for i, n in enumerate(cl.names)}

    ds = ds.map(tokenize, batched=True)
    return ds, label2id, id2label


def load_dev_train_dataset(tokenizer, task):
    def tokenize(examples):
        try:
            examples = tokenizer(examples['text_preprocessed'], truncation=True, padding='do_not_pad')
        except:
            examples = tokenizer([i or '' for i in examples['text_preprocessed']], truncation=True, padding='do_not_pad')
        return examples

    data_files = {
        k: str(v)
        for k, v in DEV_TEST_FILES_BY_TASK.items()
        if v.exists() and k.startswith(task)
    }
    print('data_files', data_files)

    ds = load_dataset('csv', data_files=data_files)
    ds = ds.map(tokenize, batched=True)

    return ds


def predict_to_file(
        trainer,
        dataset,
        int2str,
        folder,
        neptune_run,
        filename_postfix='',
):
    print('Predicting to file:', filename_postfix)

    predict_logist = trainer.predict(dataset)[0]
    predict_int = np.argmax(predict_logist, axis=-1)
    predict_labels = [int2str[i] for i in predict_int]

    pd.DataFrame(
        zip(dataset['rewire_id'], predict_labels),
        columns=['rewire_id', 'label_pred']
    ).to_csv(folder / f'prediction_{filename_postfix}.csv', index=None)

    with ZipFile(folder / f'prediction_{filename_postfix}.zip', 'w') as myzip:
        myzip.write(folder / f'prediction_{filename_postfix}.csv', arcname=f'prediction_{filename_postfix}.csv')

    neptune_run[f'predictions/prediction_{filename_postfix}.csv'].upload(File(str(folder / f'prediction_{filename_postfix}.csv')))
    neptune_run[f'predictions/prediction_{filename_postfix}.zip'].upload(File(str(folder / f'prediction_{filename_postfix}.zip')))


@app.command()
def main(
        base_model: str = typer.Option('roberta-base', help='Pretrained model to finetune: HUB or Path'),
        edos_task: str = typer.Option('A', help='EDOS to train'),
        config_name: str = typer.Option('default', help='Config name to use: default, updated-large'),
        results_folder: Path = typer.Option(ROOT_FOLDER / 'results', dir_okay=True, writable=True, help='Folder to save results'),
        save_folder: Path = typer.Option(ROOT_FOLDER / 'models' / 'dont_stop_pretraining_eval', dir_okay=True, writable=True, help='Folder to save trained model'),
):
    full_model_path_to_load = base_model
    short_model_name_to_log = base_model.split('/')[-1]
    model_name_to_save = f'finetuning-{base_model}-edos-task-{edos_task}-{config_name}'.replace('/', '-')
    model_save_folder = save_folder / model_name_to_save

    # load pretrained tokenizer
    tokenizer = AutoTokenizer.from_pretrained(full_model_path_to_load)
    # load dataset
    dataset, label2id, id2label = create_edos_dataset(tokenizer, edos_task)
    dataset_dev_train = load_dev_train_dataset(tokenizer, edos_task)
    # load pretrained model
    config = AutoConfig.from_pretrained(full_model_path_to_load, label2id=label2id, id2label=id2label)
    model = AutoModelForSequenceClassification.from_pretrained(full_model_path_to_load, config=config)

    # create metrics function
    metric_f1 = evaluate.load('f1')

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric_f1.compute(predictions=predictions, references=labels, average='macro')

    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer,
        padding='longest',
    )

    params = EDOS_EVAL_PARAMS[edos_task][config_name]

    neptune_callback = NeptuneCallback(
        tags=[short_model_name_to_log, f'edos:{edos_task}', f'conf:{config_name}'],
    )

    training_args = TrainingArguments(
        output_dir=str(results_folder / model_name_to_save),
        report_to='none',

        learning_rate=params['learning_rate'],
        lr_scheduler_type='linear',
        weight_decay=0.01,

        per_device_train_batch_size=params['batch_size'],
        per_device_eval_batch_size=params['batch_size'],
        num_train_epochs=params['max_epochs'],
        warmup_ratio=params.get('warmup_ratio', 0.01),
        warmup_steps=params.get('warmup_steps', 0),

        no_cuda=not IS_CUDA_AVAILABLE,
        fp16=IS_CUDA_AVAILABLE,
        fp16_full_eval=params.get('fp16_full_eval', False),

        logging_strategy='steps',
        logging_steps=params['eval_steps'],
        evaluation_strategy='steps',
        eval_steps=params['eval_steps'],
        save_strategy='steps',
        save_steps=params['eval_steps'],

        metric_for_best_model='eval_f1',
        greater_is_better=True,
        load_best_model_at_end=True,
        save_total_limit=1,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset['train'],
        eval_dataset=dataset['test'],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5), neptune_callback],
    )

    trainer.train()

    model_save_folder.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(model_save_folder))

    val_data = trainer.predict(dataset['test'])[-1]
    test_f1 = val_data['test_f1']
    print(val_data)
    neptune_callback.run["finetuning/final_f1"] = test_f1
    neptune_callback.run["parameters"] = {
        'full_model': base_model,
        'model': short_model_name_to_log,
        'edos_task': edos_task,
        'config_name': config_name,
    }

    predict_to_file(trainer, dataset_dev_train[f'{edos_task}_dev'], id2label, results_folder / model_name_to_save, neptune_callback.run, f'{edos_task}_dev')
    if f'{edos_task}_test' in dataset_dev_train:
        predict_to_file(trainer, dataset_dev_train[f'{edos_task}_test'], id2label, results_folder / model_name_to_save, neptune_callback.run, f'{edos_task}_test')


if __name__ == '__main__':
    app()
