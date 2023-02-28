import re
import os
import random
from pathlib import Path

from transformers import (
    AutoTokenizer, AutoModelForMaskedLM,
    DataCollatorForLanguageModeling,
    AutoModelForSequenceClassification,
    Trainer, EarlyStoppingCallback,
    TrainingArguments
)
from datasets import load_dataset, concatenate_datasets, ClassLabel
from transformers.integrations import NeptuneCallback
import numpy as np
from torchinfo import summary
import evaluate
import typer
import torch


_re_real_http = re.compile(r'(https?|ftp)://[^\s/$.?#].[^\s]*', flags=re.IGNORECASE+re.MULTILINE)  # https://mathiasbynens.be/demo/url-regex @stephenhay
_re_relative_link = re.compile(r'\/[^\s/$.?#\)]+')
_re_url_masks = re.compile(r'URL|\[URL\]|<URL>|\[http\]')
_re_md_ref = re.compile(r'\[([^\]]+?)\]\((\S+?)\)')

_re_user_at = re.compile(r'\B@\w+')
_re_user_mention = re.compile(r'<MENTION_\d+>|MENTION\d+')
_re_user_mask = re.compile(r'\[USER\]|<USER>|@USER')

_re_reddit_r = re.compile(r'/?r/([^\s/]+)')
_re_reddit_u = re.compile(r'/?u/[A-Za-z0-9_-]+')

_re_rt = re.compile(r'\bRT\b')

_re_space = re.compile(r'\s+')

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

ROOT_FOLDER = Path(__file__).parent.parent.parent  # repo root folder

IS_CUDA_AVAILABLE = torch.cuda.is_available()
print('IS_CUDA_AVAILABLE', IS_CUDA_AVAILABLE)

app = typer.Typer(add_completion=False)


# task names:
# 2M
# EDOS
# 2M_hate
# none


def mask_replacements(data):
    """replace usernames, links, subreddits with tokens + remove special characters"""
    data = _re_md_ref.sub(r'\g<1> http', data)  # replace md links ref

    data = _re_reddit_r.sub('reddit http', data)  # replace subreddits with tokens
    data = _re_reddit_u.sub('USER', data)  # replace reddit usernames

    data = _re_real_http.sub('http', data)  # replace links with http token
    data = _re_url_masks.sub('http', data)  # replace URL tokens in other datasets with http tokens

    data = _re_user_at.sub('USER', data)  # replace tw @handles with USER tokens
    data = _re_user_mention.sub('USER', data)  # replace MENTION123 and <MENTION_123> tokens in misogyny datasets with user tokens
    data = _re_user_mask.sub('USER', data)  # replace [USER] tokens in misogyny datasets with user tokens

    data = _re_rt.sub('', data)  # remove tw RT
    return data


def preprocessing(example):
    data = example['text']
    data = mask_replacements(data)
    data = _re_space.sub(' ', data)  # remove double space
    data = data.strip()  # remove whitespace
    example['text_preprocessed'] = data
    return example


def create_task_dataset(task, preprocessing_mode: str):
    datasets = []

    # for full preprocessing, take precomputed preprocessing
    # for basic, take raw text and run basic preprocessing functions here (at the function end)
    text_field = 'text_preprocessed' if preprocessing_mode == 'full' else 'text'

    if '2m' in task.lower():
        data_folder = ROOT_FOLDER / 'edos_data' / 'processed'
        gab_dataset = load_dataset("csv", data_files=str(data_folder / "gab_1M_unlabelled.csv"), split='train')
        reddit_dataset = load_dataset("csv", data_files=str(data_folder / "reddit_1M_unlabelled.csv"), split='train')
        two_m_dataset = concatenate_datasets([gab_dataset, reddit_dataset])
        two_m_dataset = two_m_dataset.remove_columns([
            col for col in two_m_dataset.column_names if col != text_field
        ])
        datasets.append(two_m_dataset)

    if 'edos' in task.lower():
        data_folder = ROOT_FOLDER / 'multitask_data' / 'formatted' / 'misogyny'
        edos = load_dataset("csv", data_files=str(data_folder / "edos_train.csv"), split='train')
        edos = edos.remove_columns([col for col in edos.column_names if col != text_field])
        datasets.append(edos)

    if 'hate' in task.lower():
        hate_datasets = []
        formatted_folder = ROOT_FOLDER / 'multitask_data' / 'formatted'
        for domain in ['hate_speech', 'misogyny']:
            for filename in os.listdir(formatted_folder / domain):
                if not filename.endswith('.csv'):
                    continue

                path = formatted_folder / domain / filename
                one_dataset = load_dataset("csv", data_files=str(path), split='train')
                one_dataset = one_dataset.filter(lambda x: x['language'] == 'en')
                one_dataset = one_dataset.remove_columns([
                    col for col in one_dataset.column_names if col != text_field
                ])
                hate_datasets.append(one_dataset)

        datasets.append(concatenate_datasets(hate_datasets))

    dataset = concatenate_datasets(datasets)
    dataset = dataset.shuffle(seed=SEED)
    if preprocessing_mode == 'basic':
        dataset = dataset.map(preprocessing)

    return dataset


def process_task_dataset(dataset, tokenizer):
    def tokenize(examples):
        examples = tokenizer(
            examples['text_preprocessed'],
            truncation=True,
            return_special_tokens_mask=True
        )
        return examples

    dataset = dataset.map(
        tokenize,
        batched=True,
        remove_columns=dataset["train"].column_names
    )
    return dataset


def create_edos_dataset(tokenizer):
    def tokenize(examples):
        examples = tokenizer(examples['text_preprocessed'], padding=True, truncation=True)
        return examples

    processed_folder = ROOT_FOLDER / 'multitask_data' / 'processed' / 'hate_speech'
    ds = load_dataset('csv', data_files={
        'train': str(processed_folder / 'edos_binary_train.csv'),
        'test': str(processed_folder / 'edos_binary_val.csv'),
    })
    ds = ds.rename_column('is_sexism', 'label')

    classes = sorted(list(ds['train'].unique('label')))
    cl = ClassLabel(names=classes)
    ds = ds.cast_column('label', cl)

    ds = ds.map(tokenize, batched=True)
    return ds


def train(
        model,
        tokenizer,
        dataset,
        output_dir,
        batch_size,
        max_epochs,
        eval_steps,
        data_collator=None,
        learning_rate=1e-5,
        weight_decay=0.01,
        adam_epsilon=1e-8,
        adam_beta1=0.9,
        adam_beta2=0.999,
        warmup_steps=0,
        warmup_ratio=0.0,
        compute_metrics=None,
        metric_for_best_model='eval_loss',
        greater_is_better=False,
        report_to='none',
        save_total_limit=3,
        neptune_callback=None,
        fp16_full_eval=IS_CUDA_AVAILABLE,
):
    if neptune_callback:
        report_to = 'none'
        neptune_callback_list = [neptune_callback]
    else:
        neptune_callback_list = []

    training_args = TrainingArguments(
        output_dir=output_dir,
        report_to=report_to,

        learning_rate=learning_rate,
        lr_scheduler_type='linear',
        weight_decay=weight_decay,
        adam_epsilon=adam_epsilon,
        adam_beta1=adam_beta1,
        adam_beta2=adam_beta2,

        auto_find_batch_size=True,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=max_epochs,
        warmup_ratio=warmup_ratio,
        warmup_steps=warmup_steps,

        no_cuda=not IS_CUDA_AVAILABLE,
        fp16=IS_CUDA_AVAILABLE,
        fp16_full_eval=fp16_full_eval,

        logging_strategy='steps',
        logging_steps=eval_steps,
        evaluation_strategy='steps',
        eval_steps=eval_steps,
        save_strategy='steps',
        save_steps=eval_steps,

        metric_for_best_model=metric_for_best_model,
        greater_is_better=greater_is_better,
        load_best_model_at_end=True,
        save_total_limit=save_total_limit
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset['train'],
        eval_dataset=dataset['test'],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)] + neptune_callback_list,
    )

    trainer.train()
    return trainer


@app.command()
def main(
        base_model: str = typer.Option('roberta-base', help='ModelHub pretrained model to finetune'),
        task_name: str = typer.Option('2M', help='Label name to train'),
        learning_rate: float = typer.Option(1e-5, help='Learning Rate'),
        batch_size: int = typer.Option(16, help='Batch Size'),
        max_epochs: int = typer.Option(100, help='Number of Epochs'),
        eval_steps: int = typer.Option(10000, help='Evaluation steps'),
        preprocessing_mode: str = typer.Option('basic', help='Preprocessing mode: full (all steps) or basic (only replacements and spaces)'),
        results_folder: Path = typer.Option(ROOT_FOLDER / 'results', dir_okay=True, writable=True, help='Folder to save results'),
        save_folder: Path = typer.Option(ROOT_FOLDER / 'models' / 'dont_stop_pretraining', dir_okay=True, writable=True, help='Folder to save trained model'),
):
    """Finetune pretrained model on train set with val set evaluation. Log training to a folder"""

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForMaskedLM.from_pretrained(base_model)

    if 'none' in task_name.lower():
        print('+++++ STOP NOT STOPPING PRETRAINING +++++')
        model_save_folder = base_model  # AutoModelForSequenceClassification will be initialized from this
        neptune_callback = NeptuneCallback(
            tags=[base_model, task_name, f'{preprocessing_mode}_preproc'],
        )
        # pretrainer = Trainer(
        #     model=model,
        #     callbacks=[neptune_callback],
        # )
    else:
        print('+++++ DON\'T STOP PRETRAINING +++++')
        print(f'base_model={base_model}, task_name={task_name}, preprocessing_mode={preprocessing_mode}, batch_size={batch_size}, max_epochs={max_epochs}, eval_steps={eval_steps}')

        # summary
        print(summary(model))

        raw_dataset = create_task_dataset(task_name, preprocessing_mode)
        dataset = raw_dataset.train_test_split(test_size=0.15, seed=SEED)
        dataset = process_task_dataset(dataset, tokenizer)

        tokenizer.pad_token = tokenizer.eos_token
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm_probability=0.15,
            pad_to_multiple_of=8,
        )

        full_model_name = f'dont_stop_pretraining-{base_model}-{task_name}-{preprocessing_mode}_preproc'.replace('/', '-')
        model_save_folder = save_folder / full_model_name

        neptune_callback = NeptuneCallback(
            tags=[base_model, task_name, f'{preprocessing_mode}_preproc'],
        )
        pretrainer = train(
            model,
            tokenizer,
            dataset,
            data_collator=data_collator,
            output_dir=str(results_folder / full_model_name),
            batch_size=batch_size,
            max_epochs=max_epochs,
            eval_steps=eval_steps,
            learning_rate=learning_rate,
            adam_epsilon=1e-6,
            adam_beta1=0.9,
            adam_beta2=0.98,
            warmup_ratio=0.06,
            report_to='none',
            neptune_callback=neptune_callback,
        )

        # save model
        model_save_folder.mkdir(parents=True, exist_ok=True)
        pretrainer.save_model(str(model_save_folder))

    edos_dataset = create_edos_dataset(tokenizer)
    clf_model = AutoModelForSequenceClassification.from_pretrained(str(model_save_folder))

    # create metrics function
    metric_f1 = evaluate.load('f1')

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric_f1.compute(predictions=predictions, references=labels, average='macro')

    finetune_trainer = train(
        clf_model,
        tokenizer,
        edos_dataset,
        output_dir=ROOT_FOLDER / 'temp_results',
        batch_size=32,
        max_epochs=5,
        eval_steps=200,
        learning_rate=2e-5,
        warmup_ratio=0.1,
        compute_metrics=compute_metrics,
        metric_for_best_model='eval_f1',
        greater_is_better=True,
        save_total_limit=1,
        fp16_full_eval=False,
    )

    # get final val score
    val_data = finetune_trainer.predict(edos_dataset['test'])[-1]
    test_f1 = val_data['test_f1']
    print(val_data)
    neptune_callback.run["finetuning/final_f1"] = test_f1
    neptune_callback.run["parameters"] = {
        'base_model': base_model,
        'task_name': task_name,
        'preprocessing': preprocessing_mode,
    }


if __name__ == '__main__':
    app()
