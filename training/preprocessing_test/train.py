from pathlib import Path
import random

from transformers import AutoConfig, AutoTokenizer, EarlyStoppingCallback, AutoModelForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding
from datasets import Dataset, DatasetDict, ClassLabel, Value, load_dataset
from transformers.integrations import NeptuneCallback
import numpy as np
from torchinfo import summary
import evaluate
import typer
import torch

from preprocessing import preprocess


ROOT_FOLDER = Path(__file__).parent.parent.parent  # repo root folder
DATA_FOLDER = ROOT_FOLDER / 'multitask_data' / 'processed' / 'hate_speech'

IS_CUDA_AVAILABLE = torch.cuda.is_available()
print('IS_CUDA_AVAILABLE', IS_CUDA_AVAILABLE)


SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True


app = typer.Typer(add_completion=False)


@app.command()
def main(
        base_model: str = typer.Option('roberta-base', help='ModelHub pretrained model to finetune'),
        task: str = typer.Option('sexism', help='Task/dataset to train edos/hate/sexism/toxicity/offense'),
        preprocess_masks: bool = typer.Option(True, help='Do masks preprocessing'),
        preprocess_hashtags: bool = typer.Option(True, help='Do hashtags preprocessing'),
        preprocess_emoji: bool = typer.Option(True, help='Do emoji preprocessing'),
        preprocess_spaces: bool = typer.Option(True, help='Do spaces preprocessing'),
        learning_rate: float = typer.Option(1e-5, help='Learning Rate'),
        batch_size: int = typer.Option(48, help='Batch Size'),
        max_epochs: int = typer.Option(5, help='Number of Epochs'),
        early_stopping_steps: int = typer.Option(5, help='Enable early stopping with n steps'),
        logging_steps: int = typer.Option(200, help='Logging steps'),
        results_folder: Path = typer.Option(ROOT_FOLDER / 'results', dir_okay=True, writable=True, help='Folder to log in'),
):
    print('+++++ TRAINING PREPROCESSING +++++')
    print(f'base_model={base_model}, task={task}, learning_rate={learning_rate}, batch_size={batch_size}, max_epochs={max_epochs}, logging_steps={logging_steps}')

    if task == 'edos':
        base_filename = 'edos_binary'
        label_field = 'is_sexism'
    elif task == 'hate':
        base_filename = 'hate_binary'
        label_field = 'is_hate'
    elif task == 'sexism':
        base_filename = 'sexism_binary'
        label_field = 'is_sexism'
    elif task == 'offense':
        base_filename = 'offense_binary'
        label_field = 'is_offense'
    elif task == 'toxicity':
        base_filename = 'toxicity_binary'
        label_field = 'is_toxicity'
    else:
        raise NotImplementedError(f'Task {task} is not supported')

    # loading dataset
    ds = load_dataset('csv', data_files={
        'train': str(DATA_FOLDER / f'{base_filename}_train.csv'),
        'val': str(DATA_FOLDER / f'{base_filename}_val.csv'),
    })
    try:
        ds = ds.remove_columns(['text_preprocessed'])
        ds = ds.remove_columns(['dataset'])
    except:
        pass
    ds = ds.rename_column(label_field, 'label')
    ds = ds.filter(lambda example: example["language"] == 'en')
    def preprocess_row(example):
        example["text"] = preprocess(example['text'], do_masks=preprocess_masks, do_emoji=preprocess_emoji, do_hashtags=preprocess_hashtags)
        return example
    ds = ds.map(preprocess_row)

    cl = ClassLabel(names=sorted(list(ds['train'].unique('label'))))
    label2id, id2label = {n: i for i, n in enumerate(cl.names)}, {i: n for i, n in enumerate(cl.names)}
    ds = ds.cast_column('label', cl)

    print('++++++++++ SAMPLES ++++++++++')
    print(ds)
    for i in range(20):
        print(ds['train'][i])

    # load pretrained model and tokenizer
    config = AutoConfig.from_pretrained(base_model, label2id=label2id, id2label=id2label)
    model = AutoModelForSequenceClassification.from_pretrained(base_model, config=config)
    tokenizer = AutoTokenizer.from_pretrained(base_model)

    # process data
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    def tokenize_function(examples):
        try:
            return tokenizer(examples['text'], padding='max_length', truncation=True)
        except:
            return tokenizer([i or '' for i in examples['text']], padding='max_length', truncation=True)
    tokenized_ds = ds.map(tokenize_function, batched=True)

    # summary
    print(summary(model))

    # create metrics function
    metric_f1 = evaluate.load('f1')

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric_f1.compute(predictions=predictions, references=labels, average='macro')

    # training
    full_model_name = f'edos-2023-preprocessing-{base_model}-{task}'.replace('/', '-')
    training_args = TrainingArguments(
        output_dir=str(results_folder / full_model_name),
        report_to=['neptune'],

        learning_rate=learning_rate,
        lr_scheduler_type='linear',
        weight_decay=0.01,

        auto_find_batch_size=True,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=max_epochs,
        warmup_steps=5,

        no_cuda=not IS_CUDA_AVAILABLE,
        # bf16=IS_CUDA_AVAILABLE,
        fp16=IS_CUDA_AVAILABLE,
        fp16_full_eval=False,

        logging_strategy='steps',
        logging_steps=logging_steps,
        evaluation_strategy='steps',
        eval_steps=logging_steps,

        save_steps=logging_steps,
        metric_for_best_model='eval_f1',
        greater_is_better=True,
        load_best_model_at_end=True,
        save_total_limit=1,

        push_to_hub=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_ds['train'],
        eval_dataset=tokenized_ds['val'],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=early_stopping_steps)] if early_stopping_steps is not None else [],
    )

    # run training
    trainer.train()
    run = NeptuneCallback.get_run(trainer)
    tags = [base_model, task, base_filename]
    if preprocess_masks:
        tags.append('preprocess_masks')
    if preprocess_emoji:
        tags.append('preprocess_emoji')
    if preprocess_hashtags:
        tags.append('preprocess_hashtags')
    if preprocess_spaces:
        tags.append('preprocess_spaces')
    run["sys/tags"].add(tags)

    # get final val score
    val_data = trainer.predict(tokenized_ds['val'])[-1]
    test_f1 = val_data['test_f1']
    print(val_data)
    run["finetuning/final_f1"] = test_f1


if __name__ == '__main__':
    app()
