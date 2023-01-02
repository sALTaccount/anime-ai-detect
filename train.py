import evaluate
from transformers import BeitFeatureExtractor, AutoModelForImageClassification, TrainingArguments, Trainer, AdamW
import torch
from datasets import load_dataset
from torchvision.transforms import (
    CenterCrop,
    Compose,
    Normalize,
    RandomHorizontalFlip,
    RandomResizedCrop,
    Resize,
    ToTensor,
)
import numpy as np
from pynvml import *
from PIL import Image, ImageFile

# make pillow work properly
Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True

if __name__ == '__main__':
    # Load models and dataset
    dataset = load_dataset("imagefolder", data_dir="dataset", num_proc=128)
    batch_size = 32
    beit_model = "microsoft/beit-base-patch16-224"  # 384
    feature_extractor = BeitFeatureExtractor.from_pretrained(beit_model)

    # create lookups
    labels = dataset["train"].features["label"].names
    label2id, id2label = dict(), dict()
    for i, label in enumerate(labels):
        label2id[label] = i
        id2label[i] = label

    # set up eval metric
    metric = evaluate.load("accuracy")


    def compute_metrics(eval_pred):
        predictions = np.argmax(eval_pred.predictions, axis=1)
        return metric.compute(predictions=predictions, references=eval_pred.label_ids)


    def collate_fn(examples):
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        labels2 = torch.tensor([example["label"] for example in examples])
        return {"pixel_values": pixel_values, "labels": labels2}


    # split into a training and testing dataset
    splits = dataset['train'].train_test_split(test_size=0.1)
    train_ds = splits['train']
    val_ds = splits['test']

    img_size = (feature_extractor.size['width'], feature_extractor.size['height'])

    # prevent overfitting
    normalize = Normalize(mean=feature_extractor.image_mean, std=feature_extractor.image_std)
    train_transforms = Compose(
        [
            RandomResizedCrop(img_size),
            RandomHorizontalFlip(),
            ToTensor(),
            normalize,
        ]
    )

    val_transforms = Compose(
        [
            Resize(img_size),
            CenterCrop(img_size),
            ToTensor(),
            normalize,
        ]
    )


    def preprocess_train(example_batch):
        example_batch["pixel_values"] = [
            train_transforms(image.convert("RGB")) for image in example_batch["image"]
        ]
        return example_batch


    def preprocess_val(example_batch):
        example_batch["pixel_values"] = [val_transforms(image.convert("RGB")) for image in example_batch["image"]]
        return example_batch


    def print_gpu_utilization():
        nvmlInit()
        handle = nvmlDeviceGetHandleByIndex(0)
        info = nvmlDeviceGetMemoryInfo(handle)
        print(f"GPU memory occupied: {info.used // 1024 ** 2} MB.")


    def print_summary(result):
        print(f"Time: {result.metrics['train_runtime']:.2f}")
        print(f"Samples/second: {result.metrics['train_samples_per_second']:.2f}")
        print_gpu_utilization()


    train_ds.set_transform(preprocess_train)
    val_ds.set_transform(preprocess_val)

    # load BEiT
    model = AutoModelForImageClassification.from_pretrained(
        beit_model,
        label2id=label2id,
        id2label=id2label,
        ignore_mismatched_sizes=True
    ).to("cuda")

    # set up fine-tuning
    train_args = TrainingArguments(
        f"salt\\beit-ai-anime-art",
        remove_unused_columns=False,
        evaluation_strategy="steps",
        eval_steps=10,
        save_strategy="steps",
        learning_rate=5e-5,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=4,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=1,
        warmup_ratio=0.1,
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        push_to_hub=False,
        report_to=["wandb"],
        lr_scheduler_type="cosine",
        optim='adamw_hf',
    )

    # create trainer
    trainer = Trainer(
        model,
        train_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=feature_extractor,
        compute_metrics=compute_metrics,
        data_collator=collate_fn,
    )

    # train
    train_results = trainer.train()

    # save training metrics and state
    trainer.save_model()
    trainer.log_metrics("train", train_results.metrics)
    trainer.save_metrics("train", train_results.metrics)
    trainer.save_state()

    # save best model
    metrics = trainer.evaluate()
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)
