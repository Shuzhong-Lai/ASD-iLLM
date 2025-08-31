import os
os.environ['CUDA_VISIBLE_DEVICES'] = '7'
from swift.llm import get_model_tokenizer, load_dataset, get_template, EncodePreprocessor
from swift.utils import get_logger, find_all_linears, get_model_parameter_info, plot_images, seed_everything
from swift.tuners import Swift, LoraConfig
from swift.trainers import Seq2SeqTrainer, Seq2SeqTrainingArguments
from functools import partial
from datetime import datetime

now = datetime.now()
current_time = now.strftime("%m-%d-%H:%M:%S")

logger = get_logger()
seed_everything(42)

model_id_or_path = 'Qwen/Qwen2.5-3B-Instruct'
system = 'You are a helpful assistant.'

output_dir = './output/' + model_id_or_path.split('/')[-1] + '/' + current_time

dataset = ['/home/lsz/projects/llm/ASD-LLM/training_dataset/05_03_new_sft_dataset.jsonl', '/home/lsz/projects/llm/ASD-LLM/training_dataset/05_03_new_synthesis_sft_dataset.jsonl']

data_seed = 42
max_length = 2048

split_dataset_ratio = 0
num_proc = 4

lora_rank = 8
lora_alpha = 32

# 训练超参数
training_args = Seq2SeqTrainingArguments(
    output_dir=output_dir,
    learning_rate=1e-4,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_checkpointing=True,
    weight_decay=0.1,
    lr_scheduler_type='cosine',
    warmup_ratio=0.05,
    report_to=['tensorboard'],
    logging_first_step=True,
    save_strategy='steps',
    save_steps=50,
    # eval_strategy='steps',
    eval_strategy='no',
    eval_steps=50,
    gradient_accumulation_steps=16,
    num_train_epochs=5,
    metric_for_best_model='loss',
    save_total_limit=2,
    logging_steps=5,
    dataloader_num_workers=1,
    data_seed=data_seed,
)

output_dir = os.path.abspath(os.path.expanduser(output_dir))
logger.info(f'output_dir: {output_dir}')

model, tokenizer = get_model_tokenizer(model_id_or_path, model_type='glm_edge')
# model, tokenizer = get_model_tokenizer(model_id_or_path)
logger.info(f'model_info:{model.model_info}')
template = get_template(model.model_meta.template, tokenizer, default_system=system, max_length=max_length)
template.set_mode('train')

target_modules = find_all_linears(model)
lora_config = LoraConfig(task_type='CAUSAL_LM', r=lora_rank, lora_alpha=lora_alpha,
                         target_modules=target_modules)
model = Swift.prepare_model(model, lora_config)
logger.info(f'lora_config: {lora_config}')

# 打印模型结构和训练的参数量
logger.info(f'model: {model}')
model_parameter_info = get_model_parameter_info(model)
logger.info(f'model_parameter_info: {model_parameter_info}')

train_dataset, val_dataset = load_dataset(dataset, split_dataset_ratio=split_dataset_ratio, num_proc=num_proc, seed=data_seed)

logger.info(f'train_dataset: {train_dataset}')
logger.info(f'val_dataset: {val_dataset}')
logger.info(f'train_dataset[0]: {train_dataset[0]}')

train_dataset = EncodePreprocessor(template=template)(train_dataset, num_proc=num_proc)
if val_dataset is not None:
    val_dataset = EncodePreprocessor(template=template)(val_dataset, num_proc=num_proc)
logger.info(f'encoded_train_dataset[0]: {train_dataset[0]}')

# 打印一条样本
template.print_inputs(train_dataset[0])

model.enable_input_require_grads()  # 兼容gradient checkpointing
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    data_collator=template.data_collator,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    template=template,
)
trainer.train()

last_model_checkpoint = trainer.state.last_model_checkpoint
logger.info(f'last_model_checkpoint: {last_model_checkpoint}')

images_dir = os.path.join(output_dir, 'images')
logger.info(f'images_dir: {images_dir}')
plot_images(images_dir, training_args.logging_dir, ['train/loss'], 0.9)  # 保存图片