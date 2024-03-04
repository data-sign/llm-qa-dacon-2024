import os
import torch

from transformers import AutoTokenizer, TrainingArguments, AutoModelForCausalLM, BitsAndBytesConfig
from datasets import load_dataset
from trl import DPOTrainer
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from utils import print_trainable_parameters, find_all_linear_names

output_dir="../dpo_results"
# model_name = "../model/solar_rag3/final" #orion
model_name = 'Upstage/SOLAR-10.7B-Instruct-v1.0'

# dataset = load_dataset("Intel/orca_dpo_pairs")
from datasets import load_dataset
dataset = load_dataset("json", data_files="../data/dpo_dataset_kor_eng_0302.json", split='train')

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

model = AutoModelForCausalLM.from_pretrained(model_name
                                            , torch_dtype=torch.bfloat16
                                            , quantization_config=bnb_config
                                            , trust_remote_code=True
                                            , use_cache=False
                                            # , device_map='auto'
                                            # , device_map={'':torch.cuda.current_device()}
                                            )

# model = torch.nn.DataParallel(model, device_ids=[0,1]) # GPU 0,1,2,3 총 4개 사용
# model.cuda()

# model.to('cuda')

model.config.use_cache = False
model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)

# torch.cuda.set_device(torch.device(f'cuda:1'))
# model_ref = AutoModelForCausalLM.from_pretrained(model_name
#                                             , torch_dtype=torch.bfloat16
#                                             , quantization_config=bnb_config
#                                             , trust_remote_code=True
#                                             , use_cache=False
#                                             , device_map='auto'
#                                             # , device_map={'':torch.cuda.current_device()}
#                                             )

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# def return_prompt_and_responses(samples):
#     return {
#         "prompt": [
#             f"한국말로 핵심키워드를 담아서 대답해주세요.\n### Question: ```{input}```\n ### Answer: "
#             # f"An AI tool that corrects and rephrase user text grammar errors delimited by triple backticks to standard English.\n### Input: ```{input}```\n ### Output: "
#             for input in samples["prompt"]
#         ],
#         "chosen": samples["chosen"],
#         "rejected": samples["rejected"],
#     }

# original_columns = dataset.column_names

# dataset = dataset.map(
#     return_prompt_and_responses,
#     batched=True,
#     remove_columns=original_columns
# )

training_args = TrainingArguments(
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,
    # gradient_checkpointing =True,
    # max_grad_norm= 0.3,
    num_train_epochs=1,
    # save_steps= 300,
    learning_rate=2e-4,
    bf16=True,
    # save_total_limit=3,
    logging_steps=10,
    output_dir=output_dir,
    optim="paged_adamw_8bit", #"paged_adamw_32bit",
    # lr_scheduler_type="cosine",
    # warmup_ratio=0.05,
    remove_unused_columns=False
)

peft_config = LoraConfig(
    r=32,
    lora_alpha=16,
    target_modules=find_all_linear_names(model),
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, peft_config)
print_trainable_parameters(model)

dpo_trainer = DPOTrainer(
    model,
    ref_model=None, # None,  #model_ref
    args=training_args,
    beta=0.1,
    train_dataset=dataset,
    tokenizer=tokenizer,
    max_prompt_length=1024,
    max_length=2048,
)

dpo_trainer.train()
dpo_trainer.save_model(output_dir)

output_dir = os.path.join(output_dir, "final_checkpoint")
dpo_trainer.model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)