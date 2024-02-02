import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, AdamW, default_data_collator, get_linear_schedule_with_warmup, AutoModelForSequenceClassification, TextGenerationPipeline, GenerationConfig
from transformers import BitsAndBytesConfig
import pandas as pd
import datasets
from datasets import Dataset
import accelerate

from omegaconf import OmegaConf
from tqdm.auto import tqdm
from torch.cuda.amp import autocast
from peft import get_peft_model, LoraConfig, prepare_model_for_kbit_training
from peft.utils.other import fsdp_auto_wrap_policy
# from utils.llama_patch import upcast_layer_for_flash_attention

from accelerate import Accelerator
from accelerate.logging import get_logger

import logging
import logging.handlers
from datetime import datetime
import numpy as np

import random
import math
import time
import json


## I HAD TO SET ... export NCCL_IGNORE_DISABLED_P2P=1 ... with a warning about NVLink
## export TOKENIZERS_PARALLELISM=false


def create_logger(cfg):
    logger = get_logger(__name__)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    current_time = datetime.now().strftime("-%m-%d-%Y-%H:%M:%S")    
    handler = logging.handlers.RotatingFileHandler(
        cfg.logfile + current_time, maxBytes=(1048576*5), backupCount=7
    )
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.logger.addHandler(handler)

    return logger


def create_dataframe(cfg):
    df1 = pd.read_excel(f'{cfg.data.data_dir}/writing-eng-1.xlsx', header=1)
    df2 = pd.read_excel(f'{cfg.data.data_dir}/writing-eng-2.xlsx', header=0)
    df = pd.concat([df1, df2])
    df.reset_index(drop=True, inplace=True)
    df.columns = ['testId', 'prompt', 'response', 'comp', 'mech', 'expr', 'overall']
    df = df[df['response'].isna()==False]
    df['prompt'] = df.apply(lambda row: str(row['prompt']).strip(), axis=1)
    df['response'] = df.apply(lambda row: str(row['response']).strip(), axis=1)

    rubric_obj = '''[{"score": 1, definition: "Candidate has no ability to write in the target language."},
{"score": 2, definition: "Writing uses only isolated words. No knowledge of grammatical structures. Excessive spelling, punctuation, and/or vocabulary mistakes are present."},
{"score": 3, definition: "Definition: Writing uses only isolated words or phrases. Grammar knowledge is very limited. Excessive spelling, punctuation, and/ or vocabulary mistakes are present."},
{"score": 4, definition: "Writing uses simple sentences, words, and/ or phrases. Candidate displays very basic knowledge of grammar structures, but makes frequent mistakes."},
{"score": 5, definition: "Definition: Writing uses simple language structures with no elaboration. Candidate displays some knowledge of grammar structures, but mistakes are present. Candidate is unable to effectively express opinions and/or explain procedures. Frequent spelling, punctuation, and/ or vocabulary mistakes are present."},
{"score": 6, definition: "Definition: Writing uses basic structures to convey meaning, but no advanced or formal structures are used correctly. Candidate demonstrates a basic understanding of grammar structures, but many mistakes are present. Candidate is unable to effectively express opinions and/or explain procedures. Spelling, punctuation, and/ or vocabulary mistakes are present."},
{"score": 7, definition: "Writing uses basic structures to convey meaning, but almost no advanced or formal structures are used correctly. Candidate demonstrates a basic understanding of grammar structures, but mistakes are present. Candidate might be unable to effectively express opinions and/ or explain procedures in a coherent manner. Spelling, punctuation, and vocabulary is good in areas of frequent usage, but mistakes are present in advanced areas.
"},
{"score": 8, definition: "Writing uses basic structures to convey meaning, but few advanced or formal structures are used correctly. Candidate understands basic grammar structures, but mistakes are present in advanced areas. Candidate might have limited ability to express opinions and explain procedures in a coherent manner. Spelling, punctuation, and vocabulary is very good in areas of frequent usage, but mistakes are present in advanced areas that may confuse the reader"},
{"score": 9, definition: "Definition: Writing uses basic and advanced structures to convey the meaning. Candidate understands basic and advanced grammar, but some mistakes are present. Candidate has basic ability to express opinions and explain procedures. Spelling, punctuation, and vocabulary is very good in areas of frequent usage, but mistakes are present in advanced areas that distract but do not confuse the reader."},
{"score": 10, definition: "Writing structure is clear and concise, but lacks style and fluidity. Candidate understands basic and advanced grammar, but a few mistakes are present. Candidate is able to express opinions and explain procedures in an informal style. Spelling, punctuation, and/or vocabulary is very good in areas of frequent and infrequent usage, but mistakes are still present."},
{"score": 11, definition: "Writing structure is clear and concise, but may lack style similar to that of a less-educated writer. Candidate use basic and advanced grammar correctly with
very minor errors. Candidate is able to express opinions and explain procedures, but may not use formal and informal styles effectively. Spelling, punctuation
and/or vocabulary mistakes are very few minor."},
{"score": 12, definition: "Writing structure is equivalent to that of a well-educated writer. Candidate is able to express opinions and explain procedures in a way that demonstrates an
ability to write formal and informal styles. Grammar, spelling, punctuation, and/or vocabulary mistakes are very minor mistakes that a native speaker would
make."},
{"score": 13, definition: "Writing structure is equivalent to that of a well-educated native writer. Complete range of linguistic nuance with no mistakes present."}]'''


    def format_instruction(sample):
        return f'''[INST] <<SYS>>
You are an English language test grader tasked with scoring written responses to English questions. Using your expertise, carefully score the below prompt and responses to determine the English proficiency of the Candidate.
<</SYS>>

Below is a grading rubric which is delimited by single ticks and example Prompts and Responses which are graded using the provided rubric to generate their Score. Please analyze the following information and then use that information to appropriatley score the Candidate's response to the prompt that measure's their English proficiency. 

`
Rubric:{rubric_obj}
`

Below are 3 examples of Prompts and Responses with their appropriate Scores. A score of 0 is the lowest score, and a score of 13 is the highest score.

Prompt: Describe a time that you were faced with an unpleasant situation either professionally or in school, and discuss what you did to overcome it.
Response: my presentation in clase for diferent temas
Score: 1

Prompt: What are the most interesting aspects of the type of work that you do now, and why?
Response: There are several interesting aspect of my work. My work is very interactive by it nature .My work involves Math and logic building .The work done my me is always highly appreciated by my clients. when my imaginations come into actions than it play a vital role in any application in which it is used.
Score: 8

Prompt: If you had to select a person to baby-sit two young children, what qualities would you look for and why?
Response: Since I have three children, this question is of great importance to me. The qualities that are absolutely mandatory are as follows: at least 16 years old; previous baby-sitting experience; a list of references of previous baby-siiting; if any of the children are girls, it would be necessary to have a female baby-sitter; a knowledge of the parents of the baby-sitter; a visible maturity; a person who likes children; good grades in school; and a desire to excel. These are necessary since they show the overall character of a person. To baby-sit my children, strength of character is most important, and these qualities show that a person has that in abundance.
Score: 13

Using the above instruction, rubric, and examples, please provide your generated score of the below Prompt and Response as a single integer value.

```
Prompt: {sample['prompt']}
Response: {sample['response']}
``` [/INST] Score: {int(sample['overall'])}'''

    def eval_format_instruction(sample):
        return f'''[INST] <<SYS>>
You are an English language test grader tasked with scoring written responses to English questions. Using your expertise, carefully score the below prompt and responses to determine the English proficiency of the Candidate.
<</SYS>>

Below is a grading rubric which is delimited by single ticks and example Prompts and Responses which are graded using the provided rubric to generate their Score. Please analyze the following information and then use that information to appropriatley score the Candidate's response to the prompt that measure's their English proficiency. 

`
Rubric:{rubric_obj}
`

Below are 3 examples of Prompts and Responses with their appropriate Scores. A score of 0 is the lowest score, and a score of 13 is the highest score.

Prompt: Describe a time that you were faced with an unpleasant situation either professionally or in school, and discuss what you did to overcome it.
Response: my presentation in clase for diferent temas
Score: 1

Prompt: What are the most interesting aspects of the type of work that you do now, and why?
Response: There are several interesting aspect of my work. My work is very interactive by it nature .My work involves Math and logic building .The work done my me is always highly appreciated by my clients. when my imaginations come into actions than it play a vital role in any application in which it is used.
Score: 8

Prompt: If you had to select a person to baby-sit two young children, what qualities would you look for and why?
Response: Since I have three children, this question is of great importance to me. The qualities that are absolutely mandatory are as follows: at least 16 years old; previous baby-sitting experience; a list of references of previous baby-siiting; if any of the children are girls, it would be necessary to have a female baby-sitter; a knowledge of the parents of the baby-sitter; a visible maturity; a person who likes children; good grades in school; and a desire to excel. These are necessary since they show the overall character of a person. To baby-sit my children, strength of character is most important, and these qualities show that a person has that in abundance.
Score: 13

Using the above instruction, rubric, and examples, please provide your generated score of the below Prompt and Response as a single integer value.

```
Prompt: {sample['prompt']}
Response: {sample['response']}
``` [/INST] Score: '''

    

    # eval_df = df.copy()
    # eval_df['qa'] = eval_df.apply(lambda row: eval_format_instruction(row), axis=1)

    df['qa'] = df.apply(lambda row: format_instruction(row), axis=1)
    df['eval_qa'] = df.apply(lambda row: eval_format_instruction(row), axis=1)

    # Filter rows where testIds occur exactly 5 times
    filtered_df = df.groupby('testId').filter(lambda x: len(x) == 5)

    # Get unique testIds from the filtered data
    selected_testIds = filtered_df['testId'].unique().tolist()
    selected_testIds = random.sample(selected_testIds, min(200, len(selected_testIds)))

    return df, selected_testIds


def create_accelerator(cfg):
    accelerator = Accelerator(log_with=cfg.tracking.report_to, project_dir=cfg.output_dir, mixed_precision='bf16', gradient_accumulation_steps=cfg.training.gradient_accumulation_steps)
    
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    return accelerator


def get_bnb_config():
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

def get_model_and_tokenizer(model_name, bnb_config=None, use_flash_attention=False):
    # model = AutoModelForCausalLM.from_pretrained(model_name,                                            
    #                                          torch_dtype=torch.bfloat16,
    #                                          trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb_config, use_cache=True, trust_remote_code=True)
    model.config.pretraining_tp = 1

    tokenizer = AutoTokenizer.from_pretrained(model_name, 
                                          trust_remote_code=True, 
                                          use_fast=True)
    
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = model.config.eos_token_id
    
    return model, tokenizer


def create_optimizer(model, cfg):
    return accelerate.utils.DummyOptim(model.parameters(), lr=cfg.training.learning_rate)


def create_dataloaders(df, tokenizer, cfg):
    if cfg.data.sample_size is not None:
        tmp_df = df.iloc[:cfg.data.sample_size]
    else:
        tmp_df = df
    qa = tmp_df['qa']

    # Convert text inputs to input tensors
    qa_tok = tokenizer(qa.tolist(), padding=True, truncation=True, return_tensors='pt', max_length=2048)
    input_ids = qa_tok['input_ids']
    attention_mask = qa_tok['attention_mask']

    data_dict = {
        'input_ids': input_ids.tolist(),
        'attention_mask': attention_mask.tolist(),
        'labels': input_ids.tolist()
    }

    dataset = Dataset.from_dict(data_dict)
    datasets = dataset.train_test_split(test_size=cfg.data.test_size)

    train_dataset = datasets["train"]
    test_dataset = datasets["test"]

    train_dataloader = DataLoader(
        train_dataset,
        collate_fn=default_data_collator,
        batch_size=cfg.training.train_batch_size,
        shuffle=True
    )

    eval_dataloader = DataLoader(
        test_dataset,
        collate_fn=default_data_collator,
        batch_size=cfg.training.eval_batch_size,
        shuffle=True
    )

    return train_dataloader, eval_dataloader


# def get_size(model):
#     print(f'{sum(p.numel() for p in model.parameters()):,}')


def main():

    # Create config and logger
    cfg = OmegaConf.load('conf/config.yaml')

    # Create accelerator
    accelerator = create_accelerator(cfg)
    accelerator.wait_for_everyone()
    
    # Create logger    
    logger = create_logger(cfg)
    logger.info('Config, Accelerator, Logger created.', main_process_only=True)

    # Import data
    df, selected_testIds = create_dataframe(cfg)
    logger.info('Dataframe created.', main_process_only=True)
    logger.info(f'Length of Dataframe: {df.shape[0]:,}.')
    logger.info(df.head())
    
    bnb_config = get_bnb_config()
    
    # Set Flash Attention
    use_flash_attention = False
    # if torch.cuda.get_device_capability()[0] >= 8:
    #     from utils.llama_patch import replace_attn_with_flash_attn
    #     print("Using flash attention")
    #     replace_attn_with_flash_attn()
    #     use_flash_attention = True 
     
    # Initialize model and tokenizer
    model, tokenizer = get_model_and_tokenizer(cfg.model.model_name, bnb_config, use_flash_attention)
    logger.info('Model and Tokenizer created.')

    # Transform data and create dataloaders
    train_dataloader, eval_dataloader = create_dataloaders(df, tokenizer, cfg)
    logger.info('Dataloaders created.')

    # Calculate total number of training steps
    total_num_steps = math.ceil(
            cfg.training.num_epochs * (len(train_dataloader) / cfg.training.gradient_accumulation_steps)
        )
    
    # Define loss function and optimizer
    optimizer = create_optimizer(model, cfg)
    lr_scheduler = accelerate.utils.DummyScheduler(optimizer, total_num_steps=total_num_steps, warmup_num_steps=cfg.training.lr_warmup_steps)
    logger.info('Optimizer and LR Scheduler created.', main_process_only=True)

    peft_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.1,
        r=64,
        bias="none",
        task_type = "CAUSAL_LM",
        # target_modules=[
        #     "q_proj",
        #     "k_proj",
        #     "v_proj",
        #     "o_proj",
        #     "gate_proj",
        #     "up_proj",
        #     "down_proj"
        #     ]
    )

    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, peft_config)
    logger.info(model.print_trainable_parameters())

    (model, optimizer, train_dataloader, eval_dataloader, lr_scheduler) = accelerator.prepare(
            model, optimizer, train_dataloader, eval_dataloader, lr_scheduler)
    logger.info('Accelerator prepared.', main_process_only=True)

    if accelerator.is_main_process:
        experiment_config = vars(cfg)
        accelerator.init_trackers(cfg.output_dir)
    
    progress_bar = tqdm(
        range(total_num_steps),
        disable=not accelerator.is_local_main_process,
    )
    
    # Set starting variables
    epoch_durations = []
    eval_checkpoint_durations = []
    completed_steps = 0

     # Training loop
    for epoch in range(cfg.training.num_epochs):
        logger.info(f'Starting epoch: {epoch}')
        model.train()
        total_loss = 0
        train_losses = []
        eval_checkpoint_start_time = time.time()

        # Loop through training set
        for step, batch in enumerate(train_dataloader):
            step_start_time = time.time()
    
            if step > cfg.training.max_train_steps_per_epoch:
                break
            outputs = model(**batch)
            loss = outputs.loss
            train_losses.append(
                accelerator.gather(loss.repeat(cfg.training.train_batch_size))
            )

            # We keep track of the loss at each step
            total_loss += loss.detach().float()
            loss = loss / cfg.training.gradient_accumulation_steps

            # Backward pass
            accelerator.backward(loss)

            # Complete step
            if (step % cfg.training.gradient_accumulation_steps == 0):
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)
                completed_steps += 1
                train_losses_tensor = torch.cat(train_losses)
                train_loss = torch.mean(train_losses_tensor)
                step_duration = time.time() - step_start_time
                accelerator.log({"train_loss": train_loss, "epoch": epoch, "step_duration": step_duration}, step=completed_steps)
                logger.info(f'epoch: {epoch}; step {step}; train_loss: {train_loss}; step_duration {step_duration}')
                
            # Run evaluation
            if step % cfg.training.eval_every == 0 and step > 0:
                # correct = torch.Tensor([0])
                # total = torch.Tensor([0])
                # failed = torch.Tensor([0])
                # avg_dist = torch.Tensor([0])
                correct, total, failed, avg_dist = 0, 0, 0, 0
                pred_scores = []
                actual_scores = []
                model.eval()

                for eval_step, testId in enumerate(selected_testIds):
                    device_num = torch.cuda.current_device()
                    if device_num == 0:
                        print(f'testId: {testId}')
                        print(f'eval_step: {eval_step}')

                    if eval_step >= cfg.training.max_eval_steps:
                        break
                    if device_num == 0:
                        print(f'Test number: {eval_step}')
                    tmp_df = df[df['testId']==testId]
                    data_dict = {
                        'qa': tmp_df['eval_qa'],
                        # 'prompt': tmp_df['prompt'],
                        # 'response': tmp_df['response'],
                        'overall': tmp_df['overall'].tolist()
                    }
                    test_dataset = Dataset.from_dict(data_dict)

                    tmp_score = 0
                    tmp_total = 0

                    for i, row in enumerate(test_dataset):
                        input_ids = tokenizer(row['qa'], return_tensors="pt", truncation=True).input_ids.cuda()
                        with torch.no_grad():
                            outputs = model.generate(input_ids=input_ids, max_new_tokens=10, do_sample=True, top_p=0.9, temperature=0.1)
                        
                        generated_score = tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0][len(row['qa']):]
                        if device_num == 0:
                            print(f'Question score: {generated_score}')
                        try:
                            # obj = json.loads(generated_score)
                            # tmp_score += obj['score']
                            tmp_score += int(generated_score)
                            tmp_total += 1
                        except:
                            logger.info("Not able to convert score to integer")
                            failed +=1

                    try:
                        test_score = round(tmp_score/tmp_total)
                        if device_num == 0:
                            print(f'Predicted test score: {test_score}')
                            print(f'Actual test score: {test_dataset["overall"][0]}')

                        pred_scores.append(test_score)
                        actual_scores.append(test_dataset['overall'][0])

                        if test_score==test_dataset['overall'][0]:
                            correct += 1
                        avg_dist += abs(test_score-test_dataset['overall'][0])
                    except:
                        logger.info("Unable to calculate test score")
                    total += 1

                # Read Out
                try:
                    accelerator.wait_for_everyone()
                    # if device_num == 0:
                    # print('\n\na\n\n')
                    # total = accelerator.gather(total)
                    # print('\n\nb\n\n')
                    # correct = accelerator.gather(correct)
                    # print('\n\nc\n\n')
                    # failed = accelerator.gather(failed)
                    # print('\n\nd\n\n')
                    # avg_dist = accelerator.gather(avg_dist)
                    # print('\n\ne\n\n')
                    if device_num == 0:
                        print(f'\nDEVICE:0 Total: {total}\n')
                    print(f'\nTOTAL TOTAL {total}\n')

                    print(f'Accuracy: {(correct / total) * 100:.2f}; Failed: {failed}; Average distance: {avg_dist/total}; device_num: {device_num}')
                    # logger.info(f"Accuracy: {(correct / total) * 100:.2f}%")
                    # logger.info(f'This many failed: {failed}')
                    # logger.info(f'Average distance: {avg_dist/total}')

                except Exception as error:
                    logger.info("Unable to perform a valid readout")
                    logger.info(f"Readout Error: {error}")


                model.train()

        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(cfg.output_dir)


if __name__ == "__main__":
    main()