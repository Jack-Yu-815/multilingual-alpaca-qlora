import os
import sys
from typing import List

from dataclasses import dataclass
import fire
import torch
import transformers
import datasets
from datasets import load_dataset, concatenate_datasets
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, LlamaTokenizerFast, RwkvForCausalLM
from trl import DataCollatorForCompletionOnlyLM, SFTTrainer, SFTConfig
from peft import prepare_model_for_kbit_training
from safetensors.torch import load_file
import torch.nn.functional as F
from tqdm.auto import tqdm
import wandb

from utils.smart_tokenizer import smart_tokenizer_and_embedding_resize
from utils.huggingface import upload_model_to_hf
"""
Unused imports:
import torch.nn as nn
import bitsandbytes as bnb
"""

from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    # prepare_model_for_int8_training,
    set_peft_model_state_dict,
)
from transformers import LlamaForCausalLM, LlamaTokenizer

from utils.prompter import Prompter
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from transformers.trainer_callback import TrainerCallback


def zip_parallel_datasets(english_dataset, *parallel_datasets, val_size=0.1):
    """
    Zips datasets together with round-robin target language selection. The new dataset will be twice 
    as long as the original English dataset, with each English instruction followed by its translation
    in the selected target language.
    
    The target language for each English instruction is selected in round-robin fashion. For example,
    with English, German, French, Spanish, Italian:
    - English1 pairs with German
    - English2 pairs with French  
    - English3 pairs with Spanish
    - English4 pairs with Italian
    - English5 pairs with German (cycle starts over)
    - and so on...

    Format:
    - English instructions are at even indices (0, 2, ...), and target languages are at odd indices (1, 3, ...).
    - This ensures that when batch size is a multiple of 2, we can always determine pairs of 
      <English, target-language> instructions.
    - The last partial batch may need to be discarded.

    Args:
        english_dataset: The base English dataset
        *datasets: Variable number of target language datasets (German, French, etc.)
        
    Example:
        zip_parallel_datasets(english_data, german_data, french_data, spanish_data, italian_data)

    Returns:
        A DatasetDict containing 'train', and 'validation' splits of the interleaved data.
    """
    if not 0 <= val_size < 1:
        raise ValueError("Invalid split proportions.")

    # Shuffle all datasets with the same seed to maintain correspondence between them, then zip
    seed = 42
    english_dataset = english_dataset.shuffle(seed=seed)
    # Keep each language dataset separate but shuffled
    target_datasets = tuple(dataset.shuffle(seed=seed) for dataset in parallel_datasets)
    
    # Create interleaved dataset
    combined_examples = []
    num_english = len(english_dataset)
    num_target_langs = len(target_datasets)
    
    for idx in range(num_english):
        # Add English example
        combined_examples.append(english_dataset[idx])
        # Add corresponding target language example
        target_lang_idx = idx % num_target_langs
        combined_examples.append(target_datasets[target_lang_idx][idx])
    
    # Create the combined dataset
    full_dataset = datasets.Dataset.from_list(combined_examples)
    
    # Split into train/validation/test sets
    splits = full_dataset.train_test_split(test_size=val_size, seed=seed)
    
    return datasets.DatasetDict({
        'train': splits['train'],
        'test': splits['test']
    })


@dataclass
class LanguageContrastiveSFTConfig(SFTConfig):
    contrastive_loss_ratio: float = 1.0


class LanguageContrastiveSFTTrainer(SFTTrainer):
    def find_first_non_ignore_index(self, tensor):
        # Find the first index in each row where the value doesn't match the given number
        ignore_index = self.data_collator.ignore_index
        indices = []
        for row in tensor:
            # Get indices where the row elements are not equal to the number
            not_matching_indices = (row != ignore_index).nonzero(as_tuple=True)[0]
            # Append the first such index, or -1 if all elements match the number
            indices.append(not_matching_indices[0].item() if not_matching_indices.numel() > 0 else -1)
        return torch.tensor(indices, device=tensor.device)

    def compute_contrastive_loss(self, outputs, first_non_ignore_indices):
        # TODO: Implement contrastive loss
        # # Get the first decoder layer's hidden states from the model. (hidden_states[0] is the embedding outbut. hidden_states[1] is the first layer output)
        # first_layer_hidden_states = outputs.hidden_states[1]
        # # find pairs of hidden states that are supposed to be contrasted
        # 
        # # Compute the contrastive loss
        """
        Compute contrastive loss based on hidden states of paired examples.
        Assumes batch is organized as [en1, tr1, en2, tr2, ...] where:
        - en1 and tr1 form a positive pair
        - en2 and tr2 form a positive pair
        - all other combinations are negative pairs
        """
        # Get first decoder layer hidden states
        hidden_states = outputs.hidden_states[1]  # shape: [batch_size, seq_len, hidden_dim]
        batch_size = hidden_states.size(0)
        
        if batch_size % 2 != 0:
            return 0  # Skip contrastive loss if batch size is odd
        
        # Extract sentence representations using the first non-ignore indices
        # Shape: [batch_size, hidden_dim]
        sentence_reps = [hidden_states[i, 0:first_non_ignore_indices[i]] for i in range(batch_size)]  # A list of [input_len, hidden_dim] arrays
        
        # # Normalize the representations (important for stable training)
        # sentence_reps = [F.normalize(hidden, p=2, dim=1) for hidden in sentence_reps]  # A list of [input_len, hidden_dim] arrays (normalized)

        # apply mean pooling
        sentence_reps = [torch.mean(hidden, dim=0) for hidden in sentence_reps]  # A list of [hidden_dim] arrays
        sentence_reps = torch.stack(sentence_reps)  # Shape: [batch_size, hidden_dim]
        
        # Calculate similarity matrix
        # Shape: [batch_size, batch_size]
        temperature = 0.05  # self.model.config.temperature
        sim_matrix = torch.matmul(sentence_reps, sentence_reps.transpose(0, 1)) / temperature
        
        # Create mask for positive pairs
        # For batch [en1, tr1, en2, tr2], positives are (en1,tr1) and (en2,tr2)
        labels = torch.zeros_like(sim_matrix)
        for i in range(0, batch_size, 2):
            if i + 1 < batch_size:
                # Mark (en->tr) and (tr->en) as positive pairs
                labels[i, i+1] = 1
                labels[i+1, i] = 1
        
        # For numerical stability
        sim_matrix_exp = torch.exp(sim_matrix)
        
        # Calculate loss for each example
        loss = 0
        num_pairs = 0
        for i in range(batch_size):
            positive_pairs = labels[i].nonzero().squeeze(dim=-1)

            # Denominator sums over all except self (diagonal)
            denominator = sim_matrix_exp[i].sum()

            if positive_pairs.numel() > 0:  # if this example has a positive pair
                # For each positive pair of current example
                for pos_idx in positive_pairs:
                    numerator = sim_matrix_exp[i, pos_idx]  # exp(sim(en, tr+))        
                    loss -= torch.log(numerator / denominator)
                    num_pairs += 1
        
        # Average loss over actual pairs
        if num_pairs > 0:
            loss = loss / num_pairs
        
        return loss
    
    # Override the `compute_loss` method to add contrastive loss
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # add output_hidden_states=True when calling the model. Same as: model(**inputs, output_hidden_states=True)
        inputs["output_hidden_states"] = True
        # print(inputs.keys())
        # compute the standard SFT loss
        nll_loss, outputs = super().compute_loss(model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch)
        
        # print the hidden states shape
        # print(len(outputs.hidden_states), "hidden states", outputs.hidden_states[0].shape)

        labels = inputs["labels"]
        first_non_ignore_indices = self.find_first_non_ignore_index(labels)  # a 1D tensor
        contrastive_loss = self.compute_contrastive_loss(outputs, first_non_ignore_indices)

        # log the original value of loss components
        if self.args.logging_steps > 0 and self.state.global_step % self.args.logging_steps == 0:
            self.log({
                "nll_loss": nll_loss.item(),
                "contrastive_loss": contrastive_loss.item(),
                'learning_rate': self.optimizer.param_groups[0]['lr'],
            })
        
        # TODO: how to assign ratio?
        loss = nll_loss + self.args.contrastive_loss_ratio * contrastive_loss

        return (loss, outputs) if return_outputs else loss


# class SavePeftModelCallback(transformers.TrainerCallback):
#     def save_model(self, args, state, kwargs):
#         print('Saving PEFT checkpoint...')
#         if state.best_model_checkpoint is not None:
#             checkpoint_folder = os.path.join(state.best_model_checkpoint, "adapter_model")
#         else:
#             checkpoint_folder = os.path.join(args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}")

#         peft_model_path = os.path.join(checkpoint_folder, "adapter_model")
#         kwargs["model"].save_pretrained(peft_model_path)

#         pytorch_model_path = os.path.join(checkpoint_folder, "pytorch_model.bin")
#         if os.path.exists(pytorch_model_path):
#             os.remove(pytorch_model_path)

#         upload_model_to_hf(checkpoint_folder, f"peft-model-{state.global_step}")

#     def on_save(self, args, state, control, **kwargs):
#         self.save_model(args, state, kwargs)
#         return control

#     def on_train_end(self, args, state, control, **kwargs):
#         def touch(fname, times=None):
#             with open(fname, 'a'):
#                 os.utime(fname, times)

#         touch(os.path.join(args.output_dir, 'completed'))
#         self.save_model(args, state, kwargs)


bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

# # Llama2 special tokens
# DEFAULT_PAD_TOKEN = "[PAD]"
# DEFAULT_EOS_TOKEN = "</s>"
# DEFAULT_BOS_TOKEN = "<s>"
# DEFAULT_UNK_TOKEN = "<unk>"

# Llama3 special tokens
DEFAULT_PAD_TOKEN = "<|finetune_right_pad_id|>"
DEFAULT_EOS_TOKEN = "<|end_of_text|>"
DEFAULT_BOS_TOKEN = "<|begin_of_text|>"
DEFAULT_UNK_TOKEN = "<unk>"

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

def take_subset(dataset, size):
    """Helper function to take a subset of a dataset"""
    if size is None or size >= len(dataset):
        return dataset
    return dataset.select(range(size))

def train(
    # model/data params
    base_model: str = "",  # the only required argument
    english_data_path: str = "yahma/alpaca-cleaned",  # base English dataset
    target_data_paths: List[str] = [],  # paths to target language datasets
    output_dir: str = "./lora-alpaca",
    device_map: str = "auto",
    # training hyperparams
    batch_size: int = 128,
    micro_batch_size: int = 4,
    num_epochs: int = 3,
    learning_rate: float = 3e-4,
    warmup_ratio: float = 0.05,
    cutoff_len: int = 256,
    contrastive_loss_ratio: float = 1.5, 
    # dataset splitting params
    val_size: float = 0.1,
    train_set_size: int = None,  # Set to None to use full training split
    val_set_size: int = None,    # Set to None to use full validation split
    # lora hyperparams
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    lora_target_modules: List[str] = [
        "q_proj",
        "v_proj",
    ],
    # llm hyperparams
    train_on_inputs: bool = True,  # if False, masks out inputs in loss
    add_eos_token: bool = False,
    group_by_length: bool = False,  # faster, but produces an odd training loss curve
    # logging params
    logging_steps: int = 10,
    wandb_project: str = None,
    wandb_run_name: str = None,
    wandb_watch: str = "",  # options: false | gradients | all
    wandb_log_model: str = "",  # options: false | true
    wandb_run_id: str = None,  # options: false | true
    resume_from_checkpoint: str = None,  # either training checkpoint or final adapter
    prompt_template_name: str = "alpaca",  # The prompt template to use, will default to alpaca.
    # experimental
    use_landmark: bool = False,
    use_rope_scaled: bool = False,
):
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print(
            f"Training Alpaca-LoRA model with params:\n"
            f"base_model: {base_model}\n"
            f"english_data_path: {english_data_path}\n"
            f"target_data_paths: {target_data_paths}\n"
            f"output_dir: {output_dir}\n"
            f"batch_size: {batch_size}\n"
            f"micro_batch_size: {micro_batch_size}\n"
            f"num_epochs: {num_epochs}\n"
            f"learning_rate: {learning_rate}\n"
            f"warmup_ratio: {warmup_ratio}\n"
            f"cutoff_len: {cutoff_len}\n"
            f"contrastive_loss_ratio: {contrastive_loss_ratio}\n"
            f"val_size: {val_size}\n"
            f"train_set_size: {train_set_size}\n"
            f"val_set_size: {val_set_size}\n"
            f"lora_r: {lora_r}\n"
            f"lora_alpha: {lora_alpha}\n"
            f"lora_dropout: {lora_dropout}\n"
            f"lora_target_modules: {lora_target_modules}\n"
            f"train_on_inputs: {train_on_inputs}\n"
            f"add_eos_token: {add_eos_token}\n"
            f"group_by_length: {group_by_length}\n"
            f"wandb_project: {wandb_project}\n"
            f"wandb_run_name: {wandb_run_name}\n"
            f"wandb_watch: {wandb_watch}\n"
            f"wandb_log_model: {wandb_log_model}\n"
            f"wandb_run_id: {wandb_run_id}\n"
            f"resume_from_checkpoint: {resume_from_checkpoint or False}\n"
            f"prompt template: {prompt_template_name}\n"
        )
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='huggyllama/llama-7b'"
    gradient_accumulation_steps = batch_size // micro_batch_size

    prompter = Prompter(prompt_template_name)

    #device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size

    # Check if parameter passed or if set within environ
    use_wandb = wandb_project is not None or (
        "WANDB_PROJECT" in os.environ and len(os.environ["WANDB_PROJECT"]) > 0
    )
    # Only overwrite environ if wandb param passed
    if wandb_project is not None:
        os.environ["WANDB_PROJECT"] = wandb_project
    if len(wandb_watch) > 0:
        os.environ["WANDB_WATCH"] = wandb_watch
    if len(wandb_log_model) > 0:
        os.environ["WANDB_LOG_MODEL"] = wandb_log_model

    if "rwkv" in base_model.lower():
        bnb_config.bnb_4bit_use_double_quant = False

    if use_landmark:
        from experiments.landmark import LlamaForCausalLM
        model = LlamaForCausalLM.from_pretrained(
            base_model,
            quantization_config=bnb_config,
            device_map=device_map,
            trust_remote_code=True,
        )

        tokenizer = AutoTokenizer.from_pretrained(base_model,
                                                  model_max_length=3000,
                                                  padding_side="right",
                                                  use_fast=False)

        mem_token = "<landmark>"
        special_tokens_dict = dict()
        if tokenizer.pad_token is None:
            special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
        if tokenizer.eos_token is None:
            special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
        if tokenizer.bos_token is None:
            special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
        if tokenizer.unk_token is None:
            special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN
        special_tokens_dict["additional_special_tokens"] = [mem_token]

        smart_tokenizer_and_embedding_resize(
            special_tokens_dict=special_tokens_dict,
            tokenizer=tokenizer,
            model=model,
        )

        mem_id = tokenizer.convert_tokens_to_ids(mem_token)
        model.set_mem_id(mem_id)
    elif use_rope_scaled:
        from experiments.llama_rope_scaled_monkey_patch import replace_llama_rope_with_scaled_rope
        replace_llama_rope_with_scaled_rope()

        from transformers import  LlamaForCausalLM

        model = LlamaForCausalLM.from_pretrained(
            base_model,
            quantization_config=bnb_config,
            device_map=device_map,
        )

        tokenizer = AutoTokenizer.from_pretrained(base_model)

    else:
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            quantization_config=bnb_config,
            device_map=device_map,
            trust_remote_code=True
        )

        tokenizer = AutoTokenizer.from_pretrained(base_model)
    
    if tokenizer._pad_token is None:
        smart_tokenizer_and_embedding_resize(
            special_tokens_dict=dict(pad_token=DEFAULT_PAD_TOKEN),
            tokenizer=tokenizer,
            model=model,
        )
    if isinstance(tokenizer, LlamaTokenizerFast):
        # LLaMA tokenizer may not have correct special tokens set.
        # Check and add them if missing to prevent them from being parsed into different tokens.
        # Note that these are present in the vocabulary. 
        # Note also that `model.config.pad_token_id` is 0 which corresponds to `<unk>` token.
        tokenizer.eos_token_id = model.config.eos_token_id
        tokenizer.pad_token_id = model.config.pad_token_id
        if hasattr(model.config, 'unk_token_id'):
            tokenizer.unk_token_id = model.config.unk_token_id
        else:
            tokenizer.unk_token_id = tokenizer.pad_token_id
            

    # tokenizer.padding_side = "left"  # Allow batched inference

    def generate_and_tokenize_prompt(data_point):
        full_prompt = prompter.generate_prompt(
            data_point["instruction"],
            data_point["input"],
            data_point["output"],
        )

        full_prompt_tokens = tokenizer.encode(full_prompt, return_tensors="pt")
        output_tokens = tokenizer.encode(data_point["output"], return_tensors="pt")

        response_start_index = full_prompt_tokens.shape[1] - output_tokens.shape[1] 

        return {
            "text": full_prompt, 
            "response_start_index": response_start_index
        }

    if isinstance(model, RwkvForCausalLM):
        use_gradient_checkpointing=False
    else:
        use_gradient_checkpointing=True
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=use_gradient_checkpointing)

    config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=lora_target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)

    # Load datasets
    if english_data_path.endswith(".json") or english_data_path.endswith(".jsonl"):
        english_dataset = load_dataset("json", data_files=english_data_path)["train"]
    else:
        english_dataset = load_dataset(english_data_path)["train"]
    
    target_datasets = []
    for data_path in target_data_paths:
        if data_path.endswith(".json") or data_path.endswith(".jsonl"):
            dataset = load_dataset("json", data_files=data_path)["train"]
        else:
            dataset = load_dataset(data_path)["train"]
        target_datasets.append(dataset)

    # Create parallel datasets
    parallel_datasets = zip_parallel_datasets(
        english_dataset, 
        *target_datasets,
        val_size=val_size
    )

    # Take subsets of each split if sizes are specified
    if train_set_size and train_set_size > 0:
        train_data = take_subset(
            parallel_datasets["train"], 
            train_set_size
        ).map(generate_and_tokenize_prompt, num_proc=os.cpu_count())
    else:
        train_data = parallel_datasets["train"].map(generate_and_tokenize_prompt, num_proc=os.cpu_count())
    
    if val_set_size and val_set_size > 0:
        val_data = take_subset(
            parallel_datasets["test"], 
            val_set_size
        ).map(generate_and_tokenize_prompt, num_proc=os.cpu_count())
    else:
        val_data = parallel_datasets["test"].map(generate_and_tokenize_prompt, num_proc=os.cpu_count())

    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print(
            f"\nDataset sizes after splitting and subsetting:\n"
            f"Training set size: {len(train_data)}\n"
            f"Validation set size: {len(val_data)}\n"
        )

    # if resume_from_checkpoint:
    #     # Check the available weights and load them
    #     checkpoint_name = os.path.join(
    #         resume_from_checkpoint, "pytorch_model.bin"
    #     )  # Full checkpoint
    #     if not os.path.exists(checkpoint_name):
    #         checkpoint_name = os.path.join(
    #             resume_from_checkpoint, "adapter_model/adapter_model.safetensors"
    #         )  # only LoRA model - LoRA config above has to fit
    #         resume_from_checkpoint = (
    #             False  # So the trainer won't try loading its state
    #         )
    #     # The two files above have a different name depending on how they were saved, but are actually the same.
    #     if os.path.exists(checkpoint_name):
    #         print(f"Restarting from {checkpoint_name}")
    #         adapters_weights = load_file(checkpoint_name)
    #         set_peft_model_state_dict(model, adapters_weights)
    #     else:
    #         print(f"Checkpoint {checkpoint_name} not found")

    print_trainable_parameters(model) # Be more transparent about the % of trainable params.

    if not ddp and torch.cuda.device_count() > 1:
        # keeps Trainer from trying its own DataParallelism when more than 1 gpu is available
        model.is_parallelizable = True
        model.model_parallel = True
    
    # data collator
    instruction_template = "### Instruction:\n"
    response_template = "### Response:\n"
    if train_on_inputs:
        collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer, 
            mlm=False,
            pad_to_multiple_of=8, 
            return_tensors="pt",
        )
    else:
        collator = DataCollatorForCompletionOnlyLM(
            instruction_template=instruction_template, 
            response_template=response_template, 
            tokenizer=tokenizer, 
            mlm=False,
            pad_to_multiple_of=8, 
            return_tensors="pt", 
        )

    with wandb.init(project=wandb_project, id=wandb_run_id, name=wandb_run_name, resume="allow") as run:
        trainer = LanguageContrastiveSFTTrainer(
            model=model,
            train_dataset=train_data,
            eval_dataset=val_data,
            args=LanguageContrastiveSFTConfig(
                contrastive_loss_ratio=contrastive_loss_ratio,
                per_device_train_batch_size=micro_batch_size,
                dataloader_drop_last=True,  # to make sure contrastive loss is computed correctly
                gradient_accumulation_steps=gradient_accumulation_steps,
                max_seq_length=cutoff_len,
                warmup_ratio=warmup_ratio,
                num_train_epochs=num_epochs,
                learning_rate=learning_rate,
                bf16=True,
                logging_steps=logging_steps,
                optim="paged_adamw_8bit",
                evaluation_strategy="steps" if val_set_size > 0 else "no",
                save_strategy="steps",
                eval_steps=500 if val_set_size > 0 else None,
                save_steps=500,
                output_dir=output_dir,
                save_total_limit=3,
                #load_best_model_at_end=True if val_set_size > 0 else False,
                load_best_model_at_end=False,
                ddp_find_unused_parameters=False if ddp else None,
                group_by_length=group_by_length,
                report_to="wandb" if use_wandb else None,
                run_name=wandb_run_name if use_wandb else None,
            ),
            data_collator=collator,
        )
        model.config.use_cache = False

    #    old_state_dict = model.state_dict
    #    model.state_dict = (
    #        lambda self, *_, **__: get_peft_model_state_dict(
    #            self, old_state_dict()
    #        )
    #    ).__get__(model, type(model))


        #if torch.__version__ >= "2" and sys.platform != "win32":
        #    model = torch.compile(model)

        trainer.train(resume_from_checkpoint=resume_from_checkpoint)

        model.save_pretrained(output_dir)

        print(
            "\n If there's a warning about missing keys above, please disregard :)"
        )


if __name__ == "__main__":
    # # execute from CLI
    # fire.Fire(train)

    # execute from code
    train(
        # model/data params
        base_model = "meta-llama/Llama-3.1-8B",  # the only required argument
        english_data_path="yahma/alpaca-cleaned",
        target_data_paths=[
            "pinzhenchen/alpaca-cleaned-de",
            "pinzhenchen/alpaca-cleaned-es",
            "pinzhenchen/alpaca-cleaned-fr"
        ],
        # training hyperparams
        batch_size = 64,
        micro_batch_size = 8,
        num_epochs = 2,
        learning_rate = 3e-4,
        cutoff_len = 512,
        contrastive_loss_ratio = 0.2, 
        val_size=0.05,
        val_set_size=1000,
        # lora hyperparams
        lora_r = 8,
        lora_alpha = 16,
        lora_dropout = 0.05,
        lora_target_modules = [
            "q_proj",
            "v_proj",
            "k_proj",
            "o_proj",
        ],
        # llm hyperparams
        train_on_inputs = False,  # if False, masks out inputs in loss
        # logging params
        # resume_from_checkpoint="lora-alpaca/checkpoint-4",
        logging_steps=5,
        wandb_log_model="checkpoint",
        wandb_project="huggingface",
        wandb_run_name="qlora-alpaca-ratio-0.2",
        wandb_run_id=None,
    )

