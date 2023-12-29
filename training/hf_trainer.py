import os
import json
import sys
import pathlib
import typing as t
from dataclasses import dataclass, field

import torch
import transformers
import wandb
from accelerate import Accelerator
from peft import PeftModel
from dataset import DataCollatorForMmapedDataset, MmappedArrowDataset
from profiling import ProfilerCallback, build_profiler_configuration

import logging
from transformers import logging as hf_logging

# Set up Python logging to file
logger = hf_logging.get_logger()
log_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%d-%d %H:%M:%S')
for handler in logger.handlers:
    handler.setFormatter(log_format)
logger.setLevel(logging.INFO)


class DualOutput:
    """
    Usage
    if __name__ == "__main__":
        sys.stdout = DualOutput("output.txt")

        # Your script here. For example:
        print("Hello, world!")
        # Rest of your code.
    """
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w")
        self.logger = logger
        handler = logging.StreamHandler(self)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%d-%d %H:%M:%S')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):  # Needed for Python 3 compatibility.
        self.terminal.flush()
        self.log.flush()


@dataclass
class ModelArguments:
    low_cpu_mem_usage: bool = field(
        metadata={"help": "Try to reduce CPU memory usage while loading the model."},
        default=True)
    model_name_or_path: t.Optional[str] = field(
        default="EleutherAI/pythia-70m-deduped")
    use_xformers: bool = field(default=False, metadata={"help": "Use xFormers' memory_efficient_attention"})


@dataclass
class DataArguments:
    train_file: str = field(metadata={"help": "Path to the training set."})
    eval_file: str = field(metadata={"help": "Path to the evaluation set."})


@dataclass
class OtherArguments:
    model_load_delay_per_rank: t.Optional[int] = field(metadata={
        "help": "Delay loading the model by (this many seconds) * (local_rank)."},
        default=None)
    enable_profiler: bool = field(
        metadata={"help": "Whether to profile the training loop."},
        default=False)
    add_special_tokens: t.Optional[str] = field(
        metadata={"help": "Extra special tokens to add to the tokenizer before training. Comma-separated."},
        default=None)
    uft: bool = field(
        metadata={"help": "Use unsupervised fine-tuning instead of supervised fine-tuning."},
        default=False)


@dataclass
class LoraArguments:
    use_lora: t.Optional[bool] = field(metadata={"help": "Whether to train a LoRA instead of the full model."},
                                       default=False)
    lora_rank: t.Optional[int] = field(metadata={"help": "LoRA rank."},
                                       default=4)
    lora_alpha: t.Optional[int] = field(metadata={"help": "LoRA alpha."},
                                        default=32)
    lora_dropout: t.Optional[float] = field(metadata={"help": "LoRA dropout."},
                                            default=0.05)
    lora_target_modules: t.Optional[str] = field(metadata={"help": "Target modules, comma-separated."},
                                                 default=None)


def save_wandb_run_logs(output_path) -> None:
    api = wandb.Api()
    run_path = f"{os.getenv('WANDB_PROJECT')}/{os.getenv('WANDB_RUN_ID')}"
    run = api.run(run_path)

    download_path = run.file('output.log').download()
    os.rename(download_path.name, f'{output_path}/train_progress.log')

    download_path = run.file('wandb-summary.json').download()
    os.rename(download_path.name, f'{output_path}/train_summary.json')

    download_path = run.file('wandb-metadata.json').download()
    os.rename(download_path.name, f'{output_path}/train_metadata.json')

    print(f"Logs from {run_path} saved to {output_path}.")


def main() -> None:
    accelerator = Accelerator()

    parser = transformers.HfArgumentParser((
        ModelArguments,
        DataArguments,
        LoraArguments,
        OtherArguments,
        transformers.TrainingArguments,
    ))
    model_args, data_args, lora_args, \
        other_args, training_args = parser.parse_args_into_dataclasses()

    log_path = "logs/train-uft/" if other_args.uft else "logs/train-sft/"
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    if accelerator.is_main_process:
        sys.stdout = DualOutput(f"{log_path}terminal_output.log")

    logger.info('Load tokenizer')
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        padding_side="right",
        use_fast=True,
    )

    # xFormers optimizations.
    if model_args.use_xformers:
        from monkeypatches import apply_xformers_monkeypatches
        apply_xformers_monkeypatches()

    if other_args.model_load_delay_per_rank is not None:
        # When working with constrained system memory, loading the model at the
        # exact same time on all training processes will likely fail due to all
        # the model copies going around. We can delay loading based on
        # local_rank so not all processes are doing this at once, which
        # alleviates the situation. Kinda silly, but it works.
        import time
        time.sleep(other_args.model_load_delay_per_rank *
                   training_args.local_rank)

    # Model loading. => local에서 받을 수 있는지
    model_load_dtype = None
    if training_args.bf16:
        model_load_dtype = torch.bfloat16
    elif training_args.fp16:
        model_load_dtype = torch.float16

    logger.info('Load model')
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        low_cpu_mem_usage=model_args.low_cpu_mem_usage,
        torch_dtype=model_load_dtype,
    ).cuda()

    if other_args.add_special_tokens is not None:
        # MAINTENANCE(11b): Big fat warning: the snippet below is copy-pasted
        # into ``./preparation/tokenize_data_{sft,uft}.py``. Make sure to always keep both
        # implementations in sync.
        special_token_contents = other_args.add_special_tokens.split(",")
        special_tokens = [
            transformers.AddedToken(
                # Heads up: this is very poorly documented in HuggingFace and
                # some old forum discussions mention that it's apparently
                # exclusive to the Rust-based tokenizers? If anything seems
                # funky about the special token behavior, this is a good place
                # to look.
                content, lstrip=True, rstrip=True)
            for content in special_token_contents
        ]

        _add_special_tokens_to_tokenizer_and_resize_model_embeddings(
            {"additional_special_tokens": special_tokens},
            tokenizer,
            model,
        )

    # LoRA setup.
    if lora_args.use_lora:
        from peft import LoraConfig, TaskType, get_peft_model

        target_modules = None
        if lora_args.lora_target_modules is not None:
            target_modules = lora_args.lora_target_modules.split(",")

        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=lora_args.lora_rank,
            lora_alpha=lora_args.lora_alpha,
            lora_dropout=lora_args.lora_dropout,
            target_modules=target_modules,
        )
        if training_args.gradient_checkpointing:
            model.enable_input_require_grads()

        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    # Silence this annoying warning.
    if training_args.gradient_checkpointing:
        model.config.use_cache = False

    # Dataset setup.
    logger.info('*** Load Training data ***')
    logger.info(f'Train file: {data_args.train_file}')
    train_dataset = MmappedArrowDataset(data_args.train_file, sft=not other_args.uft)
    logger.info(f'Train size: {len(train_dataset)} data item')
    
    logger.info('*** Load Eval data ***')
    logger.info(f'Eval file: {data_args.eval_file}')
    eval_dataset = MmappedArrowDataset(data_args.eval_file, sft=not other_args.uft)
    logger.info(f'Eval size: {len(eval_dataset)} data item')

    data_collator = DataCollatorForMmapedDataset(tokenizer=tokenizer, sft=not other_args.uft)

    trainer = transformers.Trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        args=training_args,
        callbacks=[SavePeftModelCallback] if lora_args.use_lora else None,
    )

    try:
        # Resume from checkpoint if we have any checkpoints automatically saved
        # by the HF Trainer within the output directory.
        resume_from_checkpoint = len(
            list(pathlib.Path(
                training_args.output_dir).glob("checkpoint-*"))) > 0

        if other_args.enable_profiler:
            profiler_args = build_profiler_configuration()
            with torch.profiler.profile(**profiler_args) as profiler:
                trainer.add_callback(ProfilerCallback(profiler=profiler))
                trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        else:
            trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    except KeyboardInterrupt as ex:
        # TODO(11b): Test whether this does what I expect. Idea is to have the
        # trainer save the current state when I interrupt the run so I don't
        # need to keep waiting for a checkpoint step.
        # trainer.save_model()
        # trainer.save_state()
        raise ex

    trainer.save_state()
    trainer.save_model()

    logger.info('Training finished.')

    if accelerator.is_main_process:
        os.environ["WANDB_RUN_ID"] = wandb.run.id
        wandb.finish()
        
        # Save log
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        save_wandb_run_logs(log_path)
    
    if lora_args.use_lora:
        logger.info('** Merge LORA weights to base model **')
        logger.info('Load model...')
        base_model = transformers.AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path)
        model = PeftModel.from_pretrained(base_model, training_args.output_dir)
        merged_model = model.merge_and_unload()
        
        logger.info('Load tokenizer...')
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_args.model_name_or_path)

        logger.info("Saving merged model & tokenizer...")
        merged_model.save_pretrained(
            f"{training_args.output_dir}/merged",
            push_to_hub=False
        )
        tokenizer.save_pretrained(
            f"{training_args.output_dir}/merged",
            push_to_hub=False
        )
        logger.info(f'Model saved to "{training_args.output_dir}/merged". Please use this directory to access the trained model.')


class SavePeftModelCallback(transformers.TrainerCallback):
    '''
    At some point, PEFT stopped saving just the adapter and instead started
    storing full model weights. Extracting the adapter from the weights is
    doable, but seems to result in subpar results for some unknown reason, so
    this Trainer callback saves the adapter itself during training to avoid
    this.

    https://github.com/huggingface/peft/issues/286#issuecomment-1512611968
    https://github.com/huggingface/peft/blob/main/examples/int8_training/peft_bnb_whisper_large_v2_training.ipynb
    '''

    def on_save(
        self,
        args: transformers.TrainingArguments,
        state: transformers.TrainerState,
        control: transformers.TrainerControl,
        **kwargs,
    ):
        checkpoint_folder_name = f"{transformers.trainer_utils.PREFIX_CHECKPOINT_DIR}-{state.global_step}"
        checkpoint_folder = os.path.join(args.output_dir, checkpoint_folder_name)

        peft_model_path = os.path.join(checkpoint_folder, "adapter_model")
        kwargs["model"].save_pretrained(peft_model_path)

        # pytorch_model_path = os.path.join(checkpoint_folder, "pytorch_model.bin")
        # if os.path.exists(pytorch_model_path):
        #     os.remove(pytorch_model_path)

        return control


def _add_special_tokens_to_tokenizer_and_resize_model_embeddings(
    special_tokens: t.Dict[str, t.Union[str, transformers.AddedToken]],
    tokenizer: transformers.PreTrainedTokenizerBase,
    model: transformers.PreTrainedModel,
):
    tokenizer.add_special_tokens(special_tokens)

    # Size is rounded up to the nearest number divisible by 64 for performance
    # reasons.
    new_size = _nearest_divisible(num=len(tokenizer), divisor=64)
    old_size = model.config.vocab_size

    if new_size == old_size:
        # No resizing needs to be done, let's bail!
        return

    # Need to resize the token embeddings. We initialize the new positions with
    # the mean of the existing ones to cut down on required training time.
    model.resize_token_embeddings(new_size)
    new_positions_count = new_size - old_size

    input_embeddings = model.get_input_embeddings().weight.data
    output_embeddings = model.get_output_embeddings().weight.data

    # This is just to keep the LSP happy.
    assert isinstance(input_embeddings, torch.Tensor)
    assert isinstance(output_embeddings, torch.Tensor)

    input_embeddings_avg = input_embeddings[:-new_positions_count].mean(dim=0,
                                                             keepdim=True)
    output_embeddings_avg = output_embeddings[:-new_positions_count].mean(dim=0,
                                                               keepdim=True)

    input_embeddings[-new_positions_count:] = input_embeddings_avg
    output_embeddings[-new_positions_count:] = output_embeddings_avg


def _nearest_divisible(num: int, divisor: int) -> int:
    '''Returns the nearest number to `num` that is divisible by `divisor`.'''
    return (num + divisor - 1) // divisor * divisor

if __name__ == "__main__":
    main()
