# Imports
import torch as t
from nnsight import LanguageModel
import argparse
import itertools
import os
from datetime import datetime

from dictionary_learning.training import trainSAE
from dictionary_learning.trainers.relu import ReLUTrainer
from dictionary_learning.trainers.top_k import TrainerTopK, TopKAutoEncoder
from dictionary_learning.utils import hf_dataset_to_generator
from dictionary_learning.buffer import ActivationBuffer
from dictionary_learning.dictionary import AutoEncoder


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", type=str, required=True, help="where to store sweep")
    parser.add_argument("--no_wandb_logging", action="store_true", help="omit wandb logging")
    parser.add_argument("--dry_run", action="store_true", help="dry run sweep")
    parser.add_argument("--num_tokens", type=int, required=True, help="total number of training tokens")
    parser.add_argument(
        "--layers", type=int, nargs="+", required=True, help="layers to train SAE on"
    )
    parser.add_argument(
        "--width_exponents", type=int, nargs="+", required=True, help="power of 2 for total number of SAE latents"
    )
    parser.add_argument(
        "--architectures", type=str, nargs="+", required=True, help="architecture of the SAE"
    )
    parser.add_argument("--device", type=str, help="device to train on")
    args = parser.parse_args()
    return args


def run_sae_training(
    layer: int,
    width_exponents: list[int],
    num_tokens: int,
    architectures: list[str],
    save_dir: str,
    device: str,
    dry_run: bool = False,
    no_wandb_logging: bool = False,
):
    # model and data parameters
    model_name = "google/gemma-2-2b"
    context_length = 128

    buffer_size = int(2048)
    llm_batch_size = 24  # 32 on a 24GB RTX 3090
    sae_batch_size = 2048  # 2048 on a 24GB RTX 3090

    # sae training parameters
    learning_rates = [3e-4]
    random_seeds = [0]
    dict_sizes = [int(2**i) for i in width_exponents]
    steps = int(num_tokens / sae_batch_size)  # Total number of batches to train
    warmup_steps = 1000  # Warmup period at start of training and after each resample
    resample_steps = None

    # Standard sae training parameters
    sparsity_penalties = [0.025, 0.035, 0.04, 0.05, 0.06, 0.07]

    # TopK sae training parameters
    ks = [20, 40, 80, 160, 320, 640]
    decay_start = 24000
    auxk_alpha = 1 / 32

    # Checkpoints
    desired_checkpoints = t.logspace(-3, 0, 7).tolist()
    desired_checkpoints = [0.0] + desired_checkpoints[:-1]
    desired_checkpoints.sort()

    save_steps = [int(steps * step) for step in desired_checkpoints]
    save_steps.sort()

    # Wandb logging
    # NOTE Running this script with --no_wandb_logging will disable wandb logging
    # NOTE If you want to log to wandb, you need to set the environment variable WANDB_API_KEY
    wandb_entity="canrager"
    wandb_project="checkpoint_sae_sweep"
    log_steps = 100  # Log the training on wandb
    if no_wandb_logging:
        log_steps = None

    # Model
    model = LanguageModel(
        model_name,
        device_map=device,
        low_cpu_mem_usage=True,
        attn_implementation="eager",
        torch_dtype=t.bfloat16,
        cache_dir=None,
    )
    submodule = model.model.layers[layer]
    submodule_name = f"resid_post_layer_{layer}"
    activation_dim = model.config.hidden_size

    # Dataset
    generator = hf_dataset_to_generator("monology/pile-uncopyrighted")

    # Activation buffer
    activation_buffer = ActivationBuffer(
        generator,
        model,
        submodule,
        n_ctxs=buffer_size,
        ctx_len=context_length,
        refresh_batch_size=llm_batch_size,
        out_batch_size=sae_batch_size,
        io="out",
        d_submodule=activation_dim,
        device=device,
    )

    # create the list of configs
    trainer_configs = []
    if "vanilla" in architectures:
        for seed, initial_sparsity_penalty, dict_size, learning_rate in itertools.product(
            random_seeds, sparsity_penalties, dict_sizes, learning_rates
        ):
            trainer_configs.append({
                "trainer": ReLUTrainer,
                "dict_class": AutoEncoder,
                "activation_dim": activation_dim,
                "dict_size": dict_size,
                "lr": learning_rate,
                "l1_penalty": initial_sparsity_penalty,
                "warmup_steps": warmup_steps,
                "resample_steps": resample_steps,
                "seed": seed,
                "wandb_name": f"ReLUTrainer-{model_name}-{submodule_name}",
                "layer": layer,
                "lm_name": model_name,
                "device": device,
                "submodule_name": submodule_name,
            })

    if "topk" in architectures:
        for seed, k, dict_size, learning_rate in itertools.product(
            random_seeds, ks, dict_sizes, learning_rates
        ):
            trainer_configs.append({
                "trainer": TrainerTopK,
                "dict_class": TopKAutoEncoder,
                "activation_dim": activation_dim,
                "dict_size": dict_size,
                "k": k,
                "auxk_alpha": auxk_alpha,  # see Appendix A.2
                "decay_start": decay_start,  # when does the lr decay start
                "steps": steps,  # when when does training end
                "seed": seed,
                "wandb_name": f"TopKTrainer-{model_name}-{submodule_name}",
                "device": device,
                "layer": layer,
                "lm_name": model_name,
                "submodule_name": submodule_name,
            })

    mmdd = datetime.now().strftime('%m%d')
    model_id = model_name.split('/')[1]
    width_str = "_".join([str(i) for i in width_exponents])
    architectures_str = "_".join(architectures)
    save_name = f"{model_id}_{architectures_str}_layer-{layer}_width-2pow{width_str}_date-{mmdd}"
    save_dir = os.path.join(save_dir, save_name)
    print(f"save_dir: {save_dir}")
    print(f"desired_checkpoints: {desired_checkpoints}")
    print(f"save_steps: {save_steps}")
    print(f"num_tokens: {num_tokens}")
    print(f"len trainer configs: {len(trainer_configs)}")
    print(f"trainer_configs: {trainer_configs}")
   

    if not dry_run:
        # actually run the sweep
        trainSAE(
            data=activation_buffer,
            trainer_configs=trainer_configs,
            steps=steps,
            save_steps=save_steps,
            save_dir=save_dir,
            log_steps=log_steps,
            use_wandb=not no_wandb_logging,
            wandb_entity=wandb_entity,
            wandb_project=wandb_project
        )


if __name__ == "__main__":
    args = get_args()
    for layer in args.layers:
        run_sae_training(
            layer=layer,
            save_dir=args.save_dir,
            num_tokens=args.num_tokens,
            width_exponents=args.width_exponents,
            architectures=args.architectures,
            device=args.device,
            dry_run=args.dry_run,
            no_wandb_logging=args.no_wandb_logging,
        )
