"""AlphaZero training entry point for Quoridor."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import jax
import numpy as np

from quoridor.ml.checkpoint import load_latest, save_checkpoint
from quoridor.ml.config import AlphaZeroConfig
from quoridor.ml.evaluator import evaluate_against_baseline
from quoridor.ml.network import create_network, init_params, predict
from quoridor.ml.self_play import generate_self_play_data
from quoridor.ml.trainer import Trainer


def main() -> None:
    parser = argparse.ArgumentParser(description="Train AlphaZero agent for Quoridor.")
    parser.add_argument("--iterations", type=int, default=50, help="Number of training iterations")
    parser.add_argument("--config", type=str, default=None, help="Path to ml_config.json")
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint")
    parser.add_argument("--num-games", type=int, default=None, help="Override games per iteration")
    parser.add_argument("--num-simulations", type=int, default=None, help="Override MCTS sims")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Load config
    if args.config:
        config = AlphaZeroConfig.load(args.config)
    else:
        config = AlphaZeroConfig()

    if args.num_games is not None:
        config.num_games_per_iter = args.num_games
    if args.num_simulations is not None:
        config.num_simulations = args.num_simulations

    print(f"Training config: {config}")
    print(f"JAX devices: {jax.devices()}")

    # Initialize trainer
    trainer = Trainer(config, rng_seed=args.seed)
    net = trainer.net

    # Resume from checkpoint if requested
    start_iter = 0
    if args.resume:
        result = load_latest(
            config.checkpoint_dir,
            trainer.get_variables(),
            trainer.opt_state,
        )
        if result is not None:
            variables, opt_state, step = result
            trainer.params = variables["params"]
            trainer.batch_stats = variables.get("batch_stats", trainer.batch_stats)
            if opt_state is not None:
                trainer.opt_state = opt_state
            start_iter = step // config.num_games_per_iter
            trainer.global_step = step
            print(f"Resumed from step {step} (iteration ~{start_iter})")
        else:
            print("No checkpoint found, starting fresh.")

    # TensorBoard writer
    writer = None
    try:
        from torch.utils.tensorboard import SummaryWriter
        Path(config.tensorboard_dir).mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(config.tensorboard_dir)
        print(f"TensorBoard logging to {config.tensorboard_dir}/")
    except ImportError:
        print("TensorBoard not available, skipping logging.")

    # Build predict function for self-play
    def predict_fn(variables, x):
        return predict(net, variables, x)

    best_variables = trainer.get_variables()

    # Save config alongside checkpoints
    Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    config.save(Path(config.checkpoint_dir) / "config.json")

    for iteration in range(start_iter, start_iter + args.iterations):
        iter_start = time.perf_counter()
        print(f"\n{'='*60}")
        print(f"Iteration {iteration + 1}/{start_iter + args.iterations}")
        print(f"{'='*60}")

        # --- Self-play ---
        print(f"  Self-play: generating {config.num_games_per_iter} games...")
        sp_start = time.perf_counter()
        examples = generate_self_play_data(
            predict_fn, trainer.get_variables(), config, seed=args.seed + iteration
        )
        sp_time = time.perf_counter() - sp_start
        print(f"  Self-play done: {len(examples)} examples in {sp_time:.1f}s")

        # --- Add to replay buffer ---
        trainer.buffer.add(examples)
        print(f"  Replay buffer size: {len(trainer.buffer)}")

        # --- Training ---
        print(f"  Training {config.epochs_per_iter} epochs...")
        train_start = time.perf_counter()
        for epoch in range(config.epochs_per_iter):
            losses = trainer.train_epoch(writer=writer)
        train_time = time.perf_counter() - train_start
        print(f"  Training done in {train_time:.1f}s")
        print(f"  Losses: total={losses['total_loss']:.4f} policy={losses['policy_loss']:.4f} value={losses['value_loss']:.4f}")

        # --- Checkpoint & Evaluation ---
        if (iteration + 1) % config.checkpoint_interval == 0:
            step = trainer.global_step
            ckpt_path = save_checkpoint(
                trainer.get_variables(), trainer.opt_state, step, config.checkpoint_dir
            )
            print(f"  Checkpoint saved: {ckpt_path}")

            # Evaluate against random
            print(f"  Evaluating vs random ({config.eval_games} games)...")
            eval_start = time.perf_counter()
            results = evaluate_against_baseline(
                predict_fn, trainer.get_variables(), config,
                opponent_kind="random", seed=args.seed + iteration + 1000
            )
            eval_time = time.perf_counter() - eval_start
            print(f"  vs Random: win={results['win_rate']:.1%} "
                  f"(W:{results['wins']} L:{results['losses']} D:{results['draws']}) "
                  f"in {eval_time:.1f}s")

            if writer is not None:
                writer.add_scalar("eval/win_rate_vs_random", results["win_rate"], step)

            # Update best model
            best_variables = trainer.get_variables()

        iter_time = time.perf_counter() - iter_start
        print(f"  Iteration time: {iter_time:.1f}s")

    # --- Final evaluation ---
    print(f"\n{'='*60}")
    print("Final Evaluation")
    print(f"{'='*60}")

    for opponent in ["random", "minimax"]:
        try:
            results = evaluate_against_baseline(
                predict_fn, best_variables, config,
                opponent_kind=opponent, num_games=config.eval_games,
                seed=args.seed + 9999,
            )
            print(f"  vs {opponent:>8}: win={results['win_rate']:.1%} "
                  f"(W:{results['wins']} L:{results['losses']} D:{results['draws']})")
        except Exception as e:
            print(f"  vs {opponent:>8}: ERROR - {e}")

    # Save final checkpoint
    final_path = save_checkpoint(
        best_variables, trainer.opt_state, trainer.global_step, config.checkpoint_dir
    )
    print(f"\nFinal checkpoint: {final_path}")

    if writer is not None:
        writer.close()

    print("Training complete.")


if __name__ == "__main__":
    main()
