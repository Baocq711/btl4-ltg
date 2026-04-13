"""Standalone evaluation script for trained models."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import jax

from quoridor.ml.checkpoint import list_checkpoints, load_checkpoint
from quoridor.ml.config import AlphaZeroConfig
from quoridor.ml.evaluator import evaluate_against_baseline
from quoridor.ml.network import create_network, init_params, predict


def _load_model(checkpoint_path: str, config: AlphaZeroConfig):
    net = create_network(config)
    rng_key = jax.random.PRNGKey(0)
    variables = init_params(net, rng_key)
    params, _, step = load_checkpoint(checkpoint_path, variables)
    return net, params, step


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained Quoridor model.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint dir")
    parser.add_argument("--opponent", type=str, default="random",
                        choices=["random", "minimax", "mcts"],
                        help="Baseline opponent")
    parser.add_argument("--games", type=int, default=50, help="Number of evaluation games")
    parser.add_argument("--config", type=str, default=None, help="Path to ml_config.json")
    parser.add_argument("--compare", type=str, default=None,
                        help="Second checkpoint to compare against")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Load config
    config_path = args.config or str(Path(args.checkpoint).parent / "config.json")
    if Path(config_path).exists():
        config = AlphaZeroConfig.load(config_path)
    else:
        config = AlphaZeroConfig()

    print(f"Evaluating checkpoint: {args.checkpoint}")
    print(f"Opponent: {args.opponent}, Games: {args.games}")

    net, variables, step = _load_model(args.checkpoint, config)
    print(f"Model loaded from step {step}")

    def predict_fn(variables, x):
        return predict(net, variables, x)

    # Evaluate
    results = evaluate_against_baseline(
        predict_fn, variables, config,
        opponent_kind=args.opponent,
        num_games=args.games,
        seed=args.seed,
    )

    print(f"\n{'='*40}")
    print(f"Results vs {args.opponent} ({args.games} games):")
    print(f"  Win rate:  {results['win_rate']:.1%}")
    print(f"  Wins:      {results['wins']}")
    print(f"  Losses:    {results['losses']}")
    print(f"  Draws:     {results['draws']}")
    print(f"{'='*40}")

    # Compare two checkpoints
    if args.compare:
        from quoridor.ml.evaluator import evaluate_checkpoints

        _, old_variables, old_step = _load_model(args.compare, config)
        print(f"\nComparing step {step} vs step {old_step}:")

        cmp_results = evaluate_checkpoints(
            predict_fn, variables, old_variables, config,
            num_games=args.games, seed=args.seed,
        )
        print(f"  New win rate: {cmp_results['new_win_rate']:.1%}")
        print(f"  New wins: {cmp_results['new_wins']}, Old wins: {cmp_results['old_wins']}, Draws: {cmp_results['draws']}")
        print(f"  Is better: {cmp_results['is_better']}")

    # Output as JSON
    output = {
        "checkpoint": args.checkpoint,
        "step": step,
        "opponent": args.opponent,
        **results,
    }
    print(f"\nJSON: {json.dumps(output, indent=2)}")


if __name__ == "__main__":
    main()
