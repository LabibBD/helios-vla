"""HELIOS V4 — controlled stochastic checkpoint recovery.

Recovery strategy:
1. Run SmolVLA on a deterministic LIBERO initial state.
2. Verify success using LIBERO's simulator-state oracle.
3. If unsuccessful, reset to EXACTLY the same initial scenario.
4. Reseed SmolVLA with a different, reproducible inference seed.
5. Retry up to MAX_ATTEMPTS.

This tests whether controlled stochastic replanning improves final
task success over one-shot VLA execution.
"""

import csv
import json
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from lerobot.configs import parser
from lerobot.configs.eval import EvalPipelineConfig
from lerobot.envs import (
    close_envs,
    make_env,
    make_env_pre_post_processors,
    preprocess_observation,
)
from lerobot.policies import make_policy, make_pre_post_processors
from lerobot.utils.constants import ACTION
from lerobot.utils.random_utils import set_seed

from verifier.oracle_verifier import SimulatorOracleVerifier


MAX_ATTEMPTS = 3


def seed_everything(seed: int) -> None:
    """Make one policy attempt independently reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def bool_at_zero(value: Any) -> bool:
    array = np.asarray(value)
    return bool(array.reshape(-1)[0]) if array.size else False


def infer_action(
    observation: dict,
    env,
    policy,
    env_preprocessor,
    env_postprocessor,
    preprocessor,
    postprocessor,
) -> np.ndarray:

    obs = preprocess_observation(observation)

    try:
        obs["task"] = list(env.call("task_description"))
    except (AttributeError, NotImplementedError):
        try:
            obs["task"] = list(env.call("task"))
        except (AttributeError, NotImplementedError):
            obs["task"] = [""] * env.num_envs

    obs = env_preprocessor(obs)
    obs = preprocessor(obs)

    with torch.inference_mode():
        action = policy.select_action(obs)

    action = postprocessor(action)

    transition = {ACTION: action}
    transition = env_postprocessor(transition)

    return transition[ACTION].detach().cpu().numpy()


def run_attempt(
    *,
    observation,
    env,
    policy,
    env_preprocessor,
    env_postprocessor,
    preprocessor,
    postprocessor,
    verifier,
    max_steps: int,
) -> tuple[bool, int]:

    for step in range(1, max_steps + 1):

        action = infer_action(
            observation,
            env,
            policy,
            env_preprocessor,
            env_postprocessor,
            preprocessor,
            postprocessor,
        )

        (
            observation,
            _,
            terminated,
            truncated,
            info,
        ) = env.step(action)

        if verifier.verify(info).success:
            return True, step

        if (
            bool_at_zero(terminated)
            or bool_at_zero(truncated)
        ):
            return False, step

    return False, max_steps


@parser.wrap()
def main(cfg: EvalPipelineConfig) -> None:

    base_seed = cfg.seed if cfg.seed is not None else 1000
    set_seed(base_seed)

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 76)
    print("HELIOS V4 — STOCHASTIC CHECKPOINT RECOVERY")
    print("=" * 76)

    envs = make_env(
        cfg.env,
        n_envs=1,
        use_async_envs=False,
        trust_remote_code=cfg.trust_remote_code,
    )

    print("Loading SmolVLA...")

    policy = make_policy(
        cfg=cfg.policy,
        env_cfg=cfg.env,
        rename_map=cfg.rename_map,
    )
    policy.eval()

    preprocessor_overrides = {
        "device_processor": {
            "device": str(policy.config.device),
        },
        "rename_observations_processor": {
            "rename_map": cfg.rename_map,
        },
    }

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg.policy,
        pretrained_path=cfg.policy.pretrained_path,
        preprocessor_overrides=preprocessor_overrides,
    )

    env_preprocessor, env_postprocessor = make_env_pre_post_processors(
        env_cfg=cfg.env,
        policy_cfg=cfg.policy,
    )

    verifier = SimulatorOracleVerifier()

    rows: list[dict[str, Any]] = []

    try:
        for task_group, task_envs in envs.items():
            for task_id, env in task_envs.items():

                max_steps = int(env.call("_max_episode_steps")[0])

                print()
                print("-" * 76)
                print(f"Task             : {task_group} / {task_id}")
                print(f"Attempt budget   : {max_steps} steps")
                print(f"Maximum attempts : {MAX_ATTEMPTS}")
                print("-" * 76)

                for episode_id in range(cfg.eval.n_episodes):

                    if hasattr(env, "envs"):
                        env.envs[0].init_state_id = episode_id

                    episode_seed = (
                        base_seed
                        + int(task_id) * 1000
                        + episode_id
                    )

                    start_time = time.perf_counter()

                    attempt_successes: list[bool] = []
                    attempt_steps: list[int] = []

                    final_success = False
                    successful_attempt = 0

                    for attempt in range(1, MAX_ATTEMPTS + 1):

                        # Reset to the identical LIBERO starting state.
                        observation, _ = env.reset(
                            seed=[episode_seed]
                        )

                        # Different but fully reproducible policy sample.
                        inference_seed = (
                            episode_seed
                            + (attempt - 1) * 100_000
                        )

                        seed_everything(inference_seed)
                        policy.reset()

                        print(
                            f"Task {task_id} | Ep {episode_id} | "
                            f"attempt {attempt}/{MAX_ATTEMPTS} | "
                            f"policy_seed={inference_seed}"
                        )

                        success, steps = run_attempt(
                            observation=observation,
                            env=env,
                            policy=policy,
                            env_preprocessor=env_preprocessor,
                            env_postprocessor=env_postprocessor,
                            preprocessor=preprocessor,
                            postprocessor=postprocessor,
                            verifier=verifier,
                            max_steps=max_steps,
                        )

                        attempt_successes.append(success)
                        attempt_steps.append(steps)

                        if success:
                            final_success = True
                            successful_attempt = attempt
                            break

                        if attempt < MAX_ATTEMPTS:
                            print(
                                f"Task {task_id} | Ep {episode_id}: "
                                "FAILURE -> rollback + stochastic replan"
                            )

                    elapsed = time.perf_counter() - start_time

                    first_success = attempt_successes[0]

                    recovered = (
                        final_success
                        and not first_success
                    )

                    if first_success:
                        status = "FIRST-TRY SUCCESS"
                    elif recovered:
                        status = (
                            f"RECOVERED ON ATTEMPT "
                            f"{successful_attempt}"
                        )
                    else:
                        status = "FAILED"

                    print(
                        f"Task {task_id} | Ep {episode_id}: "
                        f"{status} | "
                        f"attempts={len(attempt_successes)} | "
                        f"time={elapsed:.2f}s"
                    )

                    rows.append(
                        {
                            "task_group": task_group,
                            "task_id": task_id,
                            "episode_id": episode_id,
                            "episode_seed": episode_seed,
                            "first_attempt_success": first_success,
                            "final_success": final_success,
                            "recovered": recovered,
                            "successful_attempt": successful_attempt,
                            "attempts_used": len(attempt_successes),
                            "attempt_successes": json.dumps(
                                attempt_successes
                            ),
                            "attempt_steps": json.dumps(
                                attempt_steps
                            ),
                            "total_steps": sum(attempt_steps),
                            "elapsed_s": round(elapsed, 4),
                            "recovery_type":
                                "stochastic_checkpoint_retry",
                            "verifier":
                                "LIBERO simulator-state oracle",
                        }
                    )

    finally:
        close_envs(envs)

    total = len(rows)

    first_successes = sum(
        bool(row["first_attempt_success"])
        for row in rows
    )

    final_successes = sum(
        bool(row["final_success"])
        for row in rows
    )

    failed_first = total - first_successes

    recovered = sum(
        bool(row["recovered"])
        for row in rows
    )

    first_rate = (
        first_successes / total
        if total else 0.0
    )

    final_rate = (
        final_successes / total
        if total else 0.0
    )

    recovery_rate = (
        recovered / failed_first
        if failed_first else 0.0
    )

    summary = {
        "episodes": total,
        "max_attempts": MAX_ATTEMPTS,
        "first_attempt_successes": first_successes,
        "first_attempt_success_rate": first_rate,
        "final_successes": final_successes,
        "final_success_rate": final_rate,
        "failed_first_attempts": failed_first,
        "recovered_failures": recovered,
        "recovery_rate": recovery_rate,
        "absolute_success_gain": final_rate - first_rate,
        "mean_attempts": float(
            np.mean(
                [row["attempts_used"] for row in rows]
            )
        ),
        "mean_total_steps": float(
            np.mean(
                [row["total_steps"] for row in rows]
            )
        ),
        "mean_elapsed_s": float(
            np.mean(
                [row["elapsed_s"] for row in rows]
            )
        ),
        "recovery_type": "stochastic_checkpoint_retry",
        "verifier": "LIBERO simulator-state oracle",
    }

    csv_path = output_dir / "checkpoint_recovery_episodes.csv"
    json_path = output_dir / "checkpoint_recovery_summary.json"

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=rows[0].keys(),
        )
        writer.writeheader()
        writer.writerows(rows)

    with json_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(summary, file, indent=2)

    print()
    print("=" * 76)
    print("HELIOS V4 RESULTS")
    print("=" * 76)

    print(
        f"First-attempt success : "
        f"{first_successes}/{total} "
        f"({first_rate * 100:.1f}%)"
    )

    print(
        f"Final success         : "
        f"{final_successes}/{total} "
        f"({final_rate * 100:.1f}%)"
    )

    print(
        f"Recovered failures    : "
        f"{recovered}/{failed_first}"
    )

    print(
        f"Recovery rate         : "
        f"{recovery_rate * 100:.1f}%"
    )

    print(
        f"Absolute success gain : "
        f"{(final_rate - first_rate) * 100:+.1f} "
        "percentage points"
    )

    print(
        f"Mean attempts         : "
        f"{summary['mean_attempts']:.2f}"
    )

    print(
        f"Mean time / episode   : "
        f"{summary['mean_elapsed_s']:.2f}s"
    )

    print()
    print(f"Saved: {csv_path}")
    print(f"Saved: {json_path}")


if __name__ == "__main__":
    main()
