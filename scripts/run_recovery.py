"""HELIOS self-correction benchmark.

Protocol:
1. Run SmolVLA for the normal LIBERO episode budget.
2. Check task completion using the simulator-state oracle.
3. If unsuccessful, DO NOT reset the environment.
4. Clear the policy's internal action state.
5. Re-observe the current scene and run a second attempt.
6. Measure first-attempt vs final success.

The simulator oracle is intentionally separated from the experimental
visual verifier.
"""


import csv
import json
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


def bool_at_zero(value: Any) -> bool:
    arr = np.asarray(value)
    return bool(arr.reshape(-1)[0]) if arr.size else False


def infer_action(
    observation: dict,
    env,
    policy,
    env_preprocessor,
    env_postprocessor,
    preprocessor,
    postprocessor,
) -> np.ndarray:
    """Run the same preprocessing/inference path used by lerobot-eval."""

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
    action = transition[ACTION]

    return action.detach().cpu().numpy()


def get_base_libero_env(env):
    """Return LeRobot's underlying single LIBERO environment."""
    if not hasattr(env, "envs") or len(env.envs) != 1:
        raise RuntimeError(
            "HELIOS Recovery V3 requires one synchronous environment."
        )

    base = env.envs[0]

    if base._env is None:
        raise RuntimeError("Underlying LIBERO environment is not initialized.")

    return base


def snapshot_sim_state(env) -> np.ndarray:
    """Capture the exact MuJoCo physics state."""
    base = get_base_libero_env(env)
    sim = base._env.env.sim

    return sim.get_state().flatten().copy()


def batchify_tree(value):
    """Convert one raw env observation into vector-env batch format."""
    if isinstance(value, dict):
        return {
            key: batchify_tree(item)
            for key, item in value.items()
        }

    return np.expand_dims(np.asarray(value), axis=0)


def restore_sim_state(env, state: np.ndarray) -> dict:
    """Restore MuJoCo state and regenerate the current observation."""
    base = get_base_libero_env(env)

    inner = base._env.env
    sim = inner.sim

    sim.set_state_from_flattened(state)
    sim.forward()

    # Reset controller goals so they agree with the restored robot pose.
    for robot in base._env.robots:
        controller = robot.controller
        if hasattr(controller, "reset_goal"):
            controller.reset_goal()

    raw_obs = inner._get_observations()
    observation = base._format_raw_obs(raw_obs)

    return batchify_tree(observation)


def execute_primitive(
    *,
    observation: dict,
    env,
    verifier: SimulatorOracleVerifier,
    actions: list[tuple[np.ndarray, int]],
) -> tuple[dict, bool, int, bool]:
    """Execute a generic recovery primitive."""

    total_steps = 0

    for action, repetitions in actions:
        for _ in range(repetitions):
            (
                observation,
                _,
                terminated,
                truncated,
                info,
            ) = env.step(action)

            total_steps += 1

            if verifier.verify(info).success:
                return observation, True, total_steps, False

            if (
                bool_at_zero(terminated)
                or bool_at_zero(truncated)
            ):
                return observation, False, total_steps, True

    return observation, False, total_steps, False



def run_attempt(
    *,
    observation: dict,
    env,
    policy,
    env_preprocessor,
    env_postprocessor,
    preprocessor,
    postprocessor,
    verifier: SimulatorOracleVerifier,
    max_steps: int,
) -> tuple[dict, bool, int, bool]:
    """Run up to max_steps without resetting the world."""

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

        observation, _, terminated, truncated, info = env.step(action)

        verification = verifier.verify(info)

        if verification.success:
            return observation, True, step, False

        episode_ended = (
            bool_at_zero(terminated)
            or bool_at_zero(truncated)
        )

        if episode_ended:
            # Environment ended for a reason other than task success.
            return observation, False, step, True

    return observation, False, max_steps, False


@parser.wrap()
def main(cfg: EvalPipelineConfig) -> None:
    set_seed(cfg.seed if cfg.seed is not None else 1000)

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("HELIOS — SELF-CORRECTION BENCHMARK")
    print("=" * 72)

    # Force single-environment execution so recovery remains easy to reason about.
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

                normal_budget = int(env.call("_max_episode_steps")[0])
                recovery_budget = normal_budget

                print()
                print("-" * 72)
                print(f"Task: {task_group} / {task_id}")
                print(f"Normal attempt budget : {normal_budget} steps")
                print(f"Recovery budget       : {recovery_budget} steps")
                print("-" * 72)

                for episode_id in range(cfg.eval.n_episodes):
                    # Force the same deterministic LIBERO initial-state index
                    # for each numbered experiment.
                    if hasattr(env, "envs"):
                        env.envs[0].init_state_id = episode_id

                    seed = (
                        (cfg.seed if cfg.seed is not None else 1000)
                        + episode_id
                    )

                    observation, _ = env.reset(seed=[seed])

                    policy.reset()

                    start_time = time.perf_counter()

                    (
                        observation,
                        first_success,
                        first_steps,
                        first_ended,
                    ) = run_attempt(
                        observation=observation,
                        env=env,
                        policy=policy,
                        env_preprocessor=env_preprocessor,
                        env_postprocessor=env_postprocessor,
                        preprocessor=preprocessor,
                        postprocessor=postprocessor,
                        verifier=verifier,
                        max_steps=normal_budget,
                    )

                    retries = 0
                    recovery_success = False
                    recovery_steps = 0

                    if (
                        not first_success
                        and not first_ended
                    ):
                        retries = 1

                        print(
                            f"Task {task_id} | Ep {episode_id}: "
                            "attempt 1 FAILED -> snapshot failure state"
                        )

                        failure_state = snapshot_sim_state(env)

                        # Each branch starts from exactly the same failed
                        # MuJoCo state. Only the generic escape primitive
                        # differs before SmolVLA replans.
                        branch_specs = [
                            (
                                "lift",
                                [
                                    (
                                        np.array(
                                            [[
                                                0.0, 0.0, 0.0,
                                                0.0, 0.0, 0.0,
                                                -1.0,
                                            ]],
                                            dtype=np.float32,
                                        ),
                                        8,
                                    ),
                                    (
                                        np.array(
                                            [[
                                                0.0, 0.0, 0.18,
                                                0.0, 0.0, 0.0,
                                                -1.0,
                                            ]],
                                            dtype=np.float32,
                                        ),
                                        8,
                                    ),
                                ],
                            ),
                            (
                                "left_lift",
                                [
                                    (
                                        np.array(
                                            [[
                                                0.0, 0.0, 0.0,
                                                0.0, 0.0, 0.0,
                                                -1.0,
                                            ]],
                                            dtype=np.float32,
                                        ),
                                        8,
                                    ),
                                    (
                                        np.array(
                                            [[
                                                0.0, 0.16, 0.12,
                                                0.0, 0.0, 0.0,
                                                -1.0,
                                            ]],
                                            dtype=np.float32,
                                        ),
                                        8,
                                    ),
                                ],
                            ),
                            (
                                "right_lift",
                                [
                                    (
                                        np.array(
                                            [[
                                                0.0, 0.0, 0.0,
                                                0.0, 0.0, 0.0,
                                                -1.0,
                                            ]],
                                            dtype=np.float32,
                                        ),
                                        8,
                                    ),
                                    (
                                        np.array(
                                            [[
                                                0.0, -0.16, 0.12,
                                                0.0, 0.0, 0.0,
                                                -1.0,
                                            ]],
                                            dtype=np.float32,
                                        ),
                                        8,
                                    ),
                                ],
                            ),
                        ]

                        total_recovery_steps = 0
                        successful_branch = ""

                        for branch_index, (
                            branch_name,
                            primitive,
                        ) in enumerate(branch_specs, start=1):

                            print(
                                f"Task {task_id} | Ep {episode_id}: "
                                f"trying recovery branch "
                                f"{branch_index} ({branch_name})"
                            )

                            # Every candidate starts from the identical
                            # post-failure world state.
                            observation = restore_sim_state(
                                env,
                                failure_state,
                            )

                            policy.reset()

                            (
                                observation,
                                primitive_success,
                                primitive_steps,
                                primitive_ended,
                            ) = execute_primitive(
                                observation=observation,
                                env=env,
                                verifier=verifier,
                                actions=primitive,
                            )

                            total_recovery_steps += primitive_steps

                            if primitive_success:
                                recovery_success = True
                                successful_branch = branch_name
                                break

                            if primitive_ended:
                                continue

                            # Re-observe the perturbed state and force
                            # SmolVLA to generate a fresh action chunk.
                            policy.reset()

                            (
                                observation,
                                branch_success,
                                branch_steps,
                                _,
                            ) = run_attempt(
                                observation=observation,
                                env=env,
                                policy=policy,
                                env_preprocessor=env_preprocessor,
                                env_postprocessor=env_postprocessor,
                                preprocessor=preprocessor,
                                postprocessor=postprocessor,
                                verifier=verifier,
                                max_steps=140,
                            )

                            total_recovery_steps += branch_steps

                            if branch_success:
                                recovery_success = True
                                successful_branch = branch_name

                                print(
                                    f"Task {task_id} | Ep {episode_id}: "
                                    f"RECOVERED via {branch_name}"
                                )
                                break

                        recovery_steps = total_recovery_steps

                        if not recovery_success:
                            print(
                                f"Task {task_id} | Ep {episode_id}: "
                                "all recovery branches failed"
                            )

                    final_success = first_success or recovery_success

                    elapsed = time.perf_counter() - start_time

                    status = (
                        "FIRST-TRY SUCCESS"
                        if first_success
                        else (
                            "RECOVERED"
                            if recovery_success
                            else "FAILED"
                        )
                    )

                    print(
                        f"Task {task_id} | Ep {episode_id}: "
                        f"{status} | "
                        f"steps={first_steps + recovery_steps} | "
                        f"time={elapsed:.2f}s"
                    )

                    rows.append(
                        {
                            "task_group": task_group,
                            "task_id": task_id,
                            "episode_id": episode_id,
                            "seed": seed,
                            "first_attempt_success": first_success,
                            "final_success": final_success,
                            "recovery_success": recovery_success,
                            "retries": retries,
                            "first_attempt_steps": first_steps,
                            "recovery_steps": recovery_steps,
                            "total_steps": first_steps + recovery_steps,
                            "elapsed_s": round(elapsed, 4),
                            "verifier": "libero_check_success",
                        }
                    )

    finally:
        close_envs(envs)

    total = len(rows)

    first_successes = sum(
        bool(r["first_attempt_success"])
        for r in rows
    )

    final_successes = sum(
        bool(r["final_success"])
        for r in rows
    )

    failed_first = total - first_successes

    recovered = sum(
        bool(r["recovery_success"])
        for r in rows
    )

    first_rate = first_successes / total if total else 0.0
    final_rate = final_successes / total if total else 0.0

    recovery_rate = (
        recovered / failed_first
        if failed_first
        else 0.0
    )

    summary = {
        "episodes": total,
        "first_attempt_successes": first_successes,
        "first_attempt_success_rate": first_rate,
        "final_successes": final_successes,
        "final_success_rate": final_rate,
        "failed_first_attempts": failed_first,
        "recovered_failures": recovered,
        "recovery_rate": recovery_rate,
        "absolute_success_gain": final_rate - first_rate,
        "mean_retries": float(
            np.mean([r["retries"] for r in rows])
        ) if rows else 0.0,
        "mean_total_steps": float(
            np.mean([r["total_steps"] for r in rows])
        ) if rows else 0.0,
        "mean_elapsed_s": float(
            np.mean([r["elapsed_s"] for r in rows])
        ) if rows else 0.0,
        "verifier": "LIBERO simulator-state oracle",
    }

    csv_path = output_dir / "recovery_episodes.csv"
    json_path = output_dir / "recovery_summary.json"

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=rows[0].keys(),
        )
        writer.writeheader()
        writer.writerows(rows)

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print()
    print("=" * 72)
    print("HELIOS RECOVERY RESULTS")
    print("=" * 72)

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
        f"{(final_rate - first_rate) * 100:+.1f} percentage points"
    )

    print(f"Mean retries          : {summary['mean_retries']:.2f}")
    print(f"Mean total steps      : {summary['mean_total_steps']:.1f}")
    print(f"Mean time / episode   : {summary['mean_elapsed_s']:.2f}s")

    print()
    print(f"Saved: {csv_path}")
    print(f"Saved: {json_path}")


if __name__ == "__main__":
    main()
