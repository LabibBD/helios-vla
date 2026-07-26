"""HELIOS Phase 1: LIBERO environment smoke test."""

from pathlib import Path

import numpy as np
from PIL import Image

from lerobot.envs.factory import make_env, make_env_config


def main() -> None:
    print("=" * 60)
    print("HELIOS — LIBERO SIMULATION SMOKE TEST")
    print("=" * 60)

    # Start with a single, simple object-manipulation task.
    cfg = make_env_config(
        "libero",
        task="libero_object",
        task_ids=[0],
        observation_height=256,
        observation_width=256,
        control_mode="relative",
    )

    print("\nCreating LIBERO environment...")

    envs = make_env(
        cfg,
        n_envs=1,
        use_async_envs=False,
    )

    env = envs["libero_object"][0]

    try:
        print("Resetting environment...")
        observation, info = env.reset(seed=42)

        # Access the underlying single environment only to display
        # task metadata during this diagnostic.
        base_env = env.envs[0]

        print("\n===== TASK =====")
        print("Task ID     :", base_env.task_id)
        print("Task name   :", base_env.task)
        print("Instruction :", base_env.task_description)

        print("\n===== SPACES =====")
        print("Action space:", env.single_action_space)
        print("Observation space:")
        print(env.single_observation_space)

        print("\n===== OBSERVATION =====")

        agent_image = observation["pixels"]["image"][0]
        wrist_image = observation["pixels"]["image2"][0]

        print(
            "Agent camera:",
            agent_image.shape,
            agent_image.dtype,
        )
        print(
            "Wrist camera:",
            wrist_image.shape,
            wrist_image.dtype,
        )

        robot_state = observation["robot_state"]

        print("EEF position:",
              robot_state["eef"]["pos"][0])

        print("EEF quaternion:",
              robot_state["eef"]["quat"][0])

        print("Gripper qpos:",
              robot_state["gripper"]["qpos"][0])

        print("Joint positions:",
              robot_state["joints"]["pos"][0])

        # Save the observations exactly as received.
        output_dir = Path("assets")
        output_dir.mkdir(parents=True, exist_ok=True)

        Image.fromarray(agent_image).save(
            output_dir / "libero_agentview.png"
        )

        Image.fromarray(wrist_image).save(
            output_dir / "libero_wrist.png"
        )

        print("\nSaved:")
        print(" assets/libero_agentview.png")
        print(" assets/libero_wrist.png")

        # LIBERO's official dummy action:
        # zero Cartesian motion + inactive gripper command.
        action = np.array(
            [[0, 0, 0, 0, 0, 0, -1]],
            dtype=np.float32,
        )

        print("\n===== CONTROL TEST =====")

        for step in range(5):
            (
                observation,
                reward,
                terminated,
                truncated,
                info,
            ) = env.step(action)

            print(
                f"Step {step + 1}: "
                f"reward={reward[0]:.3f}, "
                f"terminated={bool(terminated[0])}"
            )

        print("\nLIBERO reset       : PASS")
        print("Camera observations: PASS")
        print("Robot state        : PASS")
        print("7D action execution: PASS")
        print()
        print("HELIOS PHASE 1 SMOKE TEST: PASSED")

    finally:
        env.close()


if __name__ == "__main__":
    main()
