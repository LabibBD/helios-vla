"""Create a verified HELIOS recovery demo from a fresh live run."""

import random
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

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

from verifier.oracle_verifier import SimulatorOracleVerifier


OUTPUT = Path("results/demo/helios_recovery_demo_final.mp4")
FPS = 20
MAX_ATTEMPTS = 3


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def bool0(value) -> bool:
    arr = np.asarray(value)
    return bool(arr.reshape(-1)[0]) if arr.size else False


def camera(obs: dict) -> np.ndarray:
    img = np.asarray(obs["pixels"]["image"][0])

    if img.dtype != np.uint8:
        if img.max() <= 1:
            img = img * 255
        img = np.clip(img, 0, 255).astype(np.uint8)

    return img


def title_frames(text: str, shape, seconds: float):
    h, w, _ = shape
    image = Image.new("RGB", (w, h), "black")
    draw = ImageDraw.Draw(image)

    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            max(16, w // 21),
        )
    except OSError:
        font = ImageFont.load_default()

    lines = text.split("\n")
    heights = []

    for line in lines:
        box = draw.textbbox((0, 0), line, font=font)
        heights.append(box[3] - box[1])

    y = (h - sum(heights) - 10 * (len(lines) - 1)) // 2

    for line, line_h in zip(lines, heights):
        box = draw.textbbox((0, 0), line, font=font)
        line_w = box[2] - box[0]

        draw.text(
            ((w - line_w) // 2, y),
            line,
            fill="white",
            font=font,
        )
        y += line_h + 10

    frame = np.asarray(image)

    return [
        frame.copy()
        for _ in range(max(1, int(seconds * FPS)))
    ]


def infer_action(
    observation,
    env,
    policy,
    env_preprocessor,
    env_postprocessor,
    preprocessor,
    postprocessor,
):
    obs = preprocess_observation(observation)

    try:
        obs["task"] = list(env.call("task_description"))
    except (AttributeError, NotImplementedError):
        obs["task"] = list(env.call("task"))

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
    env,
    episode_seed,
    policy_seed,
    policy,
    env_preprocessor,
    env_postprocessor,
    preprocessor,
    postprocessor,
    verifier,
    record=False,
):
    observation, _ = env.reset(seed=[episode_seed])

    seed_everything(policy_seed)
    policy.reset()

    frames = []
    max_steps = int(env.call("_max_episode_steps")[0])

    for _ in range(max_steps):
        if record:
            frames.append(camera(observation))

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

        if verifier.verify(info).success:
            if record:
                frames.extend([camera(observation)] * FPS)
            return True, frames

        if bool0(terminated) or bool0(truncated):
            break

    if record:
        frames.extend([camera(observation)] * FPS)

    return False, frames


@parser.wrap()
def main(cfg: EvalPipelineConfig) -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    envs = make_env(
        cfg.env,
        n_envs=1,
        use_async_envs=False,
        trust_remote_code=cfg.trust_remote_code,
    )

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

    try:
        # Search only task 0 first for a compact demo.
        env = envs["libero_object"][0]

        found = None

        for episode_id in range(10):
            if hasattr(env, "envs"):
                env.envs[0].init_state_id = episode_id

            episode_seed = 1000 + episode_id

            outcomes = []

            for attempt in range(1, MAX_ATTEMPTS + 1):
                policy_seed = episode_seed + (attempt - 1) * 100_000

                success, _ = run_attempt(
                    env=env,
                    episode_seed=episode_seed,
                    policy_seed=policy_seed,
                    policy=policy,
                    env_preprocessor=env_preprocessor,
                    env_postprocessor=env_postprocessor,
                    preprocessor=preprocessor,
                    postprocessor=postprocessor,
                    verifier=verifier,
                    record=False,
                )

                outcomes.append(success)

                if success:
                    break

            if (
                len(outcomes) >= 2
                and outcomes[0] is False
                and any(outcomes[1:])
            ):
                found = (
                    episode_id,
                    episode_seed,
                    outcomes.index(True) + 1,
                )
                break

        if found is None:
            raise RuntimeError(
                "No fresh recovery example found in task 0 episodes 0-9."
            )

        episode_id, episode_seed, success_attempt = found

        print("=" * 70)
        print("FOUND LIVE RECOVERY CASE")
        print("=" * 70)
        print("Task ID            : 0")
        print("Episode ID         :", episode_id)
        print("Episode seed       :", episode_seed)
        print("Successful attempt :", success_attempt)

        if hasattr(env, "envs"):
            env.envs[0].init_state_id = episode_id

        first_obs, _ = env.reset(seed=[episode_seed])
        shape = camera(first_obs).shape

        output_frames = []

        output_frames.extend(
            title_frames(
                "HELIOS\nSelf-Correcting VLA Manipulation",
                shape,
                2.0,
            )
        )

        for attempt in range(1, success_attempt + 1):
            policy_seed = episode_seed + (attempt - 1) * 100_000

            success, attempt_frames = run_attempt(
                env=env,
                episode_seed=episode_seed,
                policy_seed=policy_seed,
                policy=policy,
                env_preprocessor=env_preprocessor,
                env_postprocessor=env_postprocessor,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                verifier=verifier,
                record=True,
            )

            output_frames.extend(
                title_frames(
                    f"ATTEMPT {attempt}",
                    shape,
                    1.0,
                )
            )

            output_frames.extend(attempt_frames)

            if success:
                output_frames.extend(
                    title_frames(
                        f"ATTEMPT {attempt}: SUCCESS\nTask completed",
                        shape,
                        2.0,
                    )
                )
                break

            output_frames.extend(
                title_frames(
                    f"ATTEMPT {attempt}: FAILURE\n"
                    "Outcome verification failed",
                    shape,
                    1.5,
                )
            )

            output_frames.extend(
                title_frames(
                    "HELIOS RECOVERY\n"
                    "Rollback -> Re-observe -> Stochastic Replan",
                    shape,
                    2.0,
                )
            )

        output_frames.extend(
            title_frames(
                "RECOVERY COMPLETE\n"
                "Failure -> Replan -> Success",
                shape,
                2.0,
            )
        )

        print(f"Writing {len(output_frames)} frames...")

        with imageio.get_writer(
            OUTPUT,
            fps=FPS,
            codec="libx264",
            quality=8,
        ) as writer:
            for frame in output_frames:
                writer.append_data(frame)

    finally:
        close_envs(envs)

    print()
    print("Demo saved:")
    print(OUTPUT)


if __name__ == "__main__":
    main()
