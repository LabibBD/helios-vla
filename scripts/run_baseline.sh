#!/usr/bin/env bash
set -e

export MUJOCO_GL=egl

lerobot-eval \
  --policy.path=lerobot/smolvla_libero \
  --policy.device=cuda \
  --policy.empty_cameras=1 \
  --env.type=libero \
  --env.task=libero_object \
  --env.task_ids='[0,1,2]' \
  --env.control_mode=relative \
  --env.max_parallel_tasks=1 \
  --eval.batch_size=1 \
  --eval.n_episodes=5 \
  --eval.use_async_envs=false \
  --output_dir=results/baseline_object \
  --rename_map='{"observation.images.image":"observation.images.camera1","observation.images.image2":"observation.images.camera2"}'
