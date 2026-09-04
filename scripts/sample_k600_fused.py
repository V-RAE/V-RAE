"""Generate a K600 sample with the released DiT and V-RAE checkpoints.

Example:
    PYTHONPATH=src python scripts/sample_k600_fused.py --steps 10
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from vrae.checkpoint import load_checkpoint
from vrae.config import load_config
from vrae.paths import ProjectPaths
from vrae.training.common.engine import (
    build_class_conditional_dit,
    build_flow_transport,
    generate_class_conditional,
    load_frozen_stage1,
)
from vrae.training.common.latent_norm import LatentNormalizer
from vrae.training.common.visualization import save_video_mp4


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/training/generation/k600_videodit.yaml")
    parser.add_argument("--dit-checkpoint", default="ckpts/k600_dit/k600_dit_vjepa2.1.pt")
    parser.add_argument("--vrae-checkpoint", default="ckpts/vrae/vrae_vjepa2.1.pt")
    parser.add_argument("--latent-stats", default="ckpts/k600_dit/k600_latent_stats_vjepa2.1.pt")
    parser.add_argument("--output-dir", default="test/vis/k600_fused_check")
    parser.add_argument("--class-id", type=int, default=0)
    parser.add_argument("--steps", type=int, default=None)
    args = parser.parse_args()

    root = Path.cwd()
    paths = ProjectPaths(project_root=root)
    config = load_config(args.config)
    sampling = config.get("sampling", {})
    config["stage1"]["checkpoint"] = args.vrae_checkpoint
    config["latent_normalizer"]["path"] = args.latent_stats
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    stage1, _ = load_frozen_stage1(config, paths, device)
    num_chunks = int(config["data"]["num_frames"]) // 4
    dit = build_class_conditional_dit(config, stage1.metadata(), num_chunks=num_chunks).to(device)
    payload = load_checkpoint(paths.checkpoint(args.dit_checkpoint))
    incompatible = dit.load_state_dict(payload["model"], strict=False)
    allowed_missing = {"encoder_view_embedding.weight", "decoder_view_embedding.weight"}
    if set(incompatible.missing_keys) - allowed_missing or incompatible.unexpected_keys:
        raise RuntimeError(
            "Unexpected DiT checkpoint mismatch: "
            f"missing={incompatible.missing_keys}, unexpected={incompatible.unexpected_keys}"
        )
    dit.eval()
    normalizer = LatentNormalizer.load(paths.checkpoint(args.latent_stats)).to(device)
    transport = build_flow_transport(config)

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
        video = generate_class_conditional(
            dit,
            stage1,
            normalizer,
            transport=transport,
            sample_ids=[3407],
            labels=torch.tensor([args.class_id]),
            base_seed=int(sampling.get("base_seed", 3407)),
            num_chunks=dit.num_chunks,
            grid_size=tuple(dit.grid_size),
            channels=dit.in_channels,
            steps=int(args.steps if args.steps is not None else sampling.get("steps", 10)),
            cfg_scale=float(sampling.get("cfg_scale", 1.0)),
            internal_guidance_scale=float(sampling.get("internal_guidance_scale", 1.78)),
            internal_guidance_t_min=float(sampling.get("internal_guidance_t_min", 0.10)),
            internal_guidance_t_max=float(sampling.get("internal_guidance_t_max", 1.0)),
            device=device,
        )

    video = video.float().clamp(0.0, 1.0).cpu()
    output_dir = root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"video": video, "labels": torch.tensor([args.class_id])}, output_dir / "sample.pt")
    save_video_mp4(video[0], output_dir / "sample.mp4", fps=8)
    print(f"shape={tuple(video.shape)} min={video.min().item():.6f} max={video.max().item():.6f} mean={video.mean().item():.6f} std={video.std().item():.6f}")
    print(output_dir / "sample.mp4")


if __name__ == "__main__":
    main()
