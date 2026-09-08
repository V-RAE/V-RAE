import json
from types import SimpleNamespace

import torch
from PIL import Image

from vrae.config import load_config
from vrae.data.lerobot import build_temporal_indices
from vrae.training.fastwam.train import (
    _load_checkpoint,
    _load_weights,
    _loss_values_for_logging,
    _prepare_trainable_parameters,
    _resolve_resume_path,
    _save_checkpoint,
    _target_steps,
    validate_build,
)
from vrae.training.fastwam.profiler import StepProfiler, instrument_fastwam
from vrae.training.fastwam.sampling import (
    _pose_trajectory,
    fixed_sample_indices,
    run_sampling,
)


def test_ci2va_fastwam_config_is_buildable() -> None:
    config = load_config("configs/training/generation/libero_ci2va_mot.yaml")
    result = validate_build(config)
    assert result["task"] == "ci2va"
    assert result["frame_num"] == 21
    assert result["action_horizon"] == 20
    assert result["model_factory"] == "fastwam.runtime:create_fastwam_joint"


def test_dual_video_loss_config_is_buildable() -> None:
    config = load_config(
        "configs/training/generation/libero_ci2va_mot_dual_video_loss.yaml"
    )
    result = validate_build(config)
    assert result["base_model_depth"] == 8
    assert result["base_model_coeff"] == 1.0
    assert result["checkpoint_mode"] == "init_from"
    assert config["training"]["resume"] is None
    assert config["training"]["init_from"].endswith("latest.json")


def test_dual_video_loss_requires_base_depth() -> None:
    config = load_config("configs/training/generation/libero_ci2va_mot.yaml")
    config["model"]["factory_kwargs"]["loss"]["base_model_coeff"] = 1.0
    try:
        validate_build(config)
    except ValueError as error:
        assert "base_model_depth" in str(error)
    else:
        raise AssertionError("base_model_coeff without base_model_depth must fail")


def test_local_fastwam_is_an_explicit_experimental_adapter() -> None:
    import vrae.models.dit as dit
    from vrae.models.dit.fastwam import FastWAM

    assert not hasattr(dit, "FastWAM")
    assert hasattr(FastWAM, "from_components")
    assert not hasattr(FastWAM, "from_wan22_pretrained")


def test_ci2va_temporal_indices_allow_condition_frame() -> None:
    assert build_temporal_indices(21, 1)[-1] == 20


def test_null_max_steps_uses_epochs() -> None:
    config = {"training": {"max_steps": None, "epochs": 10}}
    assert _target_steps(config, max_steps=None, num_batches=25) == 250


class _CheckpointModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mot = torch.nn.Linear(2, 2)
        self.dit = self.mot
        self.save_calls = 0

    def forward(self, batch):
        return self.training_loss(batch)

    def save_checkpoint(
        self,
        path,
        optimizer=None,
        step=None,
        training_state=None,
    ) -> None:
        self.save_calls += 1
        torch.save(
            {
                "mot": self.mot.state_dict(),
                "optimizer": optimizer.state_dict(),
                "step": step,
                "training_state": training_state,
            },
            path,
        )

    def load_checkpoint(self, path, optimizer=None):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        self.mot.load_state_dict(payload["mot"])
        if optimizer is not None:
            optimizer.load_state_dict(payload["optimizer"])
        return payload


def test_checkpoint_is_saved_once_and_latest_pointer_resumes(tmp_path) -> None:
    model = _CheckpointModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.25)
    loss = model.mot(torch.ones(1, 2)).sum()
    loss.backward()
    optimizer.step()
    expected_weight = model.mot.weight.detach().clone()
    expected_bias = model.mot.bias.detach().clone()
    expected_state = optimizer.state_dict()

    checkpoint = _save_checkpoint(
        model,
        optimizer,
        tmp_path,
        step=3,
        training_state={"epoch": 1, "batch_in_epoch": 2},
    )
    assert model.save_calls == 1
    pointer_path = tmp_path / "checkpoints" / "latest.json"
    assert sorted(path.name for path in pointer_path.parent.iterdir()) == [
        "latest.json",
        "step-00000003.pt",
    ]
    assert json.loads(pointer_path.read_text()) == {
        "path": "step-00000003.pt",
        "step": 3,
    }
    assert _resolve_resume_path(pointer_path, tmp_path) == checkpoint
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert payload["training_state"] == {"epoch": 1, "batch_in_epoch": 2}

    restored = _CheckpointModel()
    restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=1.0)
    step = _load_checkpoint(restored, restored_optimizer, checkpoint)
    assert step == 3
    torch.testing.assert_close(restored.mot.weight, expected_weight)
    torch.testing.assert_close(restored.mot.bias, expected_bias)
    assert restored_optimizer.param_groups[0]["lr"] == 0.25
    restored_state = restored_optimizer.state_dict()["state"]
    assert restored_state.keys() == expected_state["state"].keys()
    for parameter_id, values in expected_state["state"].items():
        for name, expected in values.items():
            torch.testing.assert_close(restored_state[parameter_id][name], expected)


def test_resumed_training_logs_next_step(monkeypatch, tmp_path, capsys) -> None:
    from vrae.training.fastwam import train as train_module

    model = _CheckpointModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.25)
    checkpoint = tmp_path / "resume.pt"
    model.save_checkpoint(checkpoint, optimizer=optimizer, step=3)

    def training_loss(batch):
        loss = model.mot(batch["value"]).square().mean()
        return loss, {"loss_video": float(loss.detach()), "loss_action": 0.0}

    model.training_loss = training_loss
    config = load_config("configs/training/generation/libero_ci2va_mot.yaml")
    config["training"].update(
        {
            "num_workers": 0,
            "epochs": 1,
            "max_steps": 4,
            "resume": str(checkpoint),
            "log_interval": 1,
            "checkpoint_interval": 100,
            "profile": False,
        }
    )
    config["sampling"]["enabled"] = False
    config["wandb"]["enabled"] = False
    batches = [{"value": torch.ones(1, 2)} for _ in range(4)]
    context = SimpleNamespace(
        world_size=1,
        device=torch.device("cpu"),
        is_main=True,
        local_rank=0,
    )
    monkeypatch.setattr(train_module, "initialize_distributed", lambda: context)
    monkeypatch.setattr(train_module, "_build_dataset", lambda config, paths: batches)
    monkeypatch.setattr(train_module, "_build_model", lambda config, device: model)
    paths = SimpleNamespace(project_root=tmp_path)

    train_module.train(config, paths, tmp_path / "run", max_steps=None)

    output = capsys.readouterr().out
    assert "step=4" in output
    assert "step=5" not in output


def test_resume_at_target_does_not_train(monkeypatch, tmp_path, capsys) -> None:
    from vrae.training.fastwam import train as train_module

    model = _CheckpointModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.25)
    checkpoint = tmp_path / "resume.pt"
    model.save_checkpoint(checkpoint, optimizer=optimizer, step=3)

    def training_loss(batch):
        raise AssertionError("training_loss must not run when resume step reached max_steps")

    model.training_loss = training_loss
    config = load_config("configs/training/generation/libero_ci2va_mot.yaml")
    config["training"].update(
        {
            "num_workers": 0,
            "epochs": 1,
            "max_steps": 3,
            "resume": str(checkpoint),
            "profile": False,
        }
    )
    config["sampling"]["enabled"] = False
    config["wandb"]["enabled"] = False
    batches = [{"value": torch.ones(1, 2)} for _ in range(4)]
    context = SimpleNamespace(
        world_size=1,
        device=torch.device("cpu"),
        is_main=True,
        local_rank=0,
    )
    monkeypatch.setattr(train_module, "initialize_distributed", lambda: context)
    monkeypatch.setattr(train_module, "_build_dataset", lambda config, paths: batches)
    monkeypatch.setattr(train_module, "_build_model", lambda config, device: model)
    paths = SimpleNamespace(project_root=tmp_path)

    train_module.train(config, paths, tmp_path / "run", max_steps=None)

    assert "step=4" not in capsys.readouterr().out


def test_training_logs_all_losses_in_one_wandb_payload(monkeypatch, tmp_path) -> None:
    from vrae.training.fastwam import train as train_module

    model = _CheckpointModel()

    def training_loss(batch):
        loss = model.mot(batch["value"]).square().mean()
        return loss, {
            "loss_video_full": 2.0,
            "loss_video_base": 3.0,
            "loss_action": 1.0,
        }

    model.training_loss = training_loss
    config = load_config("configs/training/generation/libero_ci2va_mot.yaml")
    config["training"].update(
        {
            "num_workers": 0,
            "max_steps": 1,
            "resume": None,
            "init_from": None,
            "log_interval": 1,
            "checkpoint_interval": 100,
            "profile": False,
        }
    )
    config["sampling"]["enabled"] = False
    context = SimpleNamespace(
        world_size=1,
        device=torch.device("cpu"),
        is_main=True,
        local_rank=0,
    )
    logged = []
    logger_enabled = []

    class Logger:
        def __init__(self, *args, **kwargs):
            del args
            logger_enabled.append(kwargs["enabled"])

        def log(self, values, *, step):
            logged.append((dict(values), step))

        def finish(self):
            pass

    monkeypatch.setattr(train_module, "initialize_distributed", lambda: context)
    monkeypatch.setattr(
        train_module,
        "_build_dataset",
        lambda config, paths: [{"value": torch.ones(1, 2)}],
    )
    monkeypatch.setattr(train_module, "_build_model", lambda config, device: model)
    monkeypatch.setattr(train_module, "WandbLogger", Logger)
    paths = SimpleNamespace(project_root=tmp_path)

    train_module.train(config, paths, tmp_path / "run", max_steps=None)

    assert logger_enabled == [True]
    assert len(logged) == 1
    payload, logged_step = logged[0]
    assert logged_step == 1
    assert payload["train/loss_video_full"] == 2.0
    assert payload["train/loss_video_base"] == 3.0
    assert payload["train/loss_action"] == 1.0
    assert "train/loss" in payload


def test_init_from_loads_weights_without_optimizer_state(tmp_path) -> None:
    source = _CheckpointModel()
    optimizer = torch.optim.AdamW(source.parameters(), lr=0.25)
    checkpoint = tmp_path / "source.pt"
    source.save_checkpoint(checkpoint, optimizer=optimizer, step=17)

    restored = _CheckpointModel()
    source_step = _load_weights(restored, checkpoint)

    assert source_step == 17
    torch.testing.assert_close(restored.mot.weight, source.mot.weight)
    torch.testing.assert_close(restored.mot.bias, source.mot.bias)


def test_prepare_trainable_parameters_freezes_non_training_modules() -> None:
    class Model(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.dit = torch.nn.Linear(2, 2)
            self.proprio_encoder = torch.nn.Linear(2, 2)
            self.vae = torch.nn.Linear(2, 2)

    model = Model().train()
    trainable = _prepare_trainable_parameters(model)

    assert not model.training
    assert model.dit.training
    assert model.proprio_encoder.training
    assert not model.vae.training
    assert all(parameter.requires_grad for parameter in trainable)
    assert all(not parameter.requires_grad for parameter in model.vae.parameters())


def test_loss_values_for_logging_supports_dual_and_legacy_metrics() -> None:
    loss = torch.tensor(6.0)
    dual = _loss_values_for_logging(
        loss,
        {
            "loss_video_full": 2.0,
            "loss_video_base": 3.0,
            "loss_action": 1.0,
        },
    )
    legacy = _loss_values_for_logging(
        loss,
        {"loss_video": 5.0, "loss_action": 1.0},
    )

    torch.testing.assert_close(dual, torch.tensor([6.0, 2.0, 3.0, 1.0]))
    torch.testing.assert_close(legacy, torch.tensor([6.0, 5.0, 0.0, 1.0]))


def test_fixed_sample_indices_are_reproducible() -> None:
    assert fixed_sample_indices(100, 4, 3407) == fixed_sample_indices(100, 4, 3407)
    assert len(set(fixed_sample_indices(100, 4, 3407))) == 4


def test_pose_trajectory_includes_start_and_composes_axis_angle_deltas() -> None:
    action = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0, 0.0, torch.pi / 2, 0.0],
            [0.0, 2.0, 0.0, 0.0, 0.0, torch.pi / 2, 0.0],
        ]
    )
    initial_state = torch.tensor([10.0, 20.0, 30.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    layout = {
        "mode": "delta",
        "position_indices": [0, 1, 2],
        "orientation_indices": [3, 4, 5],
        "state_position_indices": [0, 1, 2],
        "state_orientation_indices": [3, 4, 5],
    }

    position, rotation = _pose_trajectory(action, initial_state, layout)

    torch.testing.assert_close(
        position,
        torch.tensor(
            [
                [10.0, 20.0, 30.0],
                [11.0, 20.0, 30.0],
                [11.0, 22.0, 30.0],
            ]
        ),
    )
    torch.testing.assert_close(rotation[0], torch.eye(3))
    torch.testing.assert_close(
        rotation[-1],
        torch.diag(torch.tensor([-1.0, -1.0, 1.0])),
        atol=1.0e-5,
        rtol=1.0e-5,
    )


def test_profiler_instruments_fastwam_phases() -> None:
    class Model(torch.nn.Module):
        def _encode_video_latents(self, value):
            return value + 1

        def _joint_denoise_core(self, value):
            return value * 2

    model = Model()
    profiler = StepProfiler(enabled=True, device=torch.device("cpu"))
    instrument_fastwam(model, profiler)

    assert model._encode_video_latents(torch.tensor(1)) == 2
    assert model._joint_denoise_core(torch.tensor(2)) == 4
    timings = profiler.averages()
    assert timings["vae_encode_ms"] >= 0
    assert timings["denoise_forward_ms"] >= 0


def test_profiler_reports_rolling_data_wait_p95() -> None:
    profiler = StepProfiler(
        enabled=True,
        device=torch.device("cpu"),
        window_steps=4,
    )
    for value in (1.0, 2.0, 3.0, 4.0, 100.0):
        profiler.record_wall("data_wait_ms", value)

    timings = profiler.averages()
    assert timings["data_wait_ms"] == 27.25
    assert timings["data_wait_p95_ms"] == 100.0


def test_sampling_saves_and_uploads_the_same_artifacts(tmp_path) -> None:
    class Dataset:
        def __getitem__(self, index):
            del index
            return {
                "video": torch.zeros(3, 5, 32, 64),
                "prompt": "test task",
                "action": torch.zeros(4, 7),
                "proprio": torch.zeros(4, 8),
            }

        @staticmethod
        def denormalize_action(value):
            return value.detach().float().cpu()

        @staticmethod
        def denormalize_state(value):
            return value.detach().float().cpu()

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.frozen = torch.nn.Linear(1, 1)

        def infer_joint(self, **kwargs):
            assert kwargs["input_image"].shape == (3, 32, 64)
            frames = [Image.new("RGB", (64, 32), color=(index, 0, 0)) for index in range(5)]
            return {"video": frames, "action": torch.zeros(4, 7)}

    class Logger:
        def __init__(self):
            self.video_calls = {}

        def log_video_files(self, name, paths, *, step):
            self.video_calls[name] = list(paths)

        def log(self, values, *, step):
            assert values == {"sample/step": step}

    config = {
        "seed": 3407,
        "fps": 2,
        "num_video_frames": 5,
        "action_horizon": 4,
        "num_inference_steps": 2,
        "save_video": True,
        "save_trajectory": True,
        "upload_wandb": True,
        "action_layout": {
            "mode": "delta",
            "position_indices": [0, 1, 2],
            "orientation_indices": [3, 4, 5],
            "gripper_index": 6,
            "state_position_indices": [0, 1, 2],
        },
    }
    model = Model().train()
    model.frozen.eval()
    logger = Logger()

    sample_ms, upload_ms = run_sampling(
        model,
        Dataset(),
        [0],
        config,
        tmp_path,
        step=4,
        wandb_logger=logger,
    )

    assert sample_ms > 0
    assert upload_ms >= 0
    assert model.training
    assert not model.frozen.training
    assert logger.video_calls["sample"][0].is_file()
    trajectory_path = logger.video_calls["sample/trajectory"][0]
    assert trajectory_path.is_file()
    assert trajectory_path.suffix == ".mp4"
