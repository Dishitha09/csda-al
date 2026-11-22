import json
from pathlib import Path
from csda_al.cli import main


ROOT = Path(__file__).resolve().parents[1]


def test_coldstart_command(capsys):
    assert main(["cold-start", "--synthetic", "--budget", "8"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert len(result["image_ids"]) == 8 and result["synthetic"] is True


def test_missing_real_dataset_is_actionable(capsys):
    assert main(["inspect-data", "--config", str(ROOT / "configs/dataset.yaml")]) == 2
    assert "image directory does not exist" in capsys.readouterr().err


def test_synthetic_feature_command(tmp_path, capsys):
    assert main(["extract-features", "--config", str(ROOT / "configs/experiment.yaml"),
                 "--dry-run", "--output", str(tmp_path / "features")]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["feature_shape"] == [32, 512]
    assert result["no_real_training"]


def test_synthetic_run_command(tmp_path, capsys):
    output = tmp_path / "run"
    argv = ["run-al", "--config", str(ROOT / "configs/experiment.yaml"), "--synthetic", "--dry-run",
            "--output", str(output)]
    assert main(argv) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["training_annotations"] == 16 and result["training_steps"] == 2
    assert result["validation_annotations"] == 8 and result["rounds"] == 2
    assert len({i for q in result["history"] for i in q["image_ids"]}) == 16
    assert main([*argv, "--resume"]) == 0
    resumed = json.loads(capsys.readouterr().out)
    assert resumed == result


def test_ssl_command_real_optimizer_steps(tmp_path, capsys):
    assert main(["pretrain-ssl", "--config", str(ROOT / "configs/simclr.yaml"), "--dry-run",
                 "--output", str(tmp_path / "ssl")]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["training"]["steps"] == 2 and result["training"]["target_labels_used"] == 0
