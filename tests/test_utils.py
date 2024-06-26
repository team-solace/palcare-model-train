# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.
from dataclasses import asdict

import os
from contextlib import redirect_stderr
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

import pytest
import torch
import torch.nn.functional as F
import yaml
from tests.conftest import RunIf
from lightning import Fabric
from lightning.fabric.loggers import CSVLogger, TensorBoardLogger
from lightning.fabric.plugins import BitsandbytesPrecision
from lightning.pytorch.loggers import WandbLogger
from lightning_utilities.core.imports import RequirementCache

from litgpt import GPT
from litgpt.args import TrainArgs
from litgpt.utils import (
    CLI,
    CycleIterator,
    capture_hparams,
    check_valid_checkpoint_dir,
    choose_logger,
    chunked_cross_entropy,
    copy_config_files,
    extend_checkpoint_dir,
    find_multiple,
    find_resume_path,
    incremental_save,
    init_out_dir,
    instantiate_bnb_optimizer,
    instantiate_torch_optimizer,
    num_parameters,
    parse_devices,
    save_hyperparameters,
)


def test_find_multiple():
    assert find_multiple(17, 5) == 20
    assert find_multiple(30, 7) == 35
    assert find_multiple(10, 2) == 10
    assert find_multiple(5, 10) == 10
    assert find_multiple(50254, 128) == 50304
    assert find_multiple(50254, 256) == 50432
    assert find_multiple(50254, 512) == 50688


# match fails on windows. why did they have to use backslashes?
@RunIf(skip_windows=True)
def test_check_valid_checkpoint_dir(tmp_path):
    os.chdir(tmp_path)

    out = StringIO()
    with pytest.raises(SystemExit), redirect_stderr(out):
        check_valid_checkpoint_dir(tmp_path)
    out = out.getvalue().strip()
    expected = f"""
checkpoint_dir '{str(tmp_path.absolute())}' is missing the files: ['lit_model.pth', 'model_config.yaml', 'tokenizer.json OR tokenizer.model', 'tokenizer_config.json'].
Find download instructions at https://github.com/Lightning-AI/litgpt/blob/main/tutorials

See all download options by running:
 litgpt download
    """.strip()
    assert out == expected

    out = StringIO()
    checkpoint_dir = tmp_path / "checkpoints" / "stabilityai" / "stablelm-base-alpha-3b"
    with pytest.raises(SystemExit), redirect_stderr(out):
        check_valid_checkpoint_dir(checkpoint_dir)
    out = out.getvalue().strip()
    expected = f"""
checkpoint_dir '{str(checkpoint_dir.absolute())}' is not a checkpoint directory.
Find download instructions at https://github.com/Lightning-AI/litgpt/blob/main/tutorials

See all download options by running:
 litgpt download
    """.strip()
    assert out == expected

    out = StringIO()
    checkpoint_dir.mkdir(parents=True)
    foo_checkpoint_dir = tmp_path / "foo"
    with pytest.raises(SystemExit), redirect_stderr(out):
        check_valid_checkpoint_dir(foo_checkpoint_dir)
    out = out.getvalue().strip()
    expected = f"""
checkpoint_dir '{str(foo_checkpoint_dir.absolute())}' is not a checkpoint directory.
Find download instructions at https://github.com/Lightning-AI/litgpt/blob/main/tutorials

You have downloaded locally:
'{str(checkpoint_dir.absolute())}'

See all download options by running:
 litgpt download
    """.strip()
    assert out == expected


def test_incremental_write(tmp_path):
    sd = {str(k): torch.randn(5, 10) for k in range(3)}
    sd["0"].someattr = 1
    sd_expected = {k: v.clone() for k, v in sd.items()}
    fn = str(tmp_path / "test.pt")
    with incremental_save(fn) as f:
        sd["0"] = f.store_early(sd["0"])
        sd["2"] = f.store_early(sd["2"])
        f.save(sd)
    sd_actual = torch.load(fn)
    assert sd_actual.keys() == sd_expected.keys()
    assert sd_actual["0"].someattr == 1  # requires PyTorch 2.0+
    for k, v_expected in sd_expected.items():
        v_actual = sd_actual[k]
        torch.testing.assert_close(v_expected, v_actual)


@pytest.mark.parametrize("B", (1, 2))
@pytest.mark.parametrize("ignore_index", (None, -1, -2, -100))
def test_chunked_cross_entropy(ignore_index, B):
    V = 50
    T = 25
    regular_logits = torch.randn(B, T, V)
    targets = torch.randint(0, V, (B, T))

    if ignore_index is not None:
        targets[:, [1, 4, 10, 19]] = ignore_index

    baseline_loss = F.cross_entropy(
        regular_logits.reshape(-1, regular_logits.size(-1)),
        targets.reshape(-1),
        ignore_index=(ignore_index if ignore_index is not None else -100),
    )

    ignore_index = ignore_index if ignore_index is not None else -100
    regular_loss = chunked_cross_entropy(regular_logits, targets, chunk_size=0, ignore_index=ignore_index)
    assert torch.equal(baseline_loss, regular_loss)
    assert regular_loss.numel() == 1

    chunked_loss = chunked_cross_entropy(regular_logits, targets, chunk_size=10, ignore_index=ignore_index)
    torch.testing.assert_close(chunked_loss, regular_loss)
    torch.testing.assert_close(chunked_loss, baseline_loss)

    logit_chunk_size = 6
    assert T % logit_chunk_size != 0  # ensure leftover
    chunked_logits = list(regular_logits.split(logit_chunk_size, dim=1))
    chunked_loss = chunked_cross_entropy(chunked_logits, targets, chunk_size=0, ignore_index=ignore_index)
    torch.testing.assert_close(chunked_loss, regular_loss)
    torch.testing.assert_close(chunked_loss, baseline_loss)

    chunked_loss = chunked_cross_entropy(chunked_logits, targets, chunk_size=10, ignore_index=ignore_index)
    torch.testing.assert_close(chunked_loss, regular_loss)
    torch.testing.assert_close(chunked_loss, baseline_loss)


def test_num_parameters():
    model = torch.nn.Linear(2, 2)
    assert num_parameters(model) == 6
    assert num_parameters(model, requires_grad=True) == 6
    assert num_parameters(model, requires_grad=False) == 0

    model = torch.nn.Linear(2, 2)
    model.bias.requires_grad = False
    assert num_parameters(model) == 6
    assert num_parameters(model, requires_grad=True) == 4
    assert num_parameters(model, requires_grad=False) == 2


@RunIf(min_cuda_gpus=1)
@pytest.mark.parametrize("mode", ["nf4", "nf4-dq", "fp4", "fp4-dq", "int8", "int8-training"])
def test_num_parameters_bitsandbytes(mode):
    plugin = BitsandbytesPrecision(mode=mode)
    fabric = Fabric(plugins=plugin, accelerator="cuda", devices=1)

    model = torch.nn.Linear(10, 10)
    model = fabric.setup(model)
    assert num_parameters(model) == 110

    with fabric.init_module(empty_init=True):
        model = GPT.from_name("pythia-14m")
    assert num_parameters(model) == 14067712


def test_cycle_iterator():
    iterator = CycleIterator([])
    with pytest.raises(StopIteration):
        next(iterator)

    iterator = CycleIterator(range(3))
    assert iterator.epoch == 0
    assert next(iterator) == 0
    assert iterator.epoch == 0
    assert next(iterator) == 1
    assert iterator.epoch == 0
    assert next(iterator) == 2
    assert iterator.epoch == 0
    assert next(iterator) == 0
    assert iterator.epoch == 1


def test_parse_devices():
    with pytest.raises(ValueError, match="must be 'auto' or a positive integer"):
        assert parse_devices(0)
    with pytest.raises(ValueError, match="must be 'auto' or a positive integer"):
        assert parse_devices(-2)

    with mock.patch("litgpt.utils.torch.cuda.device_count", return_value=0):
        assert parse_devices("auto") == 1  # CPU
        assert parse_devices(10) == 10  # leave validation up to Fabric later on
    with mock.patch("litgpt.utils.torch.cuda.device_count", return_value=1):
        assert parse_devices("auto") == 1  # CUDA
    with mock.patch("litgpt.utils.torch.cuda.device_count", return_value=3):
        assert parse_devices("auto") == 3
        assert parse_devices(-1) == 3

    assert parse_devices(5) == 5


def test_copy_config_files(fake_checkpoint_dir, tmp_path):
    copy_config_files(fake_checkpoint_dir, tmp_path)
    expected = {"model_config.yaml", "tokenizer_config.json", "tokenizer.json"}
    contents = set(os.listdir(tmp_path))
    assert expected.issubset(contents)


def test_capture_hparams():
    integer = 1
    string = "string"
    boolean = True
    none = None
    path = Path("/path")
    dataclass = TrainArgs()
    other = torch.nn.Linear(1, 1)
    hparams = capture_hparams()
    assert hparams == {
        "integer": integer,
        "string": string,
        "boolean": boolean,
        "none": none,
        "path": path,
        "dataclass": asdict(dataclass),
        "other": str(other),
    }


def _test_function(out_dir: Path, foo: bool = False, bar: int = 1):
    save_hyperparameters(_test_function, out_dir)


def test_save_hyperparameters(tmp_path):
    with mock.patch("sys.argv", ["any.py", str(tmp_path), "--foo", "True"]):
        CLI(_test_function)

    with open(tmp_path / "hyperparameters.yaml", "r", encoding="utf-8") as file:
        hparams = yaml.full_load(file)

    assert hparams["out_dir"] == str(tmp_path)
    assert hparams["foo"] is True
    assert hparams["bar"] == 1


def _test_function2(out_dir: Path, foo: bool = False, bar: int = 1):
    assert False, "I only exist as a signature, but I should not run."


@pytest.mark.parametrize(
    "command",
    [
        "any.py",
        "litgpt finetune",
        "litgpt finetune_full",
        "litgpt finetune_lora",
        "litgpt finetune_adapter",
        "litgpt finetune_adapter_v2",
        "litgpt pretrain",
    ],
)
def test_save_hyperparameters_known_commands(command, tmp_path):
    with mock.patch("sys.argv", [*command.split(" "), str(tmp_path), "--foo", "True"]):
        save_hyperparameters(_test_function2, tmp_path)

    with open(tmp_path / "hyperparameters.yaml", "r", encoding="utf-8") as file:
        hparams = yaml.full_load(file)

    assert hparams["out_dir"] == str(tmp_path)
    assert hparams["foo"] is True
    assert hparams["bar"] == 1


def test_choose_logger(tmp_path):
    assert isinstance(choose_logger("csv", out_dir=tmp_path, name="csv"), CSVLogger)
    if RequirementCache("tensorboard"):
        assert isinstance(choose_logger("tensorboard", out_dir=tmp_path, name="tb"), TensorBoardLogger)
    if RequirementCache("wandb"):
        assert isinstance(choose_logger("wandb", out_dir=tmp_path, name="wandb"), WandbLogger)

    with pytest.raises(ValueError, match="`--logger_name=foo` is not a valid option."):
        choose_logger("foo", out_dir=tmp_path, name="foo")


def test_init_out_dir(tmp_path):
    relative_path = Path("./out")
    absolute_path = tmp_path / "out"
    assert init_out_dir(relative_path) == relative_path
    assert init_out_dir(absolute_path) == absolute_path

    with mock.patch.dict(os.environ, {"LIGHTNING_ARTIFACTS_DIR": "prefix"}):
        assert init_out_dir(relative_path) == Path("prefix") / relative_path
        assert init_out_dir(absolute_path) == absolute_path


def test_find_resume_path(tmp_path):
    assert find_resume_path(resume=None, out_dir=Path("does/not/exist")) is None
    assert find_resume_path(resume=Path("does/not/exist"), out_dir=Path("does/not/matter")) == Path("does/not/exist")
    assert find_resume_path(resume=(tmp_path / "checkpoint.pt"), out_dir=Path("does/not/matter")) == (tmp_path / "checkpoint.pt")

    # `resume='auto'` does not enforce the checkpoint to exist
    assert find_resume_path(resume="auto", out_dir=Path("does/not/exist")) is None

    # `resume=True` requires a checkpoint to exist
    with pytest.raises(FileNotFoundError, match="You passed `--resume=True`, but no checkpont file was found"):
        find_resume_path(resume=True, out_dir=Path("does/not/exist"))
    with pytest.raises(FileNotFoundError, match="You passed `--resume=True`, but no checkpont file was found"):
        find_resume_path(resume=True, out_dir=tmp_path)

    (tmp_path / "step-001").mkdir()
    (tmp_path / "step-001" / "lit_model.pth").touch()
    (tmp_path / "step-002").mkdir()
    (tmp_path / "step-002" / "lit_model.pth").touch()
    (tmp_path / "step-003").mkdir()
    (tmp_path / "step-003" / "lit_model.pth").touch()

    assert find_resume_path(resume=True, out_dir=tmp_path) == (tmp_path / "step-003" / "lit_model.pth")
    assert find_resume_path(resume="auto", out_dir=tmp_path) == (tmp_path / "step-003" / "lit_model.pth")


@pytest.fixture
def model_parameters():
    return [torch.nn.Parameter(torch.randn(2, 2))]


def test_instantiate_bnb_optimizer_with_str(model_parameters):
    import bitsandbytes as bnb
    with mock.patch("litgpt.utils.get_argument_names", return_value={"lr", "eps", "weight_decay"}):
        optimizer = instantiate_bnb_optimizer("AdamW", model_parameters)
        assert isinstance(optimizer, bnb.optim.adamw.PagedAdamW)


def test_instantiate_bnb_optimizer_with_dict(model_parameters):
    import bitsandbytes as bnb
    optimizer_dict = {"class_path": "AdamW", "init_args": {"lr": 0.01}}
    with mock.patch("litgpt.utils.get_argument_names", return_value={"lr", "eps", "weight_decay"}):
        optimizer = instantiate_bnb_optimizer(optimizer_dict, model_parameters)
        assert isinstance(optimizer, bnb.optim.adamw.PagedAdamW)
        assert optimizer.param_groups[0]["lr"] == 0.01


def test_instantiate_bnb_optimizer_with_invalid_str(model_parameters):
    with pytest.raises(ValueError, match="only supports the AdamW"):
        instantiate_bnb_optimizer("SGD", model_parameters)


def test_instantiate_torch_optimizer_with_str(model_parameters):
    optimizer = instantiate_torch_optimizer("Adam", model_parameters, lr=0.01)
    assert isinstance(optimizer, torch.optim.Adam)
    assert optimizer.param_groups[0]["lr"] == 0.01


def test_instantiate_torch_optimizer_with_class(model_parameters):
    optimizer = instantiate_torch_optimizer({"class_path": "torch.optim.Adam", "init_args": {"lr": 123}}, model_parameters, lr=0.02)
    assert isinstance(optimizer, torch.optim.Adam)
    # init args gets overridden
    assert optimizer.param_groups[0]["lr"] == 0.02


@pytest.mark.parametrize("input_path, expected", [
    (Path("checkpoints/my_model"), Path("checkpoints/my_model")),
    (Path("checkpoints/my_model"), Path("./checkpoints/my_model")),
])
def test_extend_checkpoint_dir_is_prefixed(input_path, expected):
    original_dir = Path.cwd()  # Save the current directory
    with TemporaryDirectory() as tmp_dir:
        os.chdir(tmp_dir)

        try:
            if not input_path.is_absolute():
                input_path = Path(tmp_dir) / input_path
            if not expected.is_absolute():
                expected = Path(tmp_dir) / expected
            input_path.parent.mkdir(parents=True, exist_ok=True)
            input_path.touch(exist_ok=True)
            assert extend_checkpoint_dir(input_path) == expected
        finally:
            os.chdir(original_dir)  # Reset the current directory


@pytest.mark.parametrize("input_path, expected", [
    (Path("my_model"), Path("checkpoints/my_model")),
    (Path("my_model"), Path("./checkpoints/my_model")),
])
def test_extend_checkpoint_dir(input_path, expected):
    original_dir = Path.cwd()  # Save the current directory
    with TemporaryDirectory() as tmp_dir:
        os.chdir(tmp_dir)

        try:
            if not input_path.is_absolute():
                input_path = Path(tmp_dir) / "checkpoints" / input_path
            if not expected.is_absolute():
                expected = Path(tmp_dir) / expected
            input_path.parent.mkdir(parents=True, exist_ok=True)
            input_path.touch(exist_ok=True)
            assert extend_checkpoint_dir(input_path) == expected
        finally:
            os.chdir(original_dir)  # Reset the current directory


@pytest.mark.parametrize("input_path, expected", [
    (Path("my_model"), Path("my_model")),
    (Path("/my_model"), Path("/my_model")),
])
def test_extend_checkpoint_dir_dont_exist(input_path, expected):
    assert extend_checkpoint_dir(input_path) == expected
