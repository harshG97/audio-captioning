"""Tests for train.RecapTrainer: evaluation reuses one set of dataloader
workers (no process leak over a long run), dataloader worker seeding stays
out of the global RNG, and the training data order is unchanged.

Uses a toy model and dataset with real worker processes; no GPU needed.
Run: python -m pytest tests/test_recap_trainer.py -q
"""
import multiprocessing
import sys
from pathlib import Path

import pytest
import torch
from torch.utils.data import Dataset
from transformers import Trainer, TrainerCallback, TrainingArguments

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import train  # noqa: E402

WORKERS = 2


class Numbers(Dataset):
    def __init__(self, n=24):
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        return {"x": torch.full((4,), float(i)), "labels": torch.zeros(1)}


class Toy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(4, 1)
        self.drop = torch.nn.Dropout(0.5)       # draws from the global RNG in training
        self.seen = []

    def forward(self, x, labels):
        if self.training:
            self.seen += x[:, 0].long().tolist()
        return {"loss": ((self.lin(self.drop(x)) - labels) ** 2).mean()}


def _args(tmp_path, **kw):
    base = dict(output_dir=str(tmp_path / "out"), per_device_train_batch_size=4,
                per_device_eval_batch_size=4, dataloader_num_workers=WORKERS,
                dataloader_persistent_workers=True, report_to=[], use_cpu=True,
                save_strategy="no", logging_strategy="no", disable_tqdm=True, seed=42)
    base.update(kw)
    return TrainingArguments(**base)


def _workers():
    return len(multiprocessing.active_children())


class _CountAtEval(TrainerCallback):
    def __init__(self):
        self.counts = []

    def on_evaluate(self, *args, **kwargs):
        self.counts.append(_workers())


def test_evaluation_workers_are_started_once(tmp_path):
    counter = _CountAtEval()
    trainer = train.RecapTrainer(model=Toy(), args=_args(tmp_path, num_train_epochs=5, eval_strategy="epoch"),
                                 train_dataset=Numbers(), eval_dataset=Numbers(8), callbacks=[counter])
    trainer.train()
    # one set of training workers plus one set of evaluation workers, every epoch
    assert counter.counts == [2 * WORKERS] * 5


def test_repeated_evaluate_reuses_the_loader(tmp_path):
    trainer = train.RecapTrainer(model=Toy(), args=_args(tmp_path), train_dataset=Numbers(),
                                 eval_dataset=Numbers(8))
    loader = trainer.get_eval_dataloader()
    assert trainer.get_eval_dataloader(trainer.eval_dataset) is loader   # what evaluate() passes
    counts = []
    for _ in range(4):
        trainer.evaluate()
        counts.append(_workers())
    assert counts == [WORKERS] * 4


def test_evaluation_leaves_the_global_rng_alone(tmp_path):
    trainer = train.RecapTrainer(model=Toy(), args=_args(tmp_path), train_dataset=Numbers(),
                                 eval_dataset=Numbers(8))
    before = torch.get_rng_state().clone()
    trainer.evaluate()
    trainer.evaluate()
    assert torch.equal(before, torch.get_rng_state())


def test_one_off_eval_datasets_are_not_cached(tmp_path):
    trainer = train.RecapTrainer(model=Toy(), args=_args(tmp_path), train_dataset=Numbers(),
                                 eval_dataset=Numbers(8))
    other = Numbers(4)
    assert trainer.get_eval_dataloader(other) is not trainer.get_eval_dataloader(other)
    assert trainer.evaluate(eval_dataset=other)["eval_loss"] >= 0


def test_without_persistent_workers_nothing_is_cached(tmp_path):
    trainer = train.RecapTrainer(model=Toy(), args=_args(tmp_path, dataloader_persistent_workers=False),
                                 train_dataset=Numbers(), eval_dataset=Numbers(8))
    assert trainer.get_eval_dataloader() is not trainer.get_eval_dataloader()


def test_training_loader_has_its_own_generator(tmp_path):
    trainer = train.RecapTrainer(model=Toy(), args=_args(tmp_path), train_dataset=Numbers())
    loader = trainer.get_train_dataloader()
    inner = getattr(loader, "base_dataloader", loader)
    assert loader.generator is not None and inner.generator is not None


@pytest.mark.parametrize("epochs", [3])
def test_training_data_order_matches_the_stock_trainer(tmp_path, epochs):
    def order(cls):
        torch.manual_seed(0)
        model = Toy()
        cls(model=model, args=_args(tmp_path, num_train_epochs=epochs), train_dataset=Numbers()).train()
        return model.seen
    stock, ours = order(Trainer), order(train.RecapTrainer)
    assert len(ours) == epochs * len(Numbers()) and ours == stock
