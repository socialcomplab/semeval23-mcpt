import numpy as np
import torch

from contrastlearning import Trainer
from .test_base import TestBase


class TestTrainer(TestBase):
    def __init__(self, name='TestTrainer'):
        super().__init__(name)

    def test(self):
        pass


def main():
    TestTrainer().run_testcases()


if __name__ == "__main__":
    main()
