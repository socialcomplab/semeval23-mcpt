import sys

from tests import *


def main():
    args = sys.argv[1:]
    if not len(args):
        args = ['TestDataManager', 'TestLoss', 'TestContrastLoss2', 'TestSampler', 'TestTrainer', 'TestTrainerA']
    for arg in args:
        if arg == 'TestDataManager':
            TestDataManager().run_testcases()
        if arg == 'TestLoss':
            TestLoss().run_testcases()
        if arg == 'TestContrastLoss2':
            TestContrastLoss2().run_testcases()
        if arg == 'TestSampler':
            TestSampler().run_testcases()
        if arg == 'TestTrainer':
            TestTrainer().run_testcases()
        if arg == 'TestTrainerA':
            TestTrainerA().run_testcases()


if __name__ == "__main__":
    main()
