import sys
import traceback

RED = "\033[1;31m"
BLUE = "\033[1;34m"
CYAN = "\033[1;36m"
GREEN = "\033[0;32m"
RESET = "\033[0;0m"
BOLD = "\033[;1m"
REVERSE = "\033[;7m"


class TestBase:
    def __init__(self, name):
        self.name = name

    def run_testcases(self):
        sys.stdout.write(BOLD)
        print()
        print(self.name)
        sys.stdout.write(RESET)
        testcases = [tc for tc in dir(self) if tc.startswith("test_")]
        failed = 0
        for testcase in testcases:
            try:
                getattr(self, testcase)()
                sys.stdout.write(GREEN)
                print(f"{testcase:<60} SUCCESS")
            except Exception as e:
                sys.stdout.write(RED)
                print(f"{testcase:<60} FAILED")
                print(e)
                traceback.print_exc()
                failed += 1
            finally:
                sys.stdout.write(RESET)
                sys.stdout.flush()
        if failed > 0:
            sys.stdout.write(RED)
            print(f"{failed} testcases failed.")
            sys.stdout.write(RESET)
