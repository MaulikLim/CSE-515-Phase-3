import argparse


class Parser:
    def __init__(self, desc):
        self.parser = argparse.ArgumentParser(description=desc)

    def add_args(self, abbr, var_name, var_type, required):
        self.parser.add_argument(
            abbr,
            var_name,
            type=var_type,
            required=required
        )

    def parse_args(self):
        return self.parser.parse_args()
