import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=str,
                        help="input file (*.yml)")
    parser.add_argument("-v", "--verbosity", type=int, choices=[0, 1, 2],
                        help="increase output verbosity")
