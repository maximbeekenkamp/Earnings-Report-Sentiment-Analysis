import argparse
import re
import sys

from model_run import Runner


def main(embedding_type, singleCompany=False):
    training_vars = {
        "epochs": 100,
        "batch_size": 32,
        "learning_rate": 0.001,
    }

    network = Runner(singleCompany)
    network.run(embedding_type, training_vars, singleCompany)

def parse_args():
    """
    Parse command line arguments.

    Returns:
        argparse.Namespace: Arguments.
    """
    parser = argparse.ArgumentParser(description="Run the project.")
    parser.add_argument(
        "--emmbedding",
        required=True,
        choices=["tfidf", "lstm", "gru", "sa"],
        help="Type of model to run.",
    )
    parser.add_argument(
        "--company",
        action="store_true",
        help="Run the project for a specific company.",
    )

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    try:
        if args.company:
            while True:
                type_in = input(">> ")
                if type_in == ":exit":
                    break
                runquery = main(args.type, type_in)

        else:
            main(args.type)

    except (IOError, FileNotFoundError):
        print("Input error, Company not recognised.")
        sys.exit(1)
    except KeyError:
        print("Input error, Company not recognised.")
        sys.exit(1)

