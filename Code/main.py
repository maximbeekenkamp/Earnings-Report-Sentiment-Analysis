import argparse
import sys

from model_run import Runner


def main(embedding_type, singleCompany=False):
    training_vars = {
        # LM
        "embedding_size": 256,
        "num_heads": 8,
        "num_layers": 6,
        "seq_len": 256,
        "batch_size": 2,
        # VAE
        "latent_dim": 128,
        "vae epochs": 5,
        "learning_rate": 0.001,
    }

    network = Runner(embedding_type, training_vars, singleCompany)
    network.run(singleCompany)


def parse_args():
    """
    Parse command line arguments.

    Returns:
        argparse.Namespace: Arguments.
    """
    parser = argparse.ArgumentParser(description="Run the project.")
    parser.add_argument(
        "--embedding",
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
                runquery = main(args.embedding, type_in)

        else:
            main(args.embedding)

    except (IOError, FileNotFoundError):
        print("Input error, Company not recognised.")
        sys.exit(1)
    # except KeyError:
    #     print("Input error, Company not recognised.")
    #     sys.exit(1)
