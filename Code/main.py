import sys

from model_run import Runner


def main(singleCompany=False):
    network = Runner(singleCompany)
    network.run(singleCompany)


# TODO: Add a command line argument to run on a specific model.
if __name__ == "__main__":
    try:
        if len(sys.argv) - 1 == 1 and sys.argv[1] == "--company":
            while True:
                type_in = input(">> ")
                if type_in == ":exit":
                    break
                runquery = main(type_in)

        elif len(sys.argv) - 1 == 0:
            runquery = main()

        else:
            raise AttributeError("Invalid input flag.")

    except (IOError, FileNotFoundError):
        print("Input error, Company not recognised.")
        sys.exit(1)
    except KeyError:
        print("Input error, Company not recognised.")
        sys.exit(1)

