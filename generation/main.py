from Generator import Generator
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import json


def main():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--type",
        choices=[
            "poly",
            "homo2",
            "homo3",
            "inhomo2",
            "inhomo3",
        ],
        default="poly",
    )
    parser.add_argument("--config", default=None)
    parser.add_argument("--name", default="generated_pairs.csv")
    args = parser.parse_args()
    config = None
    if args.config is not None:
        with open(args.config, "r") as file:
            config = json.load(file)
    if args.type == "poly":
        if config is None:
            Generator.generate_polynomial(
                {2: 1000000, 3: 1000000, 4: 1000000, 5: 1000000}, -5000, 5000, True
            ).to_csv(args.name)
        else:
            Generator.generate_polynomial(
                config["amounts"], config["start"], config["end"], True
            ).to_csv(args.name)
    elif args.type == "homo2":
        if config is None:
            Generator.fast_linear_homogeneous_second_order(
                a_range=range(-100, 100), b_range=(-100, 100), return_df=True
            ).to_csv(args.name)
        else:
            Generator.fast_linear_homogeneous_second_order(
                a_range=config["a_range"], b_range=config["b_range"], return_df=True
            ).to_csv(args.name)
    elif args.type == "homo3":
        if config is None:
            Generator.fast_linear_homogeneous_second_order(
                a_range=range(-100, 100),
                b_range=(-100, 100),
                c_range=(-100, 100),
                return_df=True,
            ).to_csv(args.name)
        else:
            Generator.fast_linear_homogeneous_third_order(
                a_range=config["a_range"],
                b_range=config["b_range"],
                c_range=config["c_range"],
                return_df=True,
            ).to_csv(args.name)
    elif args.type == "inhomo2":
        if config is None:
            Generator().linear_inhomogeneous_second_order(
                a_range=range(-100, 100),
                b_range=(-100, 100),
                c_range=(-100, 100),
                rhs_length=2,
                return_df=True,
            ).to_csv(args.name)
        else:
            Generator().fast_linear_homogeneous_third_order(
                a_range=config["a_range"],
                b_range=config["b_range"],
                c_range=config["c_range"],
                rhs_length=config["rhs_length"],
                return_df=True,
            ).to_csv(args.name)
    else:
        if config is None:
            Generator().linear_inhomogeneous_second_order(
                a_range=range(-100, 100),
                b_range=(-100, 100),
                c_range=(-100, 100),
                d_range=(-100, 100),
                rhs_length=2,
                return_df=True,
            ).to_csv(args.name)
        else:
            Generator().fast_linear_homogeneous_third_order(
                a_range=config["a_range"],
                b_range=config["b_range"],
                c_range=config["c_range"],
                d_range=config["d_range"],
                rhs_length=config["rhs_length"],
                return_df=True,
            ).to_csv(args.name)

if __name__ == "__main__":
    main()