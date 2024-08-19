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
                {2: 1000, 3: 1000, 4: 1000, 5: 1000, 6: 1000, 7: 1000, 8: 1000, 9: 1000}, -5000, 5000, True
            ).to_csv(args.name, index=False)
        else:
            Generator.generate_polynomial(
                config["amounts"], config["start"], config["end"], True
            ).to_csv(args.name)
    elif args.type == "homo2":
        if config is None:
            Generator.fast_linear_homogeneous_second_order(
                a_range=range(-50, 50), b_range=range(-50, 50), return_df=True
            ).to_csv(args.name)
        else:
            Generator.fast_linear_homogeneous_second_order(
                a_range=config["a_range"], b_range=config["b_range"], return_df=True
            ).to_csv(args.name)
    elif args.type == "homo3":
        if config is None:
            Generator.fast_linear_homogeneous_third_order(
                a_range=range(-10, 10),
                b_range=range(-10, 10),
                c_range=range(-10, 10),
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
                a_range=range(-5, 6),
                b_range=range(-5, 6),
                c_range=range(-5, 6),
                rhs_length=1,
                return_df=True,
            ).to_csv(args.name, index=False)
        else:
            Generator().linear_inhomogeneous_second_order(
                a_range=config["a_range"],
                b_range=config["b_range"],
                c_range=config["c_range"],
                rhs_length=config["rhs_length"],
                return_df=True,
            ).to_csv(args.name, index=False)
    else:
        if config is None:
            Generator().linear_inhomogeneous_third_order(
                a_range=range(1, 5),
                b_range=range(-2, 3),
                c_range=range(-2, 3),
                d_range=range(-2, 3),
                rhs_length=1,
                return_df=True,
            ).to_csv(args.name, index=False)
        else:
            Generator().linear_inhomogeneous_third_order(
                a_range=config["a_range"],
                b_range=config["b_range"],
                c_range=config["c_range"],
                d_range=config["d_range"],
                rhs_length=config["rhs_length"],
                return_df=True,
            ).to_csv(args.name)

if __name__ == "__main__":
    main()
