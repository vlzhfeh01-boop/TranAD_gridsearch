import argparse

def get_args():
    parser = argparse.ArgumentParser(description="Time-Series Anomaly Detection")
    parser.add_argument(
        "--dataset",
        metavar="-d",
        type=str,
        required=False,
        default="BAT",
        help="dataset from ['synthetic', 'SMD']",
    )
    parser.add_argument(
        "--model",
        metavar="-m",
        type=str,
        required=False,
        default="TranAD",
        help="model name",
    )

    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default="src2/params.json",
        help="path to config json file",
    )
    parser.add_argument("--test", action="store_true", help="test the model")
    parser.add_argument("--retrain", action="store_true", help="retrain the model")
    return parser.parse_args()
