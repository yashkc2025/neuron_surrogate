import argparse
from scripts.training import train_specialist_model
from scripts.prediction import run_dynamic_hybrid_prediction

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Run specialist model training and hybrid predictions."
    )
    parser.add_argument(
        "x",
        type=int,
        default=2,
        help="Number of test runs to process with dynamic hybrid prediction",
    )
    args = parser.parse_args()
    x = args.x

    # Train model
    train_runs, test_runs = train_specialist_model()

    # Run predictions on first x test runs
    for run in test_runs[:x]:
        run_dynamic_hybrid_prediction(run)
