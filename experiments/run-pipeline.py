import argparse

from asap.core import load_instances, split_instances, Pipeline
from asap.core.runners import PipelineRunner


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("pipeline_path")
    ap.add_argument("input_path")
    ap.add_argument("output_path")
    return ap.parse_args()

if __name__ == "__main__":

    args = parse_args()

    print("Loading data...")
    # Load instance data
    data = load_instances(args.input_path)
    # split into train and test
    train, test = split_instances(data, 0.8, 42)

    print("Loading pipeline...")
    pipe = Pipeline.load(args.pipeline_path)

    runner = PipelineRunner(pipe, test, 'score1', 'prediction', output_path=args.output_path, evaluate=True)
    results = runner.run()

    print("Done")
