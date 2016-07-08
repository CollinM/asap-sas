import argparse

from asap.core import load_test_instances, Pipeline
from asap.core.runners import PipelineRunner


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("pipeline_path")
    ap.add_argument("input_path")
    ap.add_argument("output_path")
    return ap.parse_args()

if __name__ == "__main__":

    args = parse_args()

    print("Loading data...", end='')
    # Load instance data
    test = load_test_instances(args.input_path)
    print("Done")

    print("Loading pipeline...", end='')
    pipe = Pipeline.load(args.pipeline_path)
    print("Done")

    runner = PipelineRunner(pipe, test, 'score1', 'prediction', output_path=args.output_path, evaluate=True)
    results = runner.run()

    print("Done")
