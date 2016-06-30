import argparse
import os

from asap.core import load_instances, split_instances, Pipeline
from asap.core.features import Tokenizer, TFIDF
from asap.core.ml import RandomForest
from asap.core.runners import PipelineRunner


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("input_path")
    ap.add_argument("output_path")
    ap.add_argument("num_trees", type=int)
    ap.add_argument("max_depth", type=int)
    return ap.parse_args()


if __name__ == "__main__":

    args = parse_args()

    # Gathering inputs
    inputs = []
    for fname in os.listdir(args.input_path):
        full_name = os.path.join(args.input_path, fname)
        dot_idx = fname.find('.')
        num = fname[dot_idx-2:dot_idx]
        inputs.append((num, full_name))

    qwk_scores = {}
    for input_item in inputs:
        print("Working on [" + input_item[0] + "]")

        # Make output directory
        output_dir = os.path.join(args.output_path, input_item[0])
        os.makedirs(output_dir)

        print("Loading data...")
        # Load instance data
        data = load_instances(input_item[1])
        # split into train and test
        train, test = split_instances(data, 0.8, 42)

        print("Creating pipeline...", end='')
        # Create pipeline with tokenizer and TF-IDF
        pipe = Pipeline()
        pipe.add_phase(Tokenizer())
        pipe.add_phase(TFIDF(topk=2000))
        pipe.add_phase(RandomForest(num_trees=args.num_trees, max_depth=args.max_depth, target="score1", features=["tfidf"]))
        print("Done")

        runner = PipelineRunner(pipe, test, 'score1', 'prediction', train, output_dir, evaluate=True, save=True)
        results = runner.run()

        qwk_scores[input_item[0]] = runner.qwk_score

    for input_item in inputs:
        print(input_item[0] + ": " + str(qwk_scores[input_item[0]]))
    print("Avg: " + str(sum(qwk_scores.values()) / len(qwk_scores)))

    print("Done")
