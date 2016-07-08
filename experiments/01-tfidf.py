import argparse
import os

from asap.core import load_train_instances, load_test_instances, gather_input_files, Pipeline
from asap.core.features import Tokenizer, TFIDF
from asap.core.preprocess import PunctuationStripper, WhitespaceNormalizer, LowerCaser
from asap.core.ml import RandomForest
from asap.core.runners import PipelineRunner
from asap.metrics import write_qwk_markdown_table


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("input_path")
    ap.add_argument("output_path")
    ap.add_argument("num_trees", type=int)
    ap.add_argument("max_depth", type=int)
    return ap.parse_args()


def make_pipeline(top_k, trees, depth):
    pipe = Pipeline()
    # Preprocessing
    pipe.add_phase(PunctuationStripper())
    pipe.add_phase(WhitespaceNormalizer())
    pipe.add_phase(LowerCaser())
    # Features
    pipe.add_phase(Tokenizer())
    pipe.add_phase(TFIDF(topk=top_k))
    # ML
    pipe.add_phase(RandomForest(num_trees=trees, max_depth=depth, target="score1", features=["tfidf"]))
    return pipe

if __name__ == "__main__":

    args = parse_args()

    # Gathering inputs
    inputs = gather_input_files(args.input_path)

    qwk_scores = {}
    for num_id, train_path, test_path in inputs:
        print("Working on [" + num_id + "]")

        # Make output directory
        output_dir = os.path.join(args.output_path, num_id)
        os.makedirs(output_dir)

        print("Loading data...", end='')
        # Load instance data
        train = load_train_instances(train_path)
        test = load_test_instances(test_path)
        print("Done")

        print("Creating pipeline...", end='')
        # Create pipeline with tokenizer and TF-IDF
        pipe = make_pipeline(2000, args.num_trees, args.max_depth)
        print("Done")

        runner = PipelineRunner(pipe, test, 'score1', 'prediction', train, output_dir, evaluate=True, save=True)
        results = runner.run()

        qwk_scores[num_id] = runner.qwk_score

    qwk_score_path = os.path.join(args.output_path, 'qwk_scores.md')
    scores = []
    for item in inputs:
        scores.append(qwk_scores[item[0]])
    scores.append(sum(qwk_scores.values()) / len(qwk_scores))
    write_qwk_markdown_table(scores, qwk_score_path)
    print("Wrote QWK scores to " + qwk_score_path)

    print("Done")
