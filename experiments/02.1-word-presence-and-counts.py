import argparse
import os

from asap.core import load_instances, split_instances, gather_input_files, Pipeline
from asap.core.features import Tokenizer, ContainsWords, UniqueWordCount, WordCount, CharacterCount, NonWhitespaceCharacterCount
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


def make_pipeline(words_path, trees, depth):
    pipe = Pipeline()
    pipe.add_phase(Tokenizer())
    pipe.add_phase(ContainsWords(words_path))
    pipe.add_phase(UniqueWordCount())
    pipe.add_phase(WordCount())
    pipe.add_phase(CharacterCount())
    pipe.add_phase(NonWhitespaceCharacterCount())
    pipe.add_phase(RandomForest(num_trees=trees, max_depth=depth, target="score1", features=["word-presence", "unique-word-count", "word-count", "char-count", "!white-char-count"]))
    return pipe


if __name__ == "__main__":

    args = parse_args()

    # Gathering inputs
    inputs = gather_input_files(args.input_path)

    qwk_scores = {}
    for input_item in inputs:
        print("Working on [" + input_item[0] + "]")

        # Make output directory
        output_dir = os.path.join(args.output_path, input_item[0])
        os.makedirs(output_dir)

        print("Loading data...", end='')
        # Load instance data
        data = load_instances(input_item[1])
        # split into train and test
        train, test = split_instances(data, 0.8, 42)
        print("Done")

        print("Creating pipeline...", end='')
        # Create pipeline with tokenizer and word presence vector
        pipe = make_pipeline("data/keywords/keywords-" + str(input_item[0]) + ".txt", args.num_trees, args.max_depth)
        print("Done")

        runner = PipelineRunner(pipe, test, 'score1', 'prediction', train, output_dir, evaluate=True, save=True)
        results = runner.run()

        qwk_scores[input_item[0]] = runner.qwk_score

    qwk_score_path = os.path.join(args.output_path, 'qwk_scores.md')
    scores = []
    for input_item in inputs:
        scores.append(qwk_scores[input_item[0]])
    scores.append(sum(qwk_scores.values()) / len(qwk_scores))
    write_qwk_markdown_table(scores, qwk_score_path)
    print("Wrote QWK scores to " + qwk_score_path)

    print("Done")
