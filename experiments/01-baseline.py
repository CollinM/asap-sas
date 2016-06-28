import argparse
import os

from asap.core import load_instances, split_instances, Pipeline
from asap.core.features import Tokenizer, TFIDF
from asap.core.ml import RandomForest
from asap.metrics import ConfusionMatrix, write_qwk


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

        print("Loading data...")
        # Load instance data
        data = load_instances(input_item[1])
        # split into train and test
        train, test = split_instances(data, 0.8, 42)

        print("Creating pipeline...")
        # Create pipeline with tokenizer and TF-IDF
        pipe = Pipeline()
        pipe.add_phase(Tokenizer())
        pipe.add_phase(TFIDF(topk=2000))
        pipe.add_phase(RandomForest(num_trees=args.num_trees, max_depth=args.max_depth, target="score1", features=["tfidf"]))

        print("Training pipeline...")
        # Train pipeline
        trained = pipe.train(train)

        print("Running pipeline...")
        # Test pipeline
        evaled = pipe.run(test)

        # Make output directory
        output_dir = os.path.join(args.output_path, input_item[0])
        os.makedirs(output_dir)

        pipe_path = os.path.join(output_dir, "pipeline.pkl")
        print("Saving pipeline to " + pipe_path)
        pipe.save(pipe_path)

        print ("Gather metrics...")
        cm = ConfusionMatrix(range(4))
        actuals = []
        predxns = []
        for inst in evaled:
            actuals.append(str(inst.get_feature("score1")[0]))
            predxns.append(str(inst.get_feature("prediction")[0]))
            cm.increment(actuals[-1], predxns[-1])

        cm_path = os.path.join(output_dir, 'cm.csv')
        print("Writing confusion matrix to " + cm_path)
        cm.write_csv(cm_path)

        qwk_path = os.path.join(output_dir, 'qwk.txt')
        print("Writing quadratic weighted kappa to " + qwk_path)
        score = write_qwk(actuals, predxns, qwk_path)
        qwk_scores[input_item[0]] = score

    for input_item in inputs:
        print(input_item[0] + ": " + qwk_scores[input_item[0]])
    print("Avg: " + sum(qwk_scores.values() / len(qwk_scores)))

    print("Done")
