import argparse
import os.path

from asap.core import load_instances, split_instances, Pipeline
from asap.metrics import ConfusionMatrix, write_qwk


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

    print("Running pipeline...")
    evaled = pipe.run(test)

    print("Gather metrics...")
    cm = ConfusionMatrix(range(4))
    actuals = []
    predxns = []
    for inst in evaled:
        actuals.append(str(inst.get_feature("score1")[0]))
        predxns.append(str(inst.get_feature("prediction")[0]))
        cm.increment(actuals[-1], predxns[-1])

    cm_path = os.path.join(args.output_path, 'cm.csv')
    print("Writing confusion matrix to " + cm_path)
    cm.write_csv(cm_path)

    qwk_path = os.path.join(args.output_path, 'qwk.txt')
    print("Writing quadratic weighted kappa to " + qwk_path)
    write_qwk(actuals, predxns, qwk_path)

    print("Done")
