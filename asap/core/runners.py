import os
from asap.metrics import ConfusionMatrix, write_qwk, write_results


class PipelineRunner(object):
    """Pipeline Train-Test-Eval-Save Runner"""

    def __init__(self, pipeline, test_data, gs_key, target_key, train_data=None, output_path=None, evaluate=False, save=False):
        self._pipe = pipeline
        self._gs_key = gs_key
        self._target_key = target_key
        self._test = test_data
        self._train = train_data

        if output_path is not None:
            self._out_path = output_path
            self._eval = True if evaluate else False
            self._save = True if save else False
        else:
            self._eval = False
            self._save = False

        self.qwk_score = 0

    def run(self):
        # Train pipeline
        if self._train is not None:
            print("Training pipeline...")
            self._pipe.train(self._train)
            print("Done")

        # Save pipeline
        if self._save:
            pipe_path = os.path.join(self._out_path, "pipe.pkl")
            print("Saving pipeline to " + str(pipe_path) + "...", end='')
            self._pipe.save(pipe_path)
            print("Done")

        # Run test data through pipeline
        print("Testing pipeline...", end='')
        results = self._pipe.run(self._test)
        print("Done")

        # Evaluate performance
        if self._eval:
            print("Gathering metrics...", end='')

            # Create confusion matrix
            cm = ConfusionMatrix(range(4))
            triples = []
            for inst in results:
                trip = (str(inst.id),
                        str(inst.get_feature(self._gs_key)[0]),
                        str(inst.get_feature(self._target_key)[0]))
                triples.append(trip)
                cm.increment(trip[1], trip[2])
            # Write confusion matrix
            cm_path = os.path.join(self._out_path, 'cm.csv')
            cm.write_csv(cm_path)

            # Write quadratic weighted kappa
            qwk_path = os.path.join(self._out_path, 'qwk.txt')
            ids, actuals, predxns = list(zip(*triples))
            self.qwk_score = write_qwk(actuals, predxns, qwk_path)

            # Write raw results
            results_path = os.path.join(self._out_path, 'results.csv')
            write_results(triples, results_path)

            print("Done")

        return results
