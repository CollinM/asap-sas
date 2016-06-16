# Lab Notebook

(Kaggle Competition page)[https://www.kaggle.com/c/asap-sas]

## Data Investigation

I perused the data and noticed a few things immediately. There are many misspellings, some records appear to be mistakenly truncated (bad data), and the use of punctuation is all over the place. This suggests a few things that could be done to preprocess the data: spell correction, remove truncated records, and normalize or remove all punctuation.

The distribution of the data seems to be fairly even across the 10 questions, with no single question dominating the distribution. After separating the data set by the `Essay Set` column, it becomes clear that the answers use quite different language to answer each question, which makes sense as the questions don't share topics. However, this poses a few questions: do the answers share a (sub-)language? Do only certain scores overlap in terms of language (e.g., bad answers share a language across all sets, but good answers differ)? Can we capture the structure of the answer regardless of its language?

Other miscellaneous observations:
- the length of the answer seems to correlate with score, with longer answers performing better.
- some essay sets have highly regular language/vocabularies, others do not.

## Experiment 1 - Baseline

### Preprocessing the data

Using the observations above, I made a quick and dirty (Python script)[../src/main/py/preprocess.py] to clean and normalize the text of the answers. Specifically,
- Strip all punctuation
- Normalize runs of whitespace to a single space
- Make all characters lowercase
After the above steps, 10 output files are generated, one containing each essay set's answers. Note that this process does not actually *remove* any records...yet.

### Random Forest with TF-IDF

I then created a Spark pipeline that generates 4,000 TF-IDF features and trains and tests a random forest classifier (200 trees, max depth of 20 per tree). I iteratively ran this class with each of the 10 separated essay sets and generated (10 sets of results)[../output/baseline]. The following command was used to launch each pipeline: `spark-submit --class collinm.asap.BaselineTfIdf --master local[7] build\libs\asap-0-fat.jar data\processed\train-<ESSAY-SET>.tsv output/baseline/<ESSAY-SET> 200 20`

Each run generated three files in its output directory:
- matrix.csv - confusion matrix
- metrics.csv - precision, recall, and F1 score
- performance.csv - (`ID`, `Score 1`, predicted score) triples. (with an admittedly bad filename)

| Essay Set | Precision | Recall | F1     | Quadratic Weighted Kappa |
| --------- | --------- | ------ | ------ | ------------------------ |
| 1         | 63.01%    | 55.14% | 58.81% | 68.76%                   |
| 2         | 55.08%    | 50.18% | 52.52% | 46.90%                   |
| 3         | 51.46%    | 52.83% | 52.13% |  6.10%                   |
| 4         | 78.22%    | 74.09% | 76.10% | 56.50%                   |
| 5         | 75.90%    | 82.56% | 79.09% | 43.37%                   |
| 6         | 79.98%    | 87.60% | 83.61% | 37.41%                   |
| 7         | 68.83%    | 64.51% | 66.60% | 44.22%                   |
| 8         | 62.57%    | 61.41% | 61.99% | 47.68%                   |
| 9         | 71.91%    | 65.76% | 68.70% | 64.55%                   |
| 10        | 78.59%    | 74.41% | 76.44% | 57.17%                   |
| Average   | 68.55%    | 66.85% | 67.60% | 47.27%                   |

As I suspected, the typical measure of precision, recall, and F1 have little to no correlation to the target measure of quadatic weighted kappa. This makes sense as the former three metrics are more appropriate in classification tasks where the target labels are categorical and have no intrinsic ordering. Whereas quadratic weighted kappa takes into account the ordering and distribution of labels in its final score.

The TF-IDF random forest pipeline seems to do best (surprisingly well, actually) on sets 1 and 9, worst on set 3, and posts middling performance on the remainder (with an average of ~47%). Taking a look at the confusion matrices for essay sets 1 and 9, it seems to me that the best performing essay sets must contain answers with "high value" words that are highly predictive of the answer being rated high or low. Conversely, the extremely poor performance of essay set 3 was probably because it predicted nearly everything as a 1 when only ~1/2 of the data was actually scored 1 (the remainder being split between 0 and 2). TF-IDF is clearly not a very good feature set for essay set 3. Taking a look at the text of the records quickly makes it evident that the content of the answers is highly regular with very consistent language throughout. This property makes it difficult for TF-IDF to generate any "interesting" values for words.

Possible Next Steps:
- Environment
  - Move all feature engineering, and maybe model training and testing, to Pythont to make it easier to interact with and interrogate the data.
- Preprocesing
  - Double the available data by using both scores. However, this brings up a philosophical and machine learning conundrum in the way that the problem is poised. Score 1 is the actual score awarded to the answer and thus the target label, Score 2 is the score given to measure/test the reliability of Score 1. By creating duplicate instances (by feature representation) that have the same score, we're not *really* adding any new signal to the system. We could argue that we're reinforcing the signal, and that could be borne out in the experiment's results. But, what happens when the duplicate instances have different scores? Now it would seem that we're adding noise to the system in the form of philosophically incorrect target labels. But, again, it could be argued that if the reliability score disagrees with the actual score, then the actual score is not as accurate as we take it to be. The aforementioned noise could actually be a signal that represents our lack of confidence in the actual score. In short, it's not at all clear that it would be appropriate to use Score 2 as stand-in target label, but it would make for an interesting experiment.
  - Develop heuristic to remove outlier answers that are clearly mistakenly truncated, yet have a non-zero score.
- Model
  - Use linear or other generalized regression. The task would seem to fit a regression model quite well given that the type of the target label is a numeric score. However, I'll have to be careful to scale all of the features, and possibly the output, so that the prediction is not outside of [0,3].
- Features
  - Number of words
  - Number of non-whitespace characters
  - Amount of punctuation
  - Word n-grams...
  - Essay set-specific features where appropriate (e.g., "key elements" from the rubric for essay 6)
