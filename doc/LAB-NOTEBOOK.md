# Lab Notebook

[Kaggle Competition page](https://www.kaggle.com/c/asap-sas)

## Data Investigation

I perused the data and noticed a few things immediately. There are many misspellings, some records appear to be mistakenly truncated (bad data), and the use of punctuation is all over the place. This suggests a few things that could be done to preprocess the data: spell correction, remove truncated records, and normalize or remove all punctuation.

The distribution of the data seems to be fairly even across the 10 questions, with no single question dominating the distribution. After separating the data set by the `Essay Set` column, it becomes clear that the answers use quite different language to answer each question, which makes sense as the questions don't share topics. However, this poses a few questions: do the answers share a (sub-)language? Do only certain scores overlap in terms of language (e.g., bad answers share a language across all sets, but good answers differ)? Can we capture the structure of the answer regardless of its language?

Other miscellaneous observations:
- the length of the answer seems to correlate with score, with longer answers performing better.
- some essay sets have highly regular language/vocabularies, others do not.

## Experiment 1 - Baseline

### Preprocessing the data

Using the observations above, I made a quick and dirty [Python script](../src/main/py/preprocess.py) to clean and normalize the text of the answers. Specifically,
- Strip all punctuation
- Normalize runs of whitespace to a single space
- Make all characters lowercase
After the above steps, 10 output files are generated, one containing each essay set's answers. Note that this process does not actually *remove* any records...yet.

### Random Forest with TF-IDF

I then created a Spark pipeline that generates 4,000 TF-IDF features and trains and tests a random forest classifier (200 trees, max depth of 20 per tree). I iteratively ran this class with each of the 10 separated essay sets and generated [10 sets of results](../output/baseline). The following command was used to launch each pipeline: `spark-submit --class collinm.asap.BaselineTfIdf --master local[7] build\libs\asap-0-fat.jar data\processed\train-<ESSAY-SET>.tsv output/baseline/<ESSAY-SET> 200 20`

Each run generated three files in its output directory:
- matrix.csv - confusion matrix
- metrics.csv - precision, recall, and F1 score
- performance.csv - `(ID, Score 1, predicted score)` triples. (with an admittedly bad filename)

| Essay Set | Precision | Recall | F1     | Quadratic Weighted Kappa |
| --------- | --------: | -----: | -----: | -----------------------: |
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

**Possible Next Steps**:
- Environment
  - Move all feature engineering, and maybe model training and testing, to Python to make it easier to interact with and interrogate the data.
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

## Experiment 1 - Baseline Redux

Uses the same preprocessed data as "Experiment 1 - Baseline".

### Random Forest with TF-IDF

I ported my Spark pipeline from the previous experiment into Python and re-ran the experiment with different (better informed) parameters. The Python pipeline used 2,000 TF-IDF features (ranked by highest TF-IDF scores in the training data) as input to a random forest of 200 trees each with a maximum depth of 30 nodes. This pipeline was run independently for each of the 10 essay sets. Each pipeline is saved after it is trained to allow for easier replication of the experimental results. The experiment is encoded in [`experiments/01-baseline.py`](../experiments/01-baseline.py).

Each pipeline generates a separate result in the [output folder](../output/py-baseline) with the following files:
- `cm.csv` - confusion matrix
- `pipeline.pkl` - pickled pipeline
- `qwk.txt` - contains quadratic weighted kappa metric

| Essay Set | Quadratic Weighted Kappa |
| --------- | -----------------------: |
| 1         | 76.86%                   |
| 2         | 52.24%                   |
| 3         | 13.29%                   |
| 4         | 61.20%                   |
| 5         | 45.67%                   |
| 6         | 63.39%                   |
| 7         | 52.03%                   |
| 8         | 48.86%                   |
| 9         | 63.31%                   |
| 10        | 69.76%                   |
| Average   | 54.66%                   |

This pipeline, though being very similar to the last one, performed quite a bit better: +7% in quadratic weighted kappa. Every essay set performed better than the prior experiment, except for set 9 which performed only slightly worse (-1%). Without digging into the data too much, I'd attribute these incraeses to the greater generality provided by a reduction in TF-IDF features and trees that consider a greater number of features (square root of the total features, instead of log2).

## Experiment 2 - Question/Source/Prompt Vocabulary

Uses the same preprocessed data as "Experiment 1 - Baseline".

### Random Forest with bag-of-words vector

I read through each essay questions source material, if present, and question prompt, and recorded words or similar conjugates that looked to be important to the answer. Each [list of keywords](../data/keywords) is 46-70 words long and includes words that I judged to be relevant to an answer, good or bad. Admittedly, by doing this by hand, I'm imposing my own bias on this feature, but hopefully it'll be a positive bias... Each list of keywords is only used with its associated essay set. It creates a N-length (N = number of keywords) vector of 1's and 0's, where a 1 represents the word's presence in the candidate answer and a 0 represents its absence. This vector represents the only features used for learning.

This pipeline can be run with a command like `python experiments\02-word-presence.py data\processed output\exp-02\rf-200t-100d 200 100`

Each pipeline generates a separate result in the [output folder](../output/exp-02) with the following files:
- `cm.csv` - confusion matrix
- `pipe.pkl` - pickled pipeline
- `qwk.txt` - contains quadratic weighted kappa metric
- `results.csv` - for each item in the test set, records the ID of the record, the gold standard score (score1), and the raw prediction

The following table lists the quadratic weighted kappa scores for each essay set for each set of parameters. Model parameters are recorded like "N trees (t), M max depth (d)" (bolded numbers represent the best performance for each essay set):

| Essay Set | 150t, 100d | 200t, 100d | 200t, 10d | 100t, 25d |
| --------- | ---------: | ---------: | --------: | --------: |
| 1         | 55.90%     | 57.25%     | **58.80%**| 58.13%    |
| 2         | 38.65%     | 37.66%     | **38.99%**| 34.72%    |
| 3         | **14.74%** | 13.52%     | 03.91%    | 13.87%    |
| 4         | 41.88%     | 40.93%     | 40.12%    | **42.07%**|
| 5         | 58.37%     | **60.44%** | 58.97%    | 49.99%    |
| 6         | 80.50%     | 80.41%     | **81.60%**| 79.27%    |
| 7         | 38.19%     | 38.19%     | **42.11%**| 34.00%    |
| 8         | 30.73%     | 30.65%     | **39.30%**| 32.38%    |
| 9         | 55.37%     | **57.82%** | 56.81%    | 56.11%    |
| 10        | 56.67%     | 56.67%     | 53.58%    | **58.16%**|
| Avg       | 47.10%     | 47.35%     | **47.42%**| 45.87%    |

Across the board, these hand-picked features perform worse than TF-IDF by ~7%. However, I suspect that misspellings in the candidate answer set are contributing to the poorer performance. Though this same complication might also affect TF-IDF, common misspellings have a chance of actually being represented by TF-IDF; whereas a bag-of-words vector will never capture misspellings unless they're explicitly accounted for.

A variety of model parameters did not make much of a difference, on average, as long as the values were kept in a reasonable range. Specific essay sets were affected by model parameters, suggesting that some models are overfitting their essay set and poorly generalizing to the test data.

The only two essay sets whose performance improved over the baseline sets 5 and 6, the "protein synthesis" and "cell membrane" questions, respectively. The reason for this improvement is most likely that these questions are the most sensitive to pure vocabulary features. The scoring rubric is entirely based around how many of the "key elements" the candidate answer covers. This abstracted presence-absence criteria is an excellent match for our bag-of-words features, in this case. We could probably still add or subtract some words from each list to eke out some additional signal and performance.

### Random Forest with bag-of-words vector and misc counting features

This pipeline leverages a random forest against the same features as the experiment above with the addition of four counting features: word count, unique word count, character count, and non-whitespace character count. All of these counting features are an attempt to model the signal that we get from the length of the candidate answer. Shorter answers are, on average, worse than longer answers; however, this is not always true. Occassionally, a student writes an answer that qualifies for the highest score in an amount of words that would otherwise incur the lowest score. Admittedly, these features are far from perfect, but they may realize some value when paired with the bag-of-words features.

This pipeline can be run with a command like `python experiments\02.1-word-presence-and-counts.py data\processed output\exp-02.1\rf-200t-100d 200 100`

| Essay Set | 200t, 100d |
| --------- | ---------: |
| 1         | 59.95%     |
| 2         | 35.11%     |
| 3         | 21.82%     |
| 4         | 53.29%     |
| 5         | 55.81%     |
| 6         | 81.17%     |
| 7         | 37.48%     |
| 8         | 39.60%     |
| 9         | 60.25%     |
| 10        | 64.21%     |
| Avg       | 50.87%     |

On the whole, performance increased ~3%. The performance of most of the essay sets was essentially flat or a little negative, except: 3, 4, 8, 9, 10. These sets gained 3-6%. Further investigation of the training data will be required to understand why this subset responded well while the others did not respond at all or responded poorly.
