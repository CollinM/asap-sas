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

This experiment uses the same preprocessing steps as "Experiment 1 - Baseline", but I used the `train_rel_2.tsv` file instead of the original `train.tsv` file. The former is a corrected derivative of the latter, so these files should be very similar. I also used the public leaderboard data to test with, instead of doing a train-test split on the training data.

### Random Forest with TF-IDF

I ported my Spark pipeline from the previous experiment into Python and re-ran the experiment with different (better informed) parameters. The Python pipeline used 2,000 TF-IDF features (ranked by highest TF-IDF scores in the training data) as input to a random forest of 200 trees each with a maximum depth of 30 nodes. This pipeline was run independently for each of the 10 essay sets. Each pipeline is saved after it is trained to allow for easier replication of the experimental results.

The experiment code is [here](../experiments/01-tfidf.py).

Each pipeline generates a separate result in the [output folder](../output/exp-01) with the following files:
- `cm.csv` - confusion matrix
- `pipeline.pkl` - pickled pipeline
- `qwk.txt` - contains quadratic weighted kappa metric

| Essay Set | Quadratic Weighted Kappa |
| --------- | -----------------------: |
| 1         | 67.08%                   |
| 2         | 53.00%                   |
| 3         | 45.70%                   |
| 4         | 53.51%                   |
| 5         | 50.16%                   |
| 6         | 49.11%                   |
| 7         | 49.78%                   |
| 8         | 49.42%                   |
| 9         | 76.36%                   |
| 10        | 64.83%                   |
| Average   | 55.89%                   |

As noted above, this pipeline was trained on slightly different training data and tested on completely different test data. This pipeline, though being very similar to the last one, performed quite a bit better: +8% in quadratic weighted kappa. Most essay sets performed better than the prior experiment, except for a few which performed only slightly worse. Without digging into the data too much, I'd attribute these incraeses to the greater generality provided by a reduction in TF-IDF features and trees that consider a greater number of features (square root of the total features, instead of log2).

## Experiment 2 - Question/Source/Prompt Keywords

Uses the same data as "Experiment 1 - Baseline Redux".

### Random Forest with keywords vector

I read through each essay questions source material, if present, and question prompt, and recorded words or similar conjugates that looked to be important to the answer. Each [list of keywords](../data/keywords) is 46-70 words long and includes words that I judged to be relevant to an answer, good or bad. Admittedly, by doing this by hand, I'm imposing my own bias on this feature, but hopefully it'll be a positive bias... Each list of keywords is only used with its associated essay set. It creates a N-length (N = number of keywords) vector of 1's and 0's, where a 1 represents the word's presence in the candidate answer and a 0 represents its absence. This vector represents the only features used for learning.

The experiment code is [here](../experiments/02-word-presence.py).

Each pipeline generates a separate result in the [output folder](../output/exp-02) with the following files:
- `cm.csv` - confusion matrix
- `pipe.pkl` - pickled pipeline
- `qwk.txt` - contains quadratic weighted kappa metric
- `results.csv` - for each item in the test set, records the ID of the record, the gold standard score (score1), and the raw prediction

The following table lists the quadratic weighted kappa scores for each essay set for each set of parameters. Model parameters are recorded like "N trees (t), M max depth (d)" (bolded numbers represent the best performance for each essay set):

| Essay Set | 150t, 100d | 200t, 100d | 200t, 10d | 100t, 25d |
| --------- | ---------: | ---------: | --------: | --------: |
| 1         | 53.00%     | 52.13%     | **56.42%**| 51.57%    |
| 2         | **44.84%** | 37.69%     | 41.00%    | 40.65%    |
| 3         | **44.03%** | 42.45%     | 41.48%    | 41.60%    |
| 4         | **43.15%** | 41.60%     | 40.86%    | 40.65%    |
| 5         | **64.86%** | 63.16%     | 62.48%    | 64.11%    |
| 6         | 71.62%     | 72.58%     | 68.67%    | **72.73%**|
| 7         | **32.97%** | 31.01%     | 28.19%    | 32.68%    |
| 8         | 36.14%     | 34.87%     | **42.86%**| 35.42%    |
| 9         | 59.15%     | 60.45%     | **61.26%**| 59.83%    |
| 10        | **54.00%** | 52.42%     | 53.56%    | 52.81%    |
| Avg       | **50.38%** | 48.84%     | 49.68%    | 49.21%    |


Across the board, these hand-picked features perform worse than TF-IDF by ~5%. However, I suspect that misspellings in the candidate answer set are contributing to the poorer performance. Though this same complication might also affect TF-IDF, common misspellings have a chance of actually being represented by TF-IDF; whereas a keywrods vector will never capture misspellings unless they're explicitly accounted for.

A variety of model parameters did not make much of a difference, on average, as long as the values were kept in a reasonable range. Specific essay sets were affected by model parameters, suggesting that some models are overfitting their essay set and poorly generalizing to the test data. The majority of the essay sets performed best, if only slightly, with 150 trees at a maximum depth of 100 nodes.

The only two essay sets whose performance improved over the baseline sets 5 and 6, the "protein synthesis" and "cell membrane" questions, respectively. The reason for this improvement is most likely that these questions are the most sensitive to pure vocabulary features. The scoring rubric is entirely based around how many of the "key elements" the candidate answer covers. This abstracted presence-absence criteria is an excellent match for our keywords features, in this case. We could probably still add or subtract some words from each list to eke out some additional signal and performance.

### Random Forest with keywords vector and misc counting features

This pipeline leverages a random forest against the same features as the experiment above with the addition of four counting features: word count, unique word count, character count, and non-whitespace character count. All of these counting features are an attempt to model the signal that we get from the length of the candidate answer. Shorter answers are, on average, worse than longer answers; however, this is not always true. Occassionally, a student writes an answer that qualifies for the highest score in an amount of words that would otherwise incur the lowest score. Admittedly, these features are far from perfect, but they may realize some value when paired with the keywords features.

The experiment code is [here](../experiments/02.1-word-presence-and-counts.py).

| Essay Set | 150t, 100d |
| --------- | ---------: |
| 1         | 57.93%     |
| 2         | 46.65%     |
| 3         | 53.83%     |
| 4         | 54.16%     |
| 5         | 67.34%     |
| 6         | 68.40%     |
| 7         | 34.71%     |
| 8         | 41.04%     |
| 9         | 65.48%     |
| 10        | 58.76%     |
| Avg       | 54.83%     |

On the whole, performance increased ~4%. The performance of most of the essay sets was better, sometimes *much* better, expect for 6. The four counting features would seem to carry a decent amount of signal. These features might be good candidates for input to joint features.

## Experiment 3 - Bag of Words

Uses the same data as "Experiment 1 - Baseline Redux".

### Random Forest with Bag of Words Vector

This experiment is an attempt to get closer to the bag of words baseline posted on the Kaggle leaderboard. After a bit of looking, I found the code that generated the baseline [here](https://github.com/benhamner/ASAP-SAS/blob/master/Benchmarks/bag_of_words_benchmark.py). That code uses sklearn's random forest regressor to predict real value scores, while I'm using the random forest classifier to predict the score class. Because of this, I don't expect my results to be exactly the same, but they should be relatively close. Similarly, this is an opportunity to see how sensitive the results are to the size of the bag of words.

I varied the minimum number of word occurrences necessary to be included in the bag (1 = all words) and fed the bag of words vector as the only feature to a 150 tree, 100 node maximum depth random forest classifier.

The experiment code is [here](../experiments/03-bow.py).

Each pipeline generates a separate result in the [output folder](../output/exp-03) with the following files:
- `cm.csv` - confusion matrix
- `pipe.pkl` - pickled pipeline
- `qwk.txt` - contains quadratic weighted kappa metric
- `results.csv` - for each item in the test set, records the ID of the record, the gold standard score (score1), and the raw prediction

| Essay Set | Min = 1 | Min = 3 | Min = 5 | Min = 10 | Min = 25 | Min = 50 | Min = 100 |
| --------- | ------: | ------: | ------: | -------: | -------: | -------: | --------: |
| 1         | 67.34%  | 69.91%  | 68.07%  | 69.63%   | 69.86%   | 69.36%   | 68.93%    |
| 2         | 51.52%  | 60.11%  | 55.15%  | 60.35%   | 58.18%   | 62.39%   | 61.37%    |
| 3         | 47.70%  | 48.91%  | 54.38%  | 50.68%   | 54.17%   | 57.10%   | 55.24%    |
| 4         | 51.45%  | 52.23%  | 53.89%  | 56.67%   | 56.31%   | 51.21%   | 47.17%    |
| 5         | 48.94%  | 55.07%  | 55.19%  | 59.75%   | 60.27%   | 62.72%   | 61.28%    |
| 6         | 41.85%  | 53.72%  | 60.40%  | 59.37%   | 61.99%   | 65.77%   | 61.90%    |
| 7         | 50.53%  | 50.08%  | 53.19%  | 50.43%   | 51.37%   | 49.21%   | 50.70%    |
| 8         | 48.21%  | 49.30%  | 50.65%  | 49.01%   | 48.13%   | 49.54%   | 45.09%    |
| 9         | 73.94%  | 75.55%  | 77.34%  | 77.64%   | 77.16%   | 76.86%   | 75.52%    |
| 10        | 64.37%  | 65.01%  | 67.53%  | 66.71%   | 68.70%   | 66.81%   | 65.74%    |
| Avg       | 54.59%  | 57.99%  | 59.58%  | 60.02%   | 60.61%   | 61.10%   | 59.29%    |

The best bag of words model performed ~5% better than my best model prior to this (TF-IDF at ~55%). The model performed substantially better with a smaller bag of words than with a larger one. This makes sense from a complexity standpoint, but it's a little surprising that the best performance was gained by only including words that occurred 50+ times in the data. Especially considering all of the unique mispellings and the like that occur in the data. Excluding more words than that degraded performance though. That being said, the best model still doesn't perform as well as Kaggle's bag of words baseline. I'll need to try some regression models next.

Simplicity really is the ultimate sophistication...

## Experiment 4 - Regression Models

Uses the same data as "Experiment 1 - Baseline Redux".

### Ridge Regression vs Logistic Regression vs ElasticNet SVM

This experiment is to compare regression classifiers: ridge regression, logistic regression, and elastic net SVM. These three models were chosen due to their presence in some of the winning systems and their novelty to the user. Though each is fundamentally a regression algorithm, I've used sklearn's classifier wrappers for simplicity.

The experiment code is [here](../experiments/04-regression-classifiers.py).

Each pipeline generates a separate result in the [output folder](../output/exp-04) with the following files:
- `cm.csv` - confusion matrix
- `pipe.pkl` - pickled pipeline
- `qwk.txt` - contains quadratic weighted kappa metric
- `results.csv` - for each item in the test set, records the ID of the record, the gold standard score (score1), and the raw prediction

| Essay Set | Ridge, Min=25 | Ridge, Min=50 | LR, Min=25 | LR, Min=50 | SVM, Min=25 | SVM, Min=50 |
| --------: | ------------: | ------------: | ---------: | ---------: | ----------: | ----------: |
| 1         | 67.45%        | 68.48%        | 71.90%     | 70.32%     | 59.48%      | 58.00%      |
| 2         | 59.48%        | 58.47%        | 64.80%     | 65.57%     | 60.56%      | 50.83%      |
| 3         | 52.52%        | 55.40%        | 61.14%     | 60.70%     | 56.92%      | 54.36%      |
| 4         | 60.25%        | 60.51%        | 65.25%     | 65.25%     | 61.87%      | 59.15%      |
| 5         | 60.46%        | 62.10%        | 73.18%     | 72.00%     | 59.95%      | 69.02%      |
| 6         | 71.93%        | 63.47%        | 71.76%     | 71.06%     | 70.51%      | 71.85%      |
| 7         | 52.93%        | 45.01%        | 51.46%     | 44.82%     | 49.49%      | 39.29%      |
| 8         | 46.95%        | 46.90%        | 45.94%     | 46.40%     | 42.94%      | 47.41%      |
| 9         | 67.93%        | 68.54%        | 73.78%     | 75.12%     | 64.86%      | 64.24%      |
| 10        | 65.26%        | 64.59%        | 67.30%     | 66.87%     | 56.23%      | 67.04%      |
| Avg       | 60.51%        | 59.35%        | 64.65%     | 63.81%     | 58.28%      | 58.12%      |

Once again, relative simplicity reigns supreme. The best classifier was logistc regression using a bag of words with a minimum occurrence of 25. It's further worth acknowledging that this performed ~4% better than the random forest! It even performed slightly better than the Kaggle bag of words baseline (64.55%). This result indicates that with bag of words, the majority of the data is linearly separable. Of course, there's still the minority of the data that probably isn't.

## Experimetn 5 - LSTM

Uses the same data as "Experiment 1 - Baseline Redux".

I setup [Theano](https://github.com/Theano/Theano) and [keras](https://github.com/fchollet/keras) on a g2.2xlarge AWS instance ([this](http://markus.com/install-theano-on-aws/) was very helpful in accomplishing this).

### Single layer LSTM

This experiment is an attempt to leverage near-state-of-the-art recurrent neural network architectures to perform text classification. However, LSTM's pose a variety of problems for text classfication. First, they're expensive to train, though this is largely miticaged by access to a CUDA GPU. The AWS g2.2xlarge instance provides a Nvidia 970 GTX with 4 GB of memory. While this is not the beefiest card available, it's more than sufficient for my experiments. Second, LSTMs, like all neural networks, expect constant length inputs, so the input data needs to be sliced up into chunks that can be fed to the neural network in batches. Third, neural networks usually accept simplified features, in this case, bag of words. Fourth, the input slicing problem applies to both the training and testing data. This means that a test record might be represnted by N chunks each with a different score/classification. How do we combine the scores into a single score? I don't have a good answer for this, but I do have a heuristic: Take the most common score for a given record, and in case of a tie choose the smaller score. I also tried taking the rounded average and the performance was similar. Fifth, keras provides a number of objective loss functions, but none of them are quadratic weighted kappa. So, I used "categorical cross-entropy" which seemd to be a safe default choice after doing some research. I attempted to code quadratic weighted kappa myself using Theano's symbolic math libraries (a requirement) and it proved to be frutratingly difficult. The code works, but the input that it receives is not well documented and I'm limited to Theano's tensor functions for the majority of the logic which makes the whole thing an exercise in hacking cleverness...

For my first foray into LSTMs, I built a simple architecture consisting of a LSTM layer and a dense output layer. See `asap.core.ml.LSTM_Arch1` for more details. This model, combined with the bag of words features, provides 6 parameters to explore:
1. Bag of words minimum count - minimum number of occurrences required for a word to be included in the bag
2. Chunk length - the length of the BOW input chunks fed to the network
3. Chunk step size - the number of words to increment between taking new chunks of a document (only operates on in-bag words)
4. LSTM size - Number of outputs for the LSTM layer
5. Batch size - Size of the training batches
6. Epochs - Number of training epochs

1 controls the size of the input vector and the size of the raw record inputs. 2 and 3 control the amount of data fed into the network for each record, as the network sees it, and the total quantity of data. 4 controls model complexity. 5 and 6 control under-/over-fitting and training speed. Suffice to say that I was building my intuition about these parameters and their good and bad values as I was performing experiments.

Result files are similar to prior experience, except the pipelines/models are not pickled to due to greater complexity and the integration work required. Furthermore, the pickled models from prior experiments did not prove to be especially useful after the results were obtained. Because these models are much more expensive to train (20-30 minutes per experiment, instead of <2 minutes), I have run fewer of them and with greater parameter variation in order to explore more of the parameter space in a limited time. The following experiments were not necessarily performed in the order presented. Some were run in parallel.

For the experiments in the following table, chunk length = 10, chunk step size = 5, LSTM size = 512.

| Essay Set | BOW 10 (batch 64, epoch 35) | BOW 25 (batch 32, epoch 20) | BOW 50 (batch 32, epoch 20) |
| --------: | --------------------------: | --------------------------: | --------------------------: |
| 1   | 52.32% | 45.99% | 48.69% |
| 2   | 41.02% | 39.44% | 40.66% |
| 3   | 40.37% | 32.71% | 32.16% |
| 4   | 42.12% | 30.62% | 38.01% |
| 5   | 32.84% | 54.50% | 53.09% |
| 6   | 55.38% | 50.70% | 52.39% |
| 7   | 36.57% | 39.68% | 27.05% |
| 8   | 36.64% | 32.24% | 28.66% |
| 9   | 50.30% | 42.10% | 44.77% |
| 10  | 49.41% | 49.31% | 54.26% |
| Avg | 43.70% | 41.73% | 41.97% |

Larger bag of words have a positive impact on performance, though they are slower to train.

For the experiments in the following table, BOW min = 25, chunk = 10, LSTM size = 512, batch = 32.

| Essay Set | chunk step size 3 (epoch 20) | chunk step size 8 (epoch 15) |
| 1   | 55.68% | 47.31% |
| 2   | 43.42% | 33.02% |
| 3   | 30.22% | 24.18% |
| 4   | 30.20% | 20.33% |
| 5   | 48.77% | 43.60% |
| 6   | 45.91% | 54.45% |
| 7   | 37.51% | 31.35% |
| 8   | 35.27% | 35.22% |
| 9   | 39.46% | 52.56% |
| 10  | 41.38% | 47.19% |
| Avg | 40.78% | 38.92% |

Greater step size made performance worse. This makes sense as greater step sizes means fewer chunks being fed into the network and less overlap between chunks; therefore, less overall data going through the network. Of course, 5 fewer epochs could have contributed as well, but the accuracy measure done at the end of every epoch generally shows that improvement in performance becomes smaller and smaller for every epoch past 15. Although, prior experiments suggest that more epochs improve performance on some essay sets.

For the experiments in the following table, BOW min = 25, chunk size = 10, chunk step size = 5, batch = 32, epoch = 20.

| Essay Set | LSTM size 256 | LSTM size 128 |
| 1   | 50.72% | 52.06% |
| 2   | 43.76% | 42.74% |
| 3   | 32.73% | 31.21% |
| 4   | 31.28% | 33.53% |
| 5   | 52.00% | 53.47% |
| 6   | 61.39% | 53.93% |
| 7   | 38.72% | 40.02% |
| 8   | 33.20% | 31.80% |
| 9   | 49.60% | 49.22% |
| 10  | 46.15% | 53.67% |
| Avg | 43.95% | 44.16% |

Overall, decreasing the number of outputs of the LSTM layer had a positive impact on performance: +4%. However, the difference between 256 and 128 outputs is essentially negligible. They perform very similar on most essay sets, and the disimilar sets are mostly trading performance increases and decreases.

For the experiments in the following table, chunk step size = 1, LSTM size = 128, batch = 32.

| Essay Set | BOW 25, chunk size 5, epoch 20 | BOW 10, chunk size 15, epoch 25 |
| --------: | -----------------------------: | ------------------------------: |
| 1   | 55.57% | 59.51% |
| 2   | 38.42% | 40.67% |
| 3   | 08.29% | 34.74% |
| 4   | 28.42% | 35.45% |
| 5   | 38.93% | 53.64% |
| 6   | 44.89% | 55.83% |
| 7   | 39.33% | 
| 8   | 20.90% | 
| 9   | 41.15% | 
| 10  | 41.30% | 
| Avg | 35.72% | 

50+ seconds per epoch, every model takes ~20 minutes...

Thoughts:
- scoring heuristic is not great... the network may score some chunks low, because they look like bad answers, and then score a few chunks high, because they look like good answers, and the bad may outweigh the good...
- Word embeddings would provie much richer input data, though the data set may not be large enough to generate a good embedding, especially if it's partitioned by essay set.
- Different essay sets require models with different parameters. One size does _not_ fit all here.
- Additional research on how others are using recurrent neural networks for text classification led me to the techniques of stacked LSTMs, character-wise RNNs, and convolutional LSTMs. All of these approaches have their pros and cons (all are more expensive to train), but they are all closer to the state of the art in RNN text classification. Unfortunately, I did not have time to explore these techniques and architectures.
