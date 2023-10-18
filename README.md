# Superhuman Fair Classification

This repository provides Python implementation for our paper [Superhuman Fairness](https://proceedings.mlr.press/v202/memarrast23a/memarrast23a.pdf) published at ICML 2023.

### Abstract

The fairness of machine learning-based decisions has become an increasingly important focus in the design of supervised machine learning methods. Most fairness approaches optimize a specified trade-off between performance measure(s) (e.g., accuracy, log loss, or AUC) and fairness metric(s) (e.g., demographic parity, equalized odds).
This begs the question: are the right performance fairness trade-offs being specified? We instead recast fair machine learning as an imitation learning
task by introducing superhuman fairness, which seeks to simultaneously outperform human decisions on multiple predictive performance and
fairness measures. We demonstrate the benefits of this approach given suboptimal decisions.


 
### Experiments

We use `main.py` to create demonstrations, train and test our model, and compare it with other baselines.

First, we need to create a set of reference decisions (demonstrations) from training data. `-n True` adds noise to the protected attribute and predicted label of the demonstrations.

```console
$ python main.py -t prepare-demos -n [True|False] -d [Adult|COMPAS] 
```

Then, we train our model with a set of performance/fairness metrics. The goal is to outperform demonstrations in all those metrics with maximal frequency. We have implemented following metrics in our code:

* **Prediction Error** (*inacc*)
* **Demographic Parity** (*dp*)
* **Equalized Odds** (*eqodds*)
* **False Negative Rate** (*fnr* or *eqopp*)
* **False Positive Rate** (*fpr*)
* **Predictive Rate Partiy** (*prp*)
* **Positive Predictive Value** (*ppv*)
* **Negative Predictive Value** (*npv*)
* **Balanced Error Rate Difference** (*error_rate_diff*)

Note that `get_metrics_df` function in `util.py` can be extended to include additional metrics.

To train with three conflicting fairness measures (dp, eqodds, prp) and also inaccuracy, we use the following command:

```console
$ python main.py -t train -n [True|False] -d [Adult|COMPAS] -f inacc dp eqodds prp
```

To test our model and compare it with other baselines, we run:

```console
$ python main.py -t test -n [True|False] -d [Adult|COMPAS] -f inacc dp eqodds prp
```

To plot the results created by the test command, we use `plot.py` file:

```console
$ python plot.py -t test -n [True|False] -d [Adult|COMPAS] -f inacc dp eqodds prp
```
To reproduce the plots related to the performance of our model with varying degrees of noise:

```console
$ python plot.py -t noise-test -n True -d [Adult|COMPAS] -f inacc dp eqodds prp
```

python main.py -t train -n False -m "NN" -d Adult -f inacc dp eqodds prp
python main.py -t train -n False -d Adult -f inacc dp eqodds prp