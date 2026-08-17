# MSCS 634 — Project Deliverable 3

## Classification, Clustering, and Pattern Mining
### Factors Associated with Five-Year Home Price Growth in California

**Name:** Chun-Shuo Sheu
**Course:** MSCS 634 — Big Data and Data Mining
**Deliverable:** 3 of 4 — Classification, Clustering, and Pattern Mining

---

## The three questions

| Technique | Question |
|---|---|
| **Classification** | Can we *identify in advance* which ZIP codes will be high-growth? |
| **Clustering** | What *types* of housing market does California contain? |
| **Association rules** | Which *combinations* of conditions co-occur with high growth? |

All three are applied to the same dataset built in Deliverable 1 and modeled in Deliverable 2 —
`ca_zip_housing_panel_clean.csv`, 7,487 rows × 28 columns, one row per California ZIP code per base
year (2016–2020). No new dataset was introduced.

## Reproducing

```bash
pip install -r requirements.txt

# D1's cleaned panel must be present (copied in, or rebuilt via D1's scripts):
#   data/processed/ca_zip_housing_panel_clean.csv

python scripts/make_deliverable_3_notebook.py
jupyter nbconvert --execute --inplace notebooks/Project_Deliverable_3.ipynb
```

The notebook is self-contained: it imports nothing from `scripts/`, and it writes all 13 figures into
`figures/`. Every model is seeded with `random_state=42`, so the run is reproducible.

## Summary

In the first Deliverable, I built a dataset that combined Zillow home prices, Census population,
income, rent, unemployment, and education data, along with mortgage rates, and produced 7,487
observations covering California ZIP Codes across five base years. In the second Deliverable, I used
regression models to predict five-year home-price growth and found that the models could not predict
the absolute level of growth, because the average five-year growth rate fell from approximately 47.96%
in the training years to 24.31% in the test year. After the target was redefined as growth relative to
the statewide median in each year, the models produced positive test R² values of approximately 0.12.
The conclusion of the second Deliverable was that ZIP Code-level features are better suited for
comparing areas against each other than for forecasting the overall market.

The third Deliverable applies three additional data-mining techniques to the same dataset. The
classification task asks whether high-growth ZIP Codes can be identified in advance, the clustering
task asks what types of housing markets exist in California, and the association rule task asks which
combinations of conditions occur together with high growth.

For classification, the label was defined as the top quartile of five-year growth, with the threshold
computed from the training years only. Computing the threshold from the full dataset would allow the
outcomes of the test rows to help define the boundary between high growth and normal growth, which is
a form of leakage that is easy to overlook. The training-only threshold was 58.68%, while the
full-dataset threshold would have been 54.08%, and the difference would have changed the label of 572
rows. The temporal split from the second Deliverable was reused, with 2016 to 2018 as training data,
2019 as validation data, and 2020 as test data. Cross-validation used GroupKFold grouped by base year
so that rows from the same ZIP Code in adjacent years would not appear in both the training and
validation folds. Six models were fitted, including a majority-class baseline, a stratified baseline,
logistic regression, a decision tree, an untuned random forest, and a random forest tuned with
GridSearchCV over 36 parameter combinations scored on F1 rather than accuracy.

The most important result of the classification section is that the proportion of high-growth ZIP
Codes falls from 42.6% in 2016 to only 2.0% in 2020. This is the same regime shift observed in the
second Deliverable, appearing in a different form. Because the threshold is fixed while the
distribution of growth moves lower each year, the positive class almost disappears in the test year,
leaving only 30 positive cases out of 1,528 rows. As a result, the majority-class baseline reaches
98.04% accuracy on the test year simply by predicting that no ZIP Code will ever be high growth. For
this reason, accuracy is not used as the headline metric anywhere in this analysis. The tuned random
forest achieved a test F1 score of only 0.089, but its test ROC-AUC was 0.793, which is higher than
its validation ROC-AUC of 0.759. These two results are not contradictory. The average predicted
probability on the test year was 0.311 while the actual positive rate was 0.020, so the model's
probabilities were poorly calibrated by a factor of about sixteen. At the same time, the top 30 ranked
ZIP Codes contained 3.40 times their expected share of true positives, the top decile contained 2.66
times its share, and the lowest five deciles contained none of the 30 positive cases at all. In other
words, the model retained its ability to rank ZIP Codes but lost its ability to decide a cutoff.
Adjusting the decision threshold across its whole range never raised the test F1 above approximately
0.13, so the problem was not a poorly chosen threshold but the near-disappearance of the class itself.

Following the recommendation made at the end of the second Deliverable, the label was then redefined
as the top quartile within each base year. This keeps the positive rate at exactly 25% in every year
and changes the question from whether a ZIP Code will grow more than 58.68% to whether it will finish
in the top quarter of California ZIP Codes for its own period. With the same features, the same rows,
and the same folds, the test F1 of the tuned random forest rose from 0.089 to 0.406, and the test F1
of logistic regression rose from 0.023 to 0.440. Test ROC-AUC declined slightly to 0.719, because
separating the boundary of the top quartile is a harder ranking problem than isolating the extreme
tail of the distribution. Precision on the test year was 0.491, which means that about half of the ZIP
Codes flagged by the model were incorrect. Fixing the label removed the level shift, but it did not
make five-year housing outcomes easy to predict. For the grid search, the parameter that mattered most
was class_weight, since every one of the top combinations used the balanced setting while the three
worst combinations did not, and this single choice raised cross-validated F1 from approximately 0.33
to approximately 0.46.

For clustering, the goal was market segmentation rather than prediction, so the outcome variable was
excluded from the feature set. Seven variables describing the character of each market were used, and
only the 2020 slice was clustered so that each ZIP Code appears exactly once. Clustering the full
panel would place five nearly identical copies of every ZIP Code into the same space and distort the
geometry. Standardization was mandatory because home values are on the order of one million while
unemployment rates are on the order of ten, and without scaling the clustering would essentially sort
ZIP Codes by price. A small number of ZIP Codes reported population growth above 500%, which reflects
Census boundary changes rather than real housing markets, and these values were winsorized at the 1st
and 99th percentiles because a single extreme point can capture an entire cluster.

The elbow method suggested k equal to 3, while the silhouette score reached its maximum at k equal to
2. However, k equal to 2 only separates expensive from inexpensive markets and provides no useful
segmentation, and the silhouette scores for k from 3 to 6 differed by only 0.0014, so the silhouette
could not distinguish between them. The tie-break criterion was therefore stated in advance: choose
the smallest k at which clusters are distinguished by different features rather than by a single
affluence gradient. At k equal to 3 the clusters form a ladder in which every feature increases
together, while at k equal to 5 two clusters break this pattern. The average silhouette score of
approximately 0.23 is low, which indicates that California ZIP Codes lie along a continuum rather than
forming clearly separated groups, and this limitation is stated in the notebook.

The five segments were named from their profiles rather than left as numbered clusters. They are Elite
Coastal and Tech Markets, Affluent Growing Suburbs, Mid-Market Suburban California, Price-Detached
Urban Markets, and Affordable Working-Class Markets. The segments were then compared against the
growth outcome, which was never used during fitting. Affordable Working-Class Markets grew 5.99
percentage points faster than their own year's median and finished in the top quartile 39.4% of the
time, while Price-Detached Urban Markets grew 15.76 percentage points slower than their own year's
median and finished in the top quartile only 12.4% of the time. When every base year was assigned to
the clusters fitted on the 2020 data, the Price-Detached segment ranked last in all five five-year
windows, and one-way ANOVA across segments produced F equal to 27.49 with a p-value of 4.6 × 10⁻²².
The Price-Detached segment consists of 129 ZIP Codes, largely in central Los Angeles, where median
home values are approximately 1.06 million dollars while median household income is approximately
67,750 dollars. It is important to note that the price-to-income ratio is one of the clustering
features, so this result re-derives the affordability finding from the second Deliverable using an
unsupervised method rather than confirming it independently. What clustering adds is that the effect
is concentrated in a specific and identifiable type of market rather than spread smoothly across all
ZIP Codes.

For association rule mining, eight variables were divided into terciles and each ZIP-year was treated
as one transaction. The first run followed the original plan exactly and produced only four rules with
a growth consequent, and one of those rules included a mortgage-rate condition. Because the mortgage
rate is national and constant within each year, its terciles correspond exactly to base years, so that
rule was effectively conditioning on the year rather than on any market characteristic. The global
growth terciles were also close to a year indicator, since 947 of 1,452 rows in 2016 fell into the
high tercile while only 68 of 1,528 rows in 2020 did. The second run therefore removed the mortgage
rate and computed the growth terciles within each base year, which produced 26 rules with lift values
between 1.52 and 1.87 and support between 5.1% and 11.8% of the panel. Seventeen rules predict high
growth and nine predict low growth, and no rule predicts the middle tercile, which suggests that
combinations of extreme conditions identify the tails of the distribution but not average markets. The
confidence threshold was relaxed from 0.60 to 0.50 for this run, and since each growth tercile
contains exactly one third of the observations, a confidence of 0.50 already corresponds to a lift of
1.5.

The price-to-income ratio appears in all nine of the low-growth rules and in seven of the seventeen
high-growth rules, with opposite direction on each side. The two sides are not symmetric, however. A
high price-to-income ratio alone is the strongest single condition in the entire item set, with a lift
of 1.414 across 15.7% of the panel, so unaffordability predicts underperformance without needing to be
combined with anything else. A low price-to-income ratio alone reaches a lift of only 1.225 and must
be combined with low income and high unemployment before the lift reaches 1.87. In practical terms,
being unaffordable was sufficient on its own to predict below-median growth, while being affordable
was not sufficient on its own to predict above-median growth.

The most important overall result of this Deliverable is that three techniques based on completely
different assumptions all identified affordability as the central variable. Logistic regression
assigned the largest stable negative coefficient to the price-to-rent ratio, with an odds ratio of
0.21 per one standard deviation. K-Means, which never saw the outcome variable, isolated the segment
with an extreme price-to-income ratio and below-average income, and that segment finished last in
every one of the five windows. Apriori, which only counts co-occurrences among discretized bins,
placed the price-to-income ratio on both tails of the rule set. A distance-based method, a
likelihood-based method, and a frequency-based method agreeing on the same variable is stronger
evidence than any one of them alone.

In terms of practical relevance, a homebuyer can use the finding that ZIP Codes where homes cost
approximately 8.8 times local median income or more have historically been below-median appreciation
markets, although this analysis says nothing about schools, commuting, safety, or other reasons people
choose where to live. An investor can use the conjunction of low price-to-income ratio, low income,
and high unemployment as a screening rule with 62.5% confidence over 491 ZIP-years, while recognizing
that more than a third of the ZIP Codes passing that screen still failed to beat the median. A city
planner can use the pattern that jurisdictions growing in population while already unaffordable were
the most likely to underperform, which is a statement about the gap between housing costs and local
earnings rather than about investment returns.

Several challenges arose during this Deliverable. The overlapping five-year outcome windows mean that
the 7,487 rows are not independent observations, so a temporal split and GroupKFold were used
everywhere and plain KFold appears nowhere. The regime shift between base years caused the positive
class to collapse in the test year, which was reported and diagnosed rather than hidden. Class
imbalance made accuracy misleading, so two baselines were included specifically so that the reader can
see the majority classifier score 0.9804. The mortgage rate acts as a base-year index in disguise, and
its effect on the first association rule run was demonstrated with a cross-tabulation rather than
simply asserted. Extreme population growth values distorted the clustering, and the unclipped result
at k equal to 4 produced a cluster containing only two ZIP Codes. The elbow and silhouette criteria
disagreed, and the disagreement was reported along with the tie-break rule instead of being smoothed
over. Finally, columns derived from the target, such as the outlier flags created in Deliverable 1,
were explicitly excluded and asserted against, because they look like ordinary data-quality columns
while actually containing part of the answer.

All results reported here are associations measured in observational data covering a single five-year
period that includes one interest-rate cycle and one pandemic housing boom. ZIP Codes with high
price-to-income ratios subsequently grew more slowly than the median for their year. This analysis
does not claim that unaffordability caused slower growth, and no classification, clustering, or rule
mining method applied to this dataset could establish that.

## Figures

| File | Content |
|---|---|
| `fig01_label_prevalence.png` | Positive-class rate collapsing 42.6% → 2.0%, and the distribution sliding under a fixed threshold |
| `fig02_model_comparison_absolute.png` | Accuracy / F1 / ROC-AUC across six models, absolute label |
| `fig03_confusion_roc_absolute.png` | Confusion matrices and ROC curves, absolute label |
| `fig04_ranking_vs_calibration.png` | Probability distributions, decile lift, threshold sweep |
| `fig05_model_comparison_relative.png` | Same comparison under the within-year label |
| `fig06_confusion_roc_relative.png` | Confusion matrices and ROC curves, within-year label |
| `fig07_feature_importance.png` | Logistic-regression coefficients and Random Forest importances |
| `fig08_decision_tree.png` | Depth-3 decision tree |
| `fig09_elbow_silhouette.png` | Elbow, silhouette plateau, and seed stability |
| `fig10_cluster_profiles.png` | Cluster profile heatmap in standard-deviation units |
| `fig11_pca_clusters.png` | PCA projection with loadings, and the same space coloured by outcome |
| `fig12_cluster_growth.png` | Segment growth by base year, relative growth, top-quartile rate |
| `fig13_association_rules.png` | Top rules by lift and the support–confidence scatter |
