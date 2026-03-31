# Classifying Disaster Tweets with BERT

During disasters, Twitter gets flooded with tweets. Some report injuries, others ask for help, others are just people expressing sympathy. Humanitarian orgs need to sort through all of that quickly. This project tries to automate that with NLP.

I fine-tuned BERT and BERTweet on the [HumAID dataset](https://crisisnlp.qcri.org/humaid_dataset) (~77K real disaster tweets across 10 categories) and compared them against a logistic regression baseline. I wanted to see if transformers are actually worth the compute cost over simpler models for this kind of task.

## The Three Models

**Logistic Regression (TF-IDF baseline).** TF-IDF vectors + linear classifier. Trains in 30 seconds, gets 72% accuracy.

**BERT (bert-base-uncased).** 110M-parameter transformer, pre-trained on Wikipedia and books. Fine-tuned for 4 epochs with weighted cross-entropy and early stopping. 76% accuracy, ~88 minutes to train.

**BERTweet (vinai/bertweet-base).** Same architecture as BERT but pre-trained on 850M tweets. I expected tweet-specific pre-training to help more than it did. It was better on some categories (tone, slang, RT patterns) but landed at 75.8% overall, basically the same as BERT.

## Results

| Model | Accuracy | Weighted F1 | Training Time |
|-------|----------|-------------|---------------|
| Logistic Regression | 71.96% | 0.721 | 30 seconds |
| BERT | 76.13% | 0.749 | ~88 minutes |
| BERTweet | 75.75% | 0.741 | ~91 minutes |

BERT gets about a 4-point accuracy boost over logistic regression, but at 180x the training cost. BERTweet didn't beat vanilla BERT. Per-class, BERTweet is better at tone-dependent categories ("sympathy" vs "reporting injuries") but worse on the catch-all classes.

The error analysis is worth looking at. Some tweets all three models get wrong, and reading them you can tell the labels are just ambiguous. "Not water, not electricity. This earthquake has hit us horrible. We can go through this together." Is that infrastructure damage or sympathy? Hard to say.

## Data Augmentation

The rarest class (`missing_or_found_people`) had ~200 training examples. I merged in tweets from [CrisisBench](https://crisisnlp.qcri.org/crisisbench/) (same research group, compatible labels) to go from 53K to 141K training tweets. Only the training set was augmented. Validation and test stay HumAID-only.

## Running It

Designed for **Google Colab with a GPU** (T4 or better). You need a [HuggingFace account](https://huggingface.co/) for the token.

1. Upload `BERT_HumAID_Classification.ipynb` to Colab
2. Set runtime to GPU (Runtime > Change runtime type > T4 GPU)
3. Run cells in order

Full training takes about 3 hours on a T4. There are checkpoint resume cells if the runtime disconnects.

## Dependencies

- `transformers`, `datasets` (HuggingFace)
- `torch` (comes with Colab)
- `scikit-learn`, `seaborn`, `matplotlib`, `pandas`

## Dataset

- **HumAID**: Alam et al., "HumAID: Human-Annotated Disaster Incidents Data from Twitter" (ICWSM 2021)
- **CrisisBench**: Alam et al., "CrisisBench: Benchmarking Crisis-related Social Media Text Classification" (2021)

Both loaded automatically from HuggingFace by the notebook.
