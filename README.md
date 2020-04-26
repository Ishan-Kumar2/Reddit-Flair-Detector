# Reddit-Flair-Detector

This model aims to detect the flair of a Reddit post from '/india' subreddit. The model classifies the flair of a post given its URL into Coronavirus, Nonpolitical, Political and others. The model can also be hosted as a web service.

## Installation
First clone the repo to the local device using
``` bash
git clone https://github.com/Ishan-Kumar2/Reddit-Flair-Detector.git
```
This project requires
* [spacy](https://spacy.io/)
* [Flask](https://flask.palletsprojects.com/)
* [pandas](https://pandas.pydata.org/)
* [PyTorch](https://pytorch.org/)
* [nltk](https://www.nltk.org/)
* [praw](https://praw.readthedocs.io/en/latest/)
* [gunicorn](https://gunicorn.org/)

The dependencies required are given in requirements.txt
To install the requirements 
``` bash
pip install -r requirements
```

### Data Acquistion
The data can be loaded in the form of a CSV Reddit's API PRAW.
The CSV file for the dataset used for training this model is present in dataset folder as train.csv and val.csv. The process of extracting the data and applying basic processing is present in [Data_Acquistion.ipynb](https://github.com/Ishan-Kumar2/Reddit-Flair-Detector/blob/master/notebooks/Data_Acquistion.ipynb)

### EDA
Exploratory data anylsis can be found in [EDA.ipynb](https://github.com/Ishan-Kumar2/Reddit-Flair-Detector/blob/master/notebooks/EDA.ipynb). Using this analysis, the features to be used were decided and certain flairs were removed due to lack of Data. 

# Model

## Model with Title
In this example I decided to use a simple LSTM on the title. The data was preprocessed and tokenised, followed by converting to pretrained word embeddings. for word embeddings I decided to go with GloVe 50d.
Further a model replacing LSTM with Bi-LSTM was also used as it allows context from both sides.

![BiLSTM](https://github.com/Ishan-Kumar2/Reddit-Flair-Detector/blob/master/utils/images/BiLSTM.png)
![LSTM Cell](https://github.com/Ishan-Kumar2/Reddit-Flair-Detector/blob/master/utils/images/LSTM.png)

## Model with Context, Title
In this example in addition to the Bi-LSTM model for th title I decided to also use the context(body) of the post. Since the body of the post can be as large as 14k words long, using a sequential model like LSTM would be very compute expensive. Hence I decided to use fastText as proposed in Bag of Tricks for Efficient Text Classification.

![Fasttext model](https://github.com/Ishan-Kumar2/Reddit-Flair-Detector/blob/master/utils/images/fasttext.png)


### Implementation Details
* Pretrained word embedding (GloVe 50d and fasttext simple 300d)
* Average word embedding for the entire sentence
* Feed through a feed forward network.
* Loss function Negative log likelihood is used.
* Optimizer Adam for training. 

## Seq2seq Model with Attention
With the intuition that certain keywords would be extremely essential in classfiying the post to a certain flair, I decided to use Attention mechanism on top of the BiLSTM and conctaenated the output with the final hidden state for classification. The reason I did this was for example in a title with a keyword like coronavirus at the start of the sentence there is a high chance that the final hidden state has small contribution of that, thereby potentially leading to misclassifying it.

![Attention mech](https://github.com/Ishan-Kumar2/Reddit-Flair-Detector/blob/master/utils/images/attention_mechanism.jpeg)

### Implementation Details
* Pretrained word embedding (GloVe 50d and fasttext simple 300d)
* Single Layer BiLSTM
* Optimizer- Adam for training
* Loss function Negative Log likelihood


## Seq2seq model with fastText
In this attempt I decided to concatenate the output of the fastText model for context and the BiLSTM model. In addition I also used features number_comments(Number of commments) and Score(score of the post), reason in EDA. These features were first passed through a feed forward layer. Output was concatenated with that of title model and context model.

![Attention](https://github.com/Ishan-Kumar2/Reddit-Flair-Detector/blob/master/utils/images/Attention.png)

### Implementation Details
* Pretrained word embedding (GloVe 50d and fasttext simple 300d)
* Single Layer BiLSTM
* Attention applied between final hidden state and all hidden state of LSTM
* Attention between context and final hidden state
* Concatenation of above vectors and the output of number of comments and score model output
* Optimizer- Adam for training
* Loss function Negative Log likelihood


## Results
The loss progressively decreases with number of epochs.
Also the number of correct classification increases with epochs.
![Result](https://github.com/Ishan-Kumar2/Reddit-Flair-Detector/blob/master/utils/images/download.png)

## Deploying
The final model used was the seq2seq, fastText combination model. Since this was built on PyTorch and the model itself was pretty large, the cumulative size exceeded the limit of Heroku. Although the webapp can be built locally using flask 

Also since the torchtext Fields use lambda functions they cant be saved using pickle, hence I have made a model without torchtext also which is the one loaded by default on the webapp.

``` bash
cd WebApp
export FLASK_APP=app.py
flask run
```
Then copy and paste the URL on a browser.

## Automated testing
The webapp can be tested automatically using the /automated_testing method. To do the following add the links to the reddit posts in file.txt, on each line. 
Then on running the flask app
``` bash
http://127.0.0.1:5000/automated_testing
```
The output will be stored in sample.json in JSON format

## Future Work
- [1.] [Byte Pair Encoding](https://arxiv.org/abs/1508.07909)- Since there are many Out of vocabulary words in the corpus like(COVID-19,coronavirus), I decided to finetune the embedding. The performance should still be compared to BPE as that is not affected by OOV words.
- [2.] [Transformers](https://arxiv.org/abs/1706.03762)- Using BERT for classifying both title and model class. 
- [3.] [ElMo](http://jalammar.github.io/illustrated-transformer/)-Contextual embedding.
- [4.] [Text CNN](https://towardsdatascience.com/cnn-sentiment-analysis-1d16b7c5a0e7)- Using a Text CNN model in place of fastText for the context model

## References
* [A. Joulin et. al. Bag of Tricks for Efficient Text Classification](https://arxiv.org/abs/1607.01759)
* [J Pennington et.al. Glove: Global Vectors for Word Representation](https://www.aclweb.org/anthology/D14-1162/)
* https://nlp.stanford.edu/projects/glove/
* https://torchtext.readthedocs.io/en/latest/
* https://mlexplained.com/2018/02/08/a-comprehensive-tutorial-to-torchtext/
* https://www.analyticsvidhya.com/blog/2018/04/a-comprehensive-guide-to-understand-and-implement-text-classification-in-python/
* https://towardsdatascience.com/feature-selection-on-text-classification-1b86879f548e
* https://towardsdatascience.com/feature-selection-on-text-classification-1b86879f548e
* https://praw.readthedocs.io/en/latest/
* https://github.com/AnubhavGupta3377/Text-Classification-Models-Pytorch
* https://towardsdatascience.com/how-i-improved-my-text-classification-model-with-feature-engineering-98fbe6c13ef3
