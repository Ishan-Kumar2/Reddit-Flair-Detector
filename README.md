# Reddit-Flair-Detector

This model aims to detect the flair of a Reddit post from '/india' subreddit. The model classifies the flair of a post given its URL into Coronavirus, Nonpolitical, Political and others. The model can also be hosted as a web service.

## Installation
First clone the repo to the local device using

git clone https://github.com/Ishan-Kumar2/Reddit-Flair-Detector.git

This project requires
The dependencies required are given in requirements.txt
To install the requirements 
pip install -r requirements

### Data Acquistion
The data can be loaded in the form of a CSV Reddit's API PRAW.
The CSV file for the dataset used for training this model is present in dataset folder as train.csv and val.csv. The process of extracting the data and applying basic processing is present in Data_Acquistion.ipynb

### EDA
Exploratory data anylsis can be found in EDA.ipynb. Using this analysis, the features to be used were decided and certain flairs were removed due to lack of Data. 

## Model

## Model with Title
In this example I decided to use a simple LSTM on the title. The data was preprocessed and tokenised, followed by converting to pretrained word embeddings. for word embeddings I decided to go with GloVe 50d.
Further a model replacing LSTM with Bi-LSTM was also used as it allows context from both sides.

//Image of LSTM Cell

## Model with Context, Title
In this example in addition to the Bi-LSTM model for th title I decided to also use the context(body) of the post. Since the body of the post can be as large as 14k words long, using a sequential model like LSTM would be very compute expensive. Hence I decided to use fastText as proposed in Bag of Tricks for Efficient Text Classification.

### Implementation Details
Pretrained Word Embedding for encoding words
Average word embedding for the entire sentence
Feed through a feed forward network.
Loss function Negative log likelihood is used.
Optimizer Adam for training. 

## Seq2seq Model with Attention
With the intuition that certain keywords would be extremely essential in classfiying the post to a certain flair, I decided to use Attention mechanism on top of the BiLSTM and conctaenated the output with the final hidden state for classification. The reason I did this was for example in a title with a keyword like coronavirus at the start of the sentence there is a high chance that the final hidden state has small contribution of that, thereby potentially leading to misclassifying it.

### Implementation Details
Pretrained GloVe word embedding (50d)
Single Laye BiLSTM
Optimizer- Adam for training
Loss function Negative Log liklihood


## Seq2seq model with fastText
In this attempt I decided to concatenate the output of the fastText model for context and the BiLSTM model. In addition I also used features number_comments(Number of commments) and Score(score of the post), reason in EDA. These features were first passed through a feed forward layer. Output was concatenated with that of title model and context model.

## Results
The loss progressively decreases with number of epochs.
Also the number of correct classification increases with epochs.


## Deploying
The final model used was the seq2seq, fastText combination model. Since this was built on PyTorch and the model itself was pretty large, the cumulative size exceeded the limit of Heroku. Although the webapp can be built locally using 
flask 

Also since the torchtext Fields use lambda functions they cant be saved using pickle, hence I I have made a model without torchtext also which is the one loaded by default on the webapp.

## Future Work
-Byte Pair Encoding- Since there are many Out of vocabulary words in the corpus like(COVID-19,coronavirus), I decided to finetune the embedding. The performance should still be compared to BPE as that is not affected by OOV words.
-Transformers- Using BERT for classifying both title and model class. 
-ElMo-Contextual embedding.

## References
