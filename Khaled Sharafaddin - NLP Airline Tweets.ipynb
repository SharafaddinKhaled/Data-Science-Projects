{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Khaled Sharafaddin - NLP Airline Tweets Sentiment Analysis - July.25.2021\n",
    "\n",
    "\n",
    "#### Background and Context:\n",
    "\n",
    "- Twitter posses 330 million monthly active users, which allows businesses to reach a broad population and connect with customers without intermediaries. On the other side, there’s so much information that it’s difficult for brands to quickly detect negative social mentions that could harm their business.\n",
    "\n",
    "- That's why sentiment analysis/classification, which involves monitoring emotions in conversations on social media platforms, has become a key strategy in social media marketing.\n",
    "\n",
    "\n",
    "- Listening to how customers feel about the product/services on Twitter allows companies to understand their audience, keep on top of what’s being said about their brand, and their competitors, and discover new trends in the industry.\n",
    "\n",
    " \n",
    "\n",
    "#### Data Description:\n",
    "\n",
    "- A sentiment analysis job about the problems of each major U.S. airline. Twitter data was scraped from February of 2015 and contributors were asked to first classify positive, negative, and neutral tweets, followed by categorizing negative reasons (such as \"late flight\" or \"rude service\")."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 1. Import the libraries, load dataset, the print shape of data, data description. "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "[nltk_data] Downloading package punkt to\n",
      "[nltk_data]     /Users/khaledsharafaddin/nltk_data...\n",
      "[nltk_data]   Package punkt is already up-to-date!\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Shape of the data: 14640 by 15\n",
      "Description of the data: \n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "[nltk_data] Downloading package stopwords to\n",
      "[nltk_data]     /Users/khaledsharafaddin/nltk_data...\n",
      "[nltk_data]   Package stopwords is already up-to-date!\n"
     ]
    },
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>tweet_id</th>\n",
       "      <th>airline_sentiment_confidence</th>\n",
       "      <th>negativereason_confidence</th>\n",
       "      <th>retweet_count</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>count</th>\n",
       "      <td>1.464000e+04</td>\n",
       "      <td>14640.000000</td>\n",
       "      <td>10522.000000</td>\n",
       "      <td>14640.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>mean</th>\n",
       "      <td>5.692184e+17</td>\n",
       "      <td>0.900169</td>\n",
       "      <td>0.638298</td>\n",
       "      <td>0.082650</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>std</th>\n",
       "      <td>7.791112e+14</td>\n",
       "      <td>0.162830</td>\n",
       "      <td>0.330440</td>\n",
       "      <td>0.745778</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>min</th>\n",
       "      <td>5.675883e+17</td>\n",
       "      <td>0.335000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>25%</th>\n",
       "      <td>5.685592e+17</td>\n",
       "      <td>0.692300</td>\n",
       "      <td>0.360600</td>\n",
       "      <td>0.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>50%</th>\n",
       "      <td>5.694779e+17</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>0.670600</td>\n",
       "      <td>0.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>75%</th>\n",
       "      <td>5.698905e+17</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>0.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>max</th>\n",
       "      <td>5.703106e+17</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>44.000000</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "           tweet_id  airline_sentiment_confidence  negativereason_confidence  \\\n",
       "count  1.464000e+04                  14640.000000               10522.000000   \n",
       "mean   5.692184e+17                      0.900169                   0.638298   \n",
       "std    7.791112e+14                      0.162830                   0.330440   \n",
       "min    5.675883e+17                      0.335000                   0.000000   \n",
       "25%    5.685592e+17                      0.692300                   0.360600   \n",
       "50%    5.694779e+17                      1.000000                   0.670600   \n",
       "75%    5.698905e+17                      1.000000                   1.000000   \n",
       "max    5.703106e+17                      1.000000                   1.000000   \n",
       "\n",
       "       retweet_count  \n",
       "count   14640.000000  \n",
       "mean        0.082650  \n",
       "std         0.745778  \n",
       "min         0.000000  \n",
       "25%         0.000000  \n",
       "50%         0.000000  \n",
       "75%         0.000000  \n",
       "max        44.000000  "
      ]
     },
     "execution_count": 1,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Libraries \n",
    "from IPython.core.interactiveshell import InteractiveShell\n",
    "InteractiveShell.ast_node_interactivity = \"all\"\n",
    "\n",
    "import re, string, unicodedata\n",
    "import numpy as np\n",
    "import random\n",
    "import pandas as pd\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "from collections import Counter\n",
    "from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator \n",
    "import nltk\n",
    "from nltk.corpus import stopwords\n",
    "import warnings\n",
    "warnings.filterwarnings('ignore')\n",
    "from nltk.stem.porter import PorterStemmer\n",
    "nltk.download('punkt')\n",
    "from bs4 import BeautifulSoup\n",
    "import nltk\n",
    "from nltk.corpus import stopwords\n",
    "nltk.download('stopwords')\n",
    "\n",
    "from sklearn.metrics import confusion_matrix\n",
    "import seaborn as sns\n",
    "import matplotlib.pyplot as plt \n",
    "\n",
    "\n",
    "# 2. Airline Tweet Dataset\n",
    "airlinetweets = pd.read_csv('/Users/khaledsharafaddin/Documents/Univ_Austin_Texas ML_AI/DataSets/Airline_Tweets.csv')\n",
    "\n",
    "print('Shape of the data:', airlinetweets.shape[0],'by', airlinetweets.shape[1])\n",
    "print('Description of the data: ')\n",
    "airlinetweets.describe()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 2. Understand of data columns \n",
    "- Drop all other columns except “text” and “airline_sentiment”.\n",
    "- Check the shape of the data.\n",
    "- Print the first 5 rows of data."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "airlines Shape of the data: 14640 by 2\n"
     ]
    },
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>airline_sentiment</th>\n",
       "      <th>text</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>neutral</td>\n",
       "      <td>@VirginAmerica What @dhepburn said.</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>positive</td>\n",
       "      <td>@VirginAmerica plus you've added commercials t...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>neutral</td>\n",
       "      <td>@VirginAmerica I didn't today... Must mean I n...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>negative</td>\n",
       "      <td>@VirginAmerica it's really aggressive to blast...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>negative</td>\n",
       "      <td>@VirginAmerica and it's a really big bad thing...</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "  airline_sentiment                                               text\n",
       "0           neutral                @VirginAmerica What @dhepburn said.\n",
       "1          positive  @VirginAmerica plus you've added commercials t...\n",
       "2           neutral  @VirginAmerica I didn't today... Must mean I n...\n",
       "3          negative  @VirginAmerica it's really aggressive to blast...\n",
       "4          negative  @VirginAmerica and it's a really big bad thing..."
      ]
     },
     "execution_count": 2,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# a. Drop all cols except text and airline_setiment\n",
    "airlines = airlinetweets.drop(['tweet_id', 'airline_sentiment_confidence',\n",
    "       'negativereason', 'negativereason_confidence', 'airline',\n",
    "       'airline_sentiment_gold', 'name', 'negativereason_gold',\n",
    "       'retweet_count', 'tweet_coord', 'tweet_created',\n",
    "       'tweet_location', 'user_timezone'], axis=1)\n",
    "\n",
    "\n",
    "# b. Check the shape of the data\n",
    "print('airlines Shape of the data:', airlines.shape[0],'by', airlines.shape[1])\n",
    "\n",
    "# c. Print the first 5 rows of data.\n",
    "airlines.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 3. Text pre-processing: Data preparation"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [],
   "source": [
    "# a. Html tag removal.\n",
    "\n",
    "airlines_clean = airlines\n",
    "remove_html =[]\n",
    "data_size = airlines_clean.shape[0]\n",
    "\n",
    "for i in range(0, data_size):\n",
    "    remove_html.append(BeautifulSoup(airlines_clean['text'][i]).get_text())\n",
    "    \n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [],
   "source": [
    "# b. Remove the numbers, char and punctuations\n",
    "remove_numbers =[]\n",
    "\n",
    "for i in range(0, data_size):\n",
    "    remove_numbers.append(re.sub(\"[^a-zA-Z]\",\" \", remove_html[i]))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [],
   "source": [
    "# c. Cnversion to lower:\n",
    "\n",
    "to_lower = []\n",
    "for i in range(0, data_size):\n",
    "    to_lower.append(remove_numbers[i].lower()) \n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [],
   "source": [
    "# d. Tokenize the list of tweets:\n",
    "\n",
    "from nltk.corpus import stopwords\n",
    "from nltk.tokenize import word_tokenize\n",
    "stop_words = set(stopwords.words('english'))\n",
    "\n",
    "word_tokens = []\n",
    "for i in range(0, data_size):\n",
    "    word_tokens.append(word_tokenize(to_lower[i]))\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [],
   "source": [
    "# e. Remove stop words\n",
    "\n",
    "stop_words = set(stopwords.words('english'))\n",
    "remove_stopwords = []\n",
    "for i in range(0, data_size):\n",
    "    row = []\n",
    "    for airlinetweet in word_tokens[i]:\n",
    "        if airlinetweet not in stop_words:\n",
    "            row.append(airlinetweet)\n",
    "\n",
    "    remove_stopwords.append(row)\n",
    "            "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Convert list of lists to list of strings \n",
    "tokenized_tweets = [' '.join(my_list) for my_list in remove_stopwords]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "[nltk_data] Downloading package wordnet to\n",
      "[nltk_data]     /Users/khaledsharafaddin/nltk_data...\n",
      "[nltk_data]   Package wordnet is already up-to-date!\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "True"
      ]
     },
     "execution_count": 9,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# f. lemmatize text:\n",
    "from nltk.stem.wordnet import WordNetLemmatizer \n",
    "import nltk\n",
    "nltk.download('wordnet')\n",
    "lemmatizer = WordNetLemmatizer()\n",
    "\n",
    "lemmatize_text = []\n",
    "\n",
    "for i in range(0, data_size):\n",
    "     lemmatize_text.append(lemmatizer.lemmatize((tokenized_tweets[i])))\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "metadata": {},
   "outputs": [],
   "source": [
    "# g. Join the words in the list to convert back to text string in the data frame.\n",
    "\n",
    "cleaned_tweets = pd.DataFrame(\n",
    "{\n",
    "    'Airline_sentiment': airlines_clean['airline_sentiment'], \n",
    "    'Tweet_Text': lemmatize_text\n",
    "})\n",
    "\n",
    "# Copy to create word cloud before hot encoding the sentiment to numbers \n",
    "word_cloud_sentiment = cleaned_tweets\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Airline_sentiment</th>\n",
       "      <th>Tweet_Text</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>0</td>\n",
       "      <td>virginamerica dhepburn said</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>1</td>\n",
       "      <td>virginamerica plus added commercials experienc...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>0</td>\n",
       "      <td>virginamerica today must mean need take anothe...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>2</td>\n",
       "      <td>virginamerica really aggressive blast obnoxiou...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>2</td>\n",
       "      <td>virginamerica really big bad thing</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   Airline_sentiment                                         Tweet_Text\n",
       "0                  0                        virginamerica dhepburn said\n",
       "1                  1  virginamerica plus added commercials experienc...\n",
       "2                  0  virginamerica today must mean need take anothe...\n",
       "3                  2  virginamerica really aggressive blast obnoxiou...\n",
       "4                  2                 virginamerica really big bad thing"
      ]
     },
     "execution_count": 11,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# h. Replace Airline_sentiment with dummy variables (encoding)\n",
    "\n",
    "cleaned_tweets[\"Airline_sentiment\"].replace({\"neutral\": 0, \"positive\": 1, \"negative\":2}, inplace=True)\n",
    "\n",
    "# i. Print the first 5 rows of data after pre-processing.\n",
    "\n",
    "cleaned_tweets.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 4. Vectorization:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "CountVectorizer Shape: (14640, 13493)\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "TfidfVectorizer()"
      ]
     },
     "execution_count": 12,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Inverse Document Victorizer Frequencies: \n",
      "\n",
      "[4.98577902 9.89843391 9.89843391 ... 9.89843391 9.89843391 9.89843391]\n",
      "\n",
      "TfidfVectorizer Shape: (14640, 13493)\n"
     ]
    }
   ],
   "source": [
    "# a. CountVectorizer: \n",
    "from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer\n",
    "Tweets_vectorizer = CountVectorizer()\n",
    "tweets_list = list(cleaned_tweets['Tweet_Text'])\n",
    "vectorized_tweets = Tweets_vectorizer.fit_transform(cleaned_tweets['Tweet_Text']) \n",
    "vectorized_tweets = vectorized_tweets.toarray() \n",
    "\n",
    "print('CountVectorizer Shape:', vectorized_tweets.shape)\n",
    "\n",
    "# --------------------------\n",
    "\n",
    "# b. TfidfVectorizer\n",
    "Tweets_TfidfVectorizer = TfidfVectorizer()\n",
    "\n",
    "# tokenize and build vocab\n",
    "Tweets_TfidfVectorizer.fit(cleaned_tweets['Tweet_Text'])\n",
    "\n",
    "#summarize\n",
    "print('Inverse Document Victorizer Frequencies: \\n')\n",
    "print(Tweets_TfidfVectorizer.idf_)\n",
    "\n",
    "\n",
    "# encode document\n",
    "TfidfVectorizer = Tweets_TfidfVectorizer.transform(cleaned_tweets['Tweet_Text']).toarray()\n",
    "\n",
    "# summarize encoded vector\n",
    "print('\\nTfidfVectorizer Shape:', TfidfVectorizer.shape)\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 5. Fit and evaluate the model using both types of vectorization"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Training the random forest on CountVectorizer...\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "RandomForestClassifier(n_estimators=10, n_jobs=4)"
      ]
     },
     "execution_count": 13,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Accuracy on training set :  0.9844847775175644\n",
      "Accuracy on test set :  0.7363387978142076\n"
     ]
    }
   ],
   "source": [
    "from sklearn.ensemble import RandomForestClassifier\n",
    "from sklearn.model_selection import cross_val_score\n",
    "from sklearn.model_selection import train_test_split\n",
    "\n",
    "# 5. Using random forest classifier for CountVectorizer\n",
    "\n",
    "\n",
    "rf = RandomForestClassifier(n_estimators=10, n_jobs=4)\n",
    "print('Training the random forest on CountVectorizer...')\n",
    "\n",
    "X = vectorized_tweets\n",
    "y = cleaned_tweets['Airline_sentiment']\n",
    "\n",
    "\n",
    "\n",
    "# Split to train and test sets: \n",
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.30, random_state=1)\n",
    "\n",
    "# Fit the model\n",
    "rf = rf.fit(X_train, y_train)\n",
    "rf\n",
    "print(\"Accuracy on training set : \",rf.score(X_train, y_train))\n",
    "print(\"Accuracy on test set : \",rf.score(X_test, y_test))\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<matplotlib.axes._subplots.AxesSubplot at 0x7fd91cfbf7d0>"
      ]
     },
     "execution_count": 14,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "text/plain": [
       "Text(0.5, 15.0, 'True labels')"
      ]
     },
     "execution_count": 14,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "text/plain": [
       "Text(33.0, 0.5, 'Predicted labels')"
      ]
     },
     "execution_count": 14,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "text/plain": [
       "Text(0.5, 1, 'Confusion Matrix')"
      ]
     },
     "execution_count": 14,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "text/plain": [
       "[Text(0.5, 0, '0'), Text(1.5, 0, '1'), Text(2.5, 0, '2')]"
      ]
     },
     "execution_count": 14,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "text/plain": [
       "[Text(0, 0.5, '0'), Text(0, 1.5, '1'), Text(0, 2.5, '2')]"
      ]
     },
     "execution_count": 14,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      " precision and recall: \n",
      "\n",
      "              precision    recall  f1-score   support\n",
      "\n",
      "           0       0.54      0.51      0.53       936\n",
      "           1       0.68      0.54      0.60       715\n",
      "           2       0.80      0.86      0.83      2741\n",
      "\n",
      "    accuracy                           0.74      4392\n",
      "   macro avg       0.68      0.64      0.66      4392\n",
      "weighted avg       0.73      0.74      0.73      4392\n",
      "\n"
     ]
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 2 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "# Predict X_test and print Confusion Matrix:\n",
    "\n",
    "from sklearn import metrics\n",
    "\n",
    "vect_predict = rf.predict(X_test)    \n",
    "\n",
    "ax= plt.subplot()\n",
    "lebels = [0,1,2]\n",
    "cm = confusion_matrix(y_test, vect_predict, lebels)\n",
    "sns.heatmap(cm, annot=True, fmt='g', ax=ax);  \n",
    "\n",
    "# labels, title and ticks\n",
    "ax.set_xlabel('True labels');ax.set_ylabel('Predicted labels'); \n",
    "ax.set_title('Confusion Matrix'); \n",
    "ax.xaxis.set_ticklabels([0,1,2]); ax.yaxis.set_ticklabels([0,1,2]);\n",
    "\n",
    "\n",
    "# Printing the precision and recall, among other metrics\n",
    "print('\\n precision and recall: \\n')\n",
    "print(metrics.classification_report(y_test, vect_predict, labels=[0, 1, 2]))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Training the random forest on TfidfVectorizer...\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "RandomForestClassifier(n_estimators=10, n_jobs=4)"
      ]
     },
     "execution_count": 15,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Mean Cross Validation Accuracy for TfidfVectorizer: 70.0 %\n"
     ]
    }
   ],
   "source": [
    "# Create random Forest Classifier\n",
    "rf2 = RandomForestClassifier(n_estimators=10, n_jobs=4)\n",
    "print('Training the random forest on TfidfVectorizer...')\n",
    "\n",
    "# Get the train sets: \n",
    "X = TfidfVectorizer\n",
    "y = cleaned_tweets['Airline_sentiment']\n",
    "\n",
    "# Fit the model\n",
    "rf2 = rf2.fit(X, y)\n",
    "rf2\n",
    "\n",
    "# We perform k-fold cross validation with 10 folds\n",
    "# The model trains on K-1 (9) folds, and uses the 10th fold for testing accuracy. \n",
    "# This is repeated 10 times, then we take the avrage of the accuracy score of all 10 folds\n",
    "\n",
    "cv_avg_accuracy = round(np.mean(cross_val_score(rf2, X, y, cv=10)),2)\n",
    "print('Mean Cross Validation Accuracy for TfidfVectorizer:', cv_avg_accuracy*100, '%')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 43,
   "metadata": {},
   "outputs": [],
   "source": [
    "# # Word cloud for positive sentiment\n",
    "\n",
    "Positive_sent = word_cloud_sentiment[word_cloud_sentiment['Airline_sentiment']=='positive']\n",
    "Negative_sent = word_cloud_sentiment[word_cloud_sentiment['Airline_sentiment']=='negative']\n",
    "Neutral_sent = word_cloud_sentiment[word_cloud_sentiment['Airline_sentiment']=='neutral']\n",
    "\n",
    "import wordcloud\n",
    "def show_wordcloud(Positive_sent, title):\n",
    "    text = ' '.join(Positive_sent['content'].astype(str).tolist())\n",
    "    stopwords = set(wordcloud.STOPWORDS)\n",
    "    \n",
    "    fig_wordcloud = wordcloud.WordCloud(stopwords=stopwords,background_color='white',\n",
    "                    colormap='viridis', width=800, height=600).generate(text)\n",
    "    \n",
    "    plt.figure(figsize=(14,11), frameon=True)\n",
    "    plt.imshow(fig_wordcloud)  \n",
    "    plt.axis('off')\n",
    "    plt.title(title, fontsize=30)\n",
    "    plt.show()\n",
    "    "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<matplotlib.image.AxesImage at 0x7fd75ee53290>"
      ]
     },
     "execution_count": 29,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "text/plain": [
       "(-0.5, 399.5, 199.5, -0.5)"
      ]
     },
     "execution_count": 29,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "# Positive Airline Tweets: \n",
    "data = Positive_sent['Tweet_Text'].value_counts().to_dict()\n",
    "wc = WordCloud().generate_from_frequencies(data)\n",
    "\n",
    "plt.imshow(wc)\n",
    "plt.axis('off')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 36,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<matplotlib.image.AxesImage at 0x7fd75f933ad0>"
      ]
     },
     "execution_count": 36,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "text/plain": [
       "(-0.5, 399.5, 199.5, -0.5)"
      ]
     },
     "execution_count": 36,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "# Negative Airline Tweets: \n",
    "data = Negative_sent['Tweet_Text'].value_counts().to_dict()\n",
    "wc = WordCloud().generate_from_frequencies(data)\n",
    "\n",
    "plt.imshow(wc)\n",
    "plt.axis('off')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 41,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<matplotlib.image.AxesImage at 0x7fd75f933650>"
      ]
     },
     "execution_count": 41,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "text/plain": [
       "(-0.5, 399.5, 199.5, -0.5)"
      ]
     },
     "execution_count": 41,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "# Neutral Airline Tweets: \n",
    "data = Neutral_sent['Tweet_Text'].value_counts().to_dict()\n",
    "wc = WordCloud().generate_from_frequencies(data)\n",
    "\n",
    "plt.imshow(wc)\n",
    "plt.axis('off')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Summary\n",
    "- The airline tweet dataset contains 14640 rows and 15 columns that has customer reviews of airline services including United, American airllines, Southwest. \n",
    "\n",
    "- The reviews are categorized into Positive, Negative or Neutral Sentiments. \n",
    "\n",
    "- The purpose of this project is to test whether a customer review is positive, negative or neutral, using a machine learning classification algorithm.\n",
    "\n",
    "- Before applying the dataset into ML algorithm, data pre-processing steps have been taken, including: \n",
    "    1. Removing HTML tags\n",
    "    2. Remove numbers and special characters\n",
    "    3. convert text to lower case\n",
    "    4. Stop words removal\n",
    "    5. Tokenization and stemming\n",
    " \n",
    " \n",
    "- Random Forest is used to determine the category of the review, achieving 74% accuracy. \n",
    "\n",
    "- Confusion Matrix is created, proving important metrics such as precision, recall and F score. \n",
    "\n",
    "- Wordcloud has been generated to see the most frequent words used in the reviews. \n",
    "- Positive words include thank you, great, and love.\n",
    "- Negative sentiments include cancelled, bad, and worst.\n",
    "- Neutral sentiments include americanair, usairways and southwestair, which don't provide analytical value. \n",
    "\n",
    "- The sentiment analysis model can be imporved further using Neural Networks, or ensemble methods to increase accuracy of sentiment prediction.  "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.4"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
