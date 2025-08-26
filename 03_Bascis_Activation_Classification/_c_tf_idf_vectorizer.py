# takes into accounts filler words like "the" which should have no meaning
# TF means here term and how often a term appears here
# TF is number of times of term tin document d / total numbers of terms in document d
# hello world = 2 terms, for term hello = 1/2 = 0.5
# short message: each word has a higher weight or meaning, if we have a longer message, each of the term has lower weight due to the division
# the idea is to reduce the weight of common term like the word the

# IDF = Inverse document frequency, the log lets it fade out:
# (to avoid divison by zero we add a 1, but we can do log( (1+ total number of documents)/(1+total number of documents containing the term) )
# when you have the word the which is appearing 5000 times in 5000 documents, the IDF would be 0 turning it into a total meaningless filler word
# this is totally different if a word appears only once in a document

#TF-IDF = TF * IDF and it prioritizes important, rare words over frequently occuring less meaningful words

import polars as pl
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

df = pd.read_csv("./data/SMSSpamCollection", sep="\t", names=["type", "message"])
df = pl.DataFrame(df)
df = (
    df
    .with_columns(
        spam=pl.when(pl.col("type").eq("spam")).then(pl.lit(1)).otherwise(0)
    )
    .select("message", "spam")
)

tfidf = TfidfVectorizer(max_features=1000)
messages = tfidf.fit_transform(df["message"]) #both steps together
print(messages)
print(messages[0]) # prints a word with the index 349
print(tfidf.get_feature_names_out()[349]) # the word is "go" 888 would be "until"