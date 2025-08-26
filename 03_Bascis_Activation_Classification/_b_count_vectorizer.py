import polars as pl
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

df = pd.read_csv("./data/SMSSpamCollection", sep="\t", names=["type", "message"])
df = pl.DataFrame(df)
df = (
    df
    .with_columns(
        spam=pl.when(pl.col("type").eq("spam")).then(pl.lit(1)).otherwise(0)
    )
    .select("message", "spam")
)

#--- basic introduction
#
# the goal of the count vectorizer is to create a sparse matrix of token counts
# the frequency of tokens is counted and this sparse matrix is useful for having a lot of token counts with 0
## max features on how many words this vectorizer will examine 
## it also removes upper lower case and specific signs, so there is some preprocessing going obn, so we do not have to this ourselves
#cv = CountVectorizer(max_features=6) #again max_features max amount of words
#documents = [
#    "Hello world. Hello, again, my name is Patrick!",
#    "Hello earth. MY NAME is Ben."
#]
#cv.fit(documents)
#print(cv.get_feature_names_out())
#
#out = cv.transform(documents)
#print(out)
#
#print(out.todense())
#gives two rows and shows how much each of the max features words occur per document
#this gives out a matrix, but we have so many 0s it will show a sparse version of it. 

cv = CountVectorizer(max_features=1000)
messages = cv.fit_transform(df["message"]) #both steps together
print(messages)
print(messages[0]) # prints a word with the index 349
print(cv.get_feature_names_out()[349]) # the word is "go" 888 would be "until"

# Now we have roughly 1000 columns with words