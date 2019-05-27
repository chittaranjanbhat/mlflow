import re
import matplotlib
matplotlib.use("Agg")
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
import string
import nltk
import warnings 
warnings.filterwarnings("ignore", category=DeprecationWarning)





train_data  = pd.read_csv('train_E6oV3lV.csv')
test_data = pd.read_csv('test_tweets_anuFYb8.csv')
combi = train_data.append(test_data, ignore_index=True)
def remove_pattern(input_txt, pattern):
    r = re.findall(pattern, input_txt)
    for i in r:
        input_txt = re.sub(i, '', input_txt)
        
    return input_txt    


### Remove any of the characters starting with @....so as to ignore the user handle mentions
combi['tidy_tweet'] = np.vectorize(remove_pattern)(combi['tweet'], "@[\w]*")

###Replace all the special characters with space
combi['tidy_tweet'] = combi['tidy_tweet'].str.replace("[^a-zA-Z#]", " ")
print(train_data.head())


#### Restrictind the use of words having length less than 3
combi['tidy_tweet'] = combi["tidy_tweet"].apply(lambda x : " ".join([w for w  in  x.split()  if len(w)>3]))




#### Tokenization of the data



tokenized_text = combi["tidy_tweet"].apply(lambda x : x.split())
print(tokenized_text)

"""
Stemming
"""
from nltk.stem.porter import *

stemmer = PorterStemmer()

tokenized_text = tokenized_text.apply(lambda x: [stemmer.stem(i) for i in x]) # stemming
print("Tokenized tweet handle",tokenized_text.head())


for i in range(len(tokenized_text)):
	tokenized_text[i] = " ".join(tokenized_text[i])
combi["token_tweet"] = tokenized_text


print("A wordcloud is a visualization wherein the most frequent words appear in large size and the less frequent words appear in smaller sizes")



total_words = " ".join(i for i in combi["token_tweet"])
#print(total_words)


from wordcloud import WordCloud
wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(total_words)

plt.figure(figsize=(10, 7))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.savefig("wordcloud.png")
plt.close()
print("positive and neagtive tweets display")



hash_all_positive = " ".join([i for  i in combi["token_tweet"][combi["label"]==0]])

hash_all_negative = " ".join([i for  i in combi["token_tweet"][combi["label"]==1]])
wd1 = WordCloud(width=1000,height=600,random_state=21,max_font_size=100).generate(hash_all_positive)
plt.figure(figsize=(10, 7))
plt.imshow(wd1, interpolation="bilinear")
plt.axis('off')
plt.savefig("wordcloud2.png")
plt.close()


wd2 = WordCloud(width=1000,height=600,random_state=21,max_font_size=100).generate(hash_all_negative)
plt.figure(figsize=(10, 7))
plt.imshow(wd2, interpolation="bilinear")
plt.axis('off')
plt.savefig("wordcloud3.png")
plt.close()


####Draw wordcloud for the negative and positive tweets
def hashtag_find(x):
	hashtags=[]
	for i in x:
		h = re.findall(r"#(\w+)", i)
		hashtags.append(h)
	print("hashtags",hashtags)
	return hashtags



print("hashtags from the positive tweets")

hash_positive = hashtag_find(combi["token_tweet"][combi["label"]==0])


hash_negative = hashtag_find(combi["token_tweet"][combi["label"]==1])
hash_positive = sum(hash_positive,[])
hash_negative = sum(hash_negative,[])


###Frequency Distribution


a = nltk.FreqDist(hash_positive)
print("hashpositive frequency distribution",a)
d = pd.DataFrame({'Hashtag': list(a.keys()),
                  'Count': list(a.values())})
d = d.nlargest(columns="Count", n = 10) 
plt.figure(figsize=(16,5))
ax = sns.barplot(data=d, x= "Hashtag", y = "Count")
ax.set(ylabel = 'Count')
plt.savefig("hashtag_positive.png")
plt.close()



b = nltk.FreqDist(hash_negative)
d2 = pd.DataFrame({"Hashtag":list(b.keys()),"Counts":list(b.values())})
plt.figure(figsize=(16,5))
ax = sns.barplot(data=d2, x= "Hashtag", y = "Counts")
ax.set(ylabel = 'Counts')
plt.savefig("hashtag_negative.png")
plt.close()





print("Removing the stop words and building a count vectorizer")


from sklearn.feature_extraction.text import CountVectorizer
bow_vectorizer = CountVectorizer(max_df=0.90, min_df=2, max_features=1000, stop_words='english')
# bag-of-words feature matrix
bow = bow_vectorizer.fit_transform(combi['token_tweet'])

print("Working with the tf-idf vectorizer")


from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer(max_df=0.90, min_df=2, max_features=1000, stop_words='english')
# TF-IDF feature matrix
tfidf = tfidf_vectorizer.fit_transform(combi['tidy_tweet'])



print("Building the model by using the bag of words")
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

train_bow = bow[:31962,:]
test_bow = bow[31962:,:]


xtrain_bow, xvalid_bow, ytrain, yvalid = train_test_split(train_bow, train_data['label'], random_state=42, test_size=0.3)




lreg = LogisticRegression()
lreg.fit(xtrain_bow, ytrain) # training the model


prediction = lreg.predict_proba(xvalid_bow) # predicting on the validation set
print(prediction)
print(prediction[:,1])
prediction_int = prediction[:,1] >= 0.3 # if prediction is greater than or equal to 0.3 than 1 else 0
prediction_int = prediction_int.astype(np.int)
print("f1_score",f1_score(yvalid, prediction_int))


test_pred = lreg.predict_proba(test_bow)
test_pred_int = test_pred[:,1] >= 0.3
test_pred_int = test_pred_int.astype(np.int)
test_data['label'] = test_pred_int
submission = test[['id','label']]
submission.to_csv('sub_lreg_bow.csv', index=False) # writing data to a CSV file


print("Building the model by using tf-idf vectorizer")

train_tfidf = tfidf[:31962,:]
test_tfidf = tfidf[31962:,:]


xtrain_tfidf = train_tfidf[ytrain.index]
xvalid_tfidf = train_tfidf[yvalid.index]


lreg.fit(xtrain_tfidf, ytrain)



prediction = lreg.predict_proba(xvalid_tfidf)
prediction_int = prediction[:,1] >= 0.3
prediction_int = prediction_int.astype(np.int)


print(f1_score(yvalid, prediction_int))



test_pred_tfidf = lreg.predict_probab(test_tfidf)
test_pred_tfidf_int = test_pred_tfidf[:,1] >= 0.3
test_pred_tfidf_int = test_pred_tfidf_int.astype(np.int)


test_data['label2'] = test_pred_tfidf_int
submission = test[['id','label2']]
submission.to_csv('sub_lreg_tfidf.csv', index=False)
