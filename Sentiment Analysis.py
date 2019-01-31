import pandas as pd  
from bs4 import BeautifulSoup
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


train = pd.read_csv('labeledTrainData.tsv', header=0, delimiter='\t', quoting=3)     

def review_to_words(raw_review):
    removedTags = BeautifulSoup(raw_review)         #1. Remove HTML
    upperAndLowerRemains = re.sub('[^a-zA-Z]'," ",removedTags.get_text()) #2. Remove non letters
    toLowerAndSplit = upperAndLowerRemains.lower().split() #3. Convert to lowercase and split it into words
    stops = set(stopwords.words('english'))
    stopwordsRemoved = [w for w in toLowerAndSplit if not w in stops]  #4. Remove stops words
    complete_review = " ".join(stopwordsRemoved);  #5. Joint back and return the joined sentence
    return complete_review
    
CleanedListOfReviews = []
BagOfWords = []
for iterator in range(0,train["review"].size): 
    if iterator%1000 == 0 or iterator==24999:     #Checking progress after every 1000 Reviews
        print("Cleaned Reviews: ",iterator)
    complete_review = review_to_words(train["review"][iterator])
    CleanedListOfReviews.append(complete_review)
    BagOfWords.append(train["sentiment"][iterator])

# validationSet = 0.2  #20% is validation set
# trainingSet = 0.8
# vocabularySize = 3000
# smoothingFactor = 0.00001
# vectorizer = CountVectorizer(analyzer = "word", tokenizer = None, preprocessor = None, stop_words = None, max_features = vocabularySize)
# train['IMDB'] = train['review'].apply(review_to_words)  #Apply the function of review_to_words for all reviews and all the cleaned reviews are stored in the list of train['IMDB'] 
# X = vectorizer.fit_transform(train['IMDB'])    
# X = X.toarray()
# Y = np.array(train['sentiment'])  
# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=validationSet)
# clf = MultinomialNB(alpha=smoothingFactor) # alpha=0 means no laplace smoothing
# clf.fit(X_train, Y_train)  
# temp = clf.predict(X_test); 
# temp2 = (float)(accuracy_score(Y_test,temp,normalize=True));
# accuracy = temp2*100; 
# print("\nAccuracy on 20% validation set with smoothing factor 0.00001 and vocabulary size",vocabularySize," is:",accuracy)
#print (accuracy);


vocabularySize = 5000
smoothingFactor = 5
vectorizer = CountVectorizer(analyzer = "word", tokenizer = None, preprocessor = None, stop_words = None, max_features = vocabularySize)  # Initialize the "CountVectorizer" object, which is scikit-learn's bag of words tool.  

# ReviewsTrainedSet = CleanedListOfReviews[0:int(len(CleanedListOfReviews) * 0.8)]  #Gets only 80% of reviews
# ReviewsValidationSet = CleanedListOfReviews[int(len(CleanedListOfReviews) * 0.8):len(CleanedListOfReviews)]    #Gets only 20% of reviews
# 
# BagOfWordsTrainedSet = BagOfWords[0:len(BagOfWords) * 0.8]  #Gets only 80% Bag of Words
# BagOfWordsValidationSet = BagOfWords[len(CleanedListOfReviews) * 0.8:len(CleanedListOfReviews)]   #Gets only 20% Bag of Words
data_features = vectorizer.fit_transform(CleanedListOfReviews)
data_features = data_features.toarray()

trainingSet = 0.8
dataSize = train['review'].size
trainSize = dataSize * trainingSet
SentimentsTrainedSet = []
ReviewsTrainedSet = []
ReviewsValidationSet = []
SentimentsValidationSet = []
for i in range( 0, dataSize):
	if(i < trainSize):
		SentimentsTrainedSet.append(BagOfWords[i])
		ReviewsTrainedSet.append(data_features[i])
	else:
		SentimentsValidationSet.append(BagOfWords[i])
		ReviewsValidationSet.append(data_features[i])



# Fitting the model to Naive Bayes Classifier
clf = MultinomialNB(alpha=smoothingFactor)
clf.fit(np.array(ReviewsTrainedSet), np.array(SentimentsTrainedSet))


#Predicting on Validation set
pred_labels = clf.predict(np.array(ReviewsValidationSet))
val_labels = np.array(SentimentsValidationSet)

#Calculating Accuracy
accuracy = float((pred_labels == val_labels).sum())
total = val_labels.size

acc_perc = (accuracy/total)*100
print("\nAccuracy on 20% validation set with smoothing factor ",smoothingFactor," and vocabulary size ",vocabularySize," is: ",acc_perc)




# #Calculating Accuracy
# acc = float((pred_labels == val_labels).sum())
# total = val_labels.size
# 
# 
# 
# #tX = vectorizer.fit_transform(ReviewsValidationSet).toarray()
# predictions = clf.predict(tX)
# 
# for i,pred in enumerate(predictions):
# 	if pred == BagOfWordsValidationSet[i]:
# 		accuracy += 1
# 
# accuracy = 100 * (accuracy/len(predictions))
# print("\nAccuracy on 20% validation set with smoothing factor ",smoothingFactor," and vocabulary size ",vocabularySize," is: ",accuracy)
#  
