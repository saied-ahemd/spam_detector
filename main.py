from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
import pandas as pd
import nltk
import pickle
# let's load the data
df = pd.read_csv('../data/archive/spam.csv', encoding='ISO-8859-1')
df = df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1)
df.columns = ['label', 'data']

# convert the label to binary label
df['b_label'] = df['label'].map({'ham': 0, 'spam': 1})
y = df['b_label'].values
# to convert out data to text
vec = TfidfVectorizer()
X = vec.fit_transform(df['data'])
# now let's spilt our data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
# now we can train our model after we process our data first naive Byes
model = SVC()
model.fit(X_train, y_train)
score_test = model.score(X_test, y_test)
score_train = model.score(X_train, y_train)
y_pre = model.predict(X_test)
print(f'test model score {score_test}')
print(f'train model score {score_train}')
# now let's use accuracy_score to see the score
a_score = accuracy_score(y_test, y_pre)
print(f'the accuracy_score is {a_score}')
# now you can check the values that the model was wrong on it
# now i get the sneaky spam that the model was wrong about
df['pre'] = model.predict(X)
sneaky_spam = df[(df['pre'] == 0) & (df['b_label'] == 1)]['data']
# for ms in sneaky_spam:
#     print(ms)

# get the most common word in the spam or ham
doc = df[df['b_label'] == 0]['data']
all_ham_data = []
for i in doc:
    all_ham_data.append(i.lower())
all_ham_word = []
# get all the word in the all the rows data
for w in all_ham_data:
    all_ham_word.append(nltk.word_tokenize(w))

wo = []
# get all the word in the hole data set
for je in all_ham_word:
    je = nltk.FreqDist(je)
    for i in je.keys():
        wo.append(i)

wo = nltk.FreqDist(wo)
most_common = wo.most_common(50)
print(most_common)
# now we will save the model
# sa = open('spam_model.pickle', 'wb')
# pickle.dump(model, sa)
# load the model
spam_model = pickle.load(open('spam_model.pickle', 'rb'))
y_pre2 = spam_model.predict(X_test)
ac = accuracy_score(y_test, y_pre2)






