import streamlit as st
import string
import pandas as pd
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
nltk.download('stopwords')
from nltk.corpus import stopwords


df = pd.read_csv(r'spam.csv', encoding='ISO-8859-1')
df = df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1)
df.rename(columns = {'v1': 'labels', 'v2': 'message'}, inplace=True)
df.drop_duplicates(inplace=True)
df['labels'] = df['labels'].map({'ham': 0, 'spam': 1})
# print(df.head(10))


def clean_data(message):
    message_wo_pmc = [character for character in message if character not in string.punctuation]
    message_wo_pmc = ''.join(message_wo_pmc)

    sep = ' '
    return sep.join([word for word in message_wo_pmc.split() if word.lower() not in stopwords.words('english')])


df['message'] = df['message'].apply(clean_data)
x = df['message']
y = df['labels']
vect = CountVectorizer()
x = vect.fit_transform(x)


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=123)
model = MultinomialNB()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
# print(accuracy_score(y_test, y_pred))


def predict(text):
    labels = ['Not Spam', 'Spam']
    x = vect.transform(text).toarray()
    p = model.predict(x)
    s = [str(i) for i in p]
    v = int(''.join(s))
    return str(labels[v])


st.title('Email-Spam detection')
st.image('spam.png')
user_in = st.text_input('Input text')
submit = st.button('predict')
if submit:
    user_out = predict([user_in])
    st.text(user_out)

