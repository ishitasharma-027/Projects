# import streamlit as st
# import pandas as pd
# from sklearn.linear_model import LogisticRegression
# st.title("Fake news Analysis App")
# df=pd.read_csv('Fake.csv')
# x= df['Text']
# y= df['label']
# model=LogisticRegression()
# model.fit(x,y)
# text=st.text_input('Enter the news')
# pred=model.predict([[text]])
# if st.button('Analyse'):
#     st.write()
import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer

st.title("Fake News Analysis App")

df = pd.read_csv('Fake.csv')

# Assuming 'Text' is the news text and 'label' is the target (0: Original, 1: Fake)
X = df['Text']
y = df['label']

# Vectorize the text data
vectorizer = TfidfVectorizer()
X_vec = vectorizer.fit_transform(X)

# Train the model
model = LogisticRegression()
model.fit(X_vec, y)

# Streamlit input
text = st.text_input('Enter the news')

if st.button('Analyse'):
    if text:
        text_vec = vectorizer.transform([text])
        pred = model.predict(text_vec)[0]
        if pred == 1:
            st.warning('Prediction: Fake News')
        else:
            st.success('Prediction: Original News')
    else:
        st.write('Please enter some news text.')
