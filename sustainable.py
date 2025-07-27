import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

df= pd.read_csv('sustainable.csv')
df.drop_duplicates(inplace=True)
df.isnull().sum()
df.dropna(inplace=True)
import re

def clean(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)  # remove punctuation
    text = re.sub(r'\s+', ' ', text)  # remove extra spaces
    return text.strip()

df['Clean_Description'] = df['Description'].apply(clean)
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

def remove_stopwords(text):
    return ' '.join(word for word in text.split() if word not in stop_words)

df['Clean_Description'] = df['Clean_Description'].apply(remove_stopwords)
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()

def lemmatize_text(text):
    return ' '.join([lemmatizer.lemmatize(word) for word in text.split()])

df['Clean_Description'] = df['Clean_Description'].apply(lemmatize_text)
final_df = df[['Clean_Description', 'Label']]
# Assuming 'final_df' contains your cleaned data
vectorizer = TfidfVectorizer(max_features=3000, ngram_range=(1,2))  # unigrams + bigrams
X = vectorizer.fit_transform(final_df['Clean_Description'])
from sklearn.model_selection import train_test_split

y = final_df['Label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluate
acc = model.score(X_test, y_test)
print("Accuracy: ", acc)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# Define model
model = Sequential()
model.add(Dense(512, input_dim=X_train.shape[1], activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(1, activation='sigmoid'))  # Binary output

# Compile
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train with multiple epochs
history = model.fit(X_train.toarray(), y_train, epochs=10, batch_size=32,
                    validation_data=(X_test.toarray(), y_test),
                    callbacks=[EarlyStopping(patience=3, restore_best_weights=True)])
model.save("sustainability_model.h5")
import streamlit as st
import joblib
import numpy as np
from tensorflow.keras.models import load_model

# Load model and vectorizer
model = load_model("sustainability_model.h5")
vectorizer = joblib.load("vectorizer.pkl")

# UI
st.title("🌱 Sustainable Product Classifier")
st.markdown("Classify if a product is **eco-friendly** based on its description. 🌍")

user_input = st.text_area("Enter product description:", "")

if st.button("Check Sustainability"):
    if user_input.strip() != "":
        input_vec = vectorizer.transform([user_input])
        prediction = model.predict(input_vec.toarray())[0][0]

        if prediction >= 0.5:
            st.success("✅ This product is **Eco-Friendly** 💚")
        else:
            st.error("❌ This product is **Not Eco-Friendly** 💔")
    else:
        st.warning("Please enter a description to check.")
