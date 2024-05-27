import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from tensorflow.keras.layers import Bidirectional, Embedding, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import ReduceLROnPlateau

data = pd.read_csv('/content/spam.csv', encoding='latin1')
data.head()
print(data)
data.rename({'label' : 'Label'},axis=1,inplace=True)
data.info
print(data.duplicated().sum())
data.drop_duplicates(inplace = True)
data.shape
data.isnull().sum()
print("Ham texts:")
print(data[data['v1'] == 'ham']['v2'].head())
print("Spams:")
print(data[data['v1'] == 'spam']['v2'].head())

def text_cleaning(text):

    text = re.sub(r'[^\w\s]', '', text)

    text = text.lower()

    text = re.sub(r'\d+','',text)

    stop = stopwords.words('english')
    text = " ".join([word for word in text.split() if word not in stop])

    lemmatizer = WordNetLemmatizer()
    text = " ".join([lemmatizer.lemmatize(word) for word in text.split()])

    text = text.strip()

    return text

nltk.download('wordnet')
nltk.download('stopwords')

X_train, X_test, y_train,y_test = train_test_split(data['v2'], data['v1'], test_size = 0.3, random_state = 48)
X_train = X_train.apply(text_cleaning)
X_test = X_test.apply(text_cleaning)
X_train

# Data tokenization and padding
maxLength = max([len(i) for i in X_train])
tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_train)

X_train = tokenizer.texts_to_sequences(X_train)
X_train = pad_sequences(X_train, maxlen = maxLength)
X_test = tokenizer.texts_to_sequences(X_test)
X_test = pad_sequences(X_test, maxlen = maxLength)

#Fixing class imbalance with SMOTE
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

y_train.value_counts()

# Encoding target variable
encoder = LabelEncoder()
y_train = encoder.fit_transform(y_train)
y_test = encoder.transform(y_test)

# Defining model structure
model = Sequential()

model.add(Embedding(input_dim=len(tokenizer.word_index)+1, output_dim = 100, input_length = maxLength))

model.add(LSTM(units=32, return_sequences = True))
model.add(LSTM(units=32))

model.add(Dense(units=32, activation = 'relu'))
# Adding a Dropout layer, in order to prevent overfitting
model.add(Dropout(rate=0.5))

model.add(Dense(units=1, activation = 'sigmoid'))

model.summary()

# Compiling model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics = ['accuracy'])
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

model_fit=model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test), callbacks=[early_stopping])