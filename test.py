import argparse
import re
import nltk
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

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

def load_and_prepare_texts(texts, tokenizer, maxlen):
    cleaned_texts = [text_cleaning(text) for text in texts]
    sequences = tokenizer.texts_to_sequences(cleaned_texts)
    padded_sequences = pad_sequences(sequences, maxlen=maxlen)
    return padded_sequences

def main(model_path, tokenizer_path, maxlen, text):
    # Load the saved model
    model = load_model(model_path)
    
    # Load the tokenizer
    tokenizer = Tokenizer()
    tokenizer.word_index = np.load(tokenizer_path, allow_pickle=True).item()
    
    # Prepare the input text
    input_texts = [text]
    prepared_texts = load_and_prepare_texts(input_texts, tokenizer, maxlen)
    
    # Make a prediction
    prediction = model.predict(prepared_texts)
    
    # Output the result
    result = 'spam' if prediction[0][0] > 0.5 else 'ham'
    print(f"Prediction: {result}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict spam or ham for given text.")
    parser.add_argument('--model_path', type=str, required=True, help="Path to the saved model file (H5).")
    parser.add_argument('--tokenizer_path', type=str, required=True, help="Path to the saved tokenizer file (NPY).")
    parser.add_argument('--maxlen', type=int, required=True, help="The max length used for padding the sequences.")
    parser.add_argument('--text', type=str, required=True, help="Text to classify.")

    args = parser.parse_args()
    main(args.model_path, args.tokenizer_path, args.maxlen, args.text)
