# Importing essential libraries
import re
import joblib
import nltk
import pandas as pd
from flask import Flask, render_template, request
from nltk.corpus import stopwords
from spylls.hunspell import Dictionary

# Download list of words to be removed if found during pre-processing of the dataset
nltk.download('stopwords')
# Set the language of the stopwords dictionary to English
stop_words = set(stopwords.words('english'))

# Use joblib to load in the pre-trained model
model = joblib.load('model.pkl')

# Create a dictionary
dictionary = Dictionary.from_files('en_US')

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get the user input from the extension
        sent = request.form['message']
        print(sent)

        # Pre-processing the user input
        words = sent.split()
        print(words)
        string = words[:len(words) - 1]
        print(string)
        clean_words = []
        upper = 0
        up = 0
        url_y = 0
        # Set the words to lookout for when searching for URL's
        url = ["http", "https"]
        misspelled = []

        # Loop to iterate and remove stop words
        for i in string:
            w = re.sub(r'[^\w\s]', ' ', i)
            # Add word to file if it is not a stop word
            if w not in stop_words:
                clean_words.append(w)

            # Add misspelled words to the misspelled array
            if not dictionary.lookup(w):
                misspelled.append(w)

            # Check to see if there is any URL tags
            for i in url:
                if i in w:
                    url_y = 1

            # Checking the total number of letters that are in uppercase
            if w.isupper():
                up = up + 1

        missed = len(misspelled)
        total = len(words)

        if total > 0:
            miss = missed / total
            upper = up / total
        else:
            miss = 0
            up = 0

        # Make DataFrame for model
        print(miss)
        print(url_y)
        print(upper)
        print(total)
        input_features = pd.DataFrame([[miss, url_y, upper, total]],
                                      columns=['Percent Misspelled', 'Has URL', 'Fully Uppercase', 'Total Words'],
                                      dtype=float)
        prediction = model.predict(input_features)
        print(input_features.head())

        # Return the result page to show if potential phishing email or not
        return render_template('result.html', prediction=prediction)


if __name__ == '__main__':
    app.run()
