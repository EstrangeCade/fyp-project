# Import packages
import re
import pandas as pd
import numpy as np
import joblib
import nltk
from nltk.corpus import stopwords
from spylls.hunspell import Dictionary
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# Download list of words to be removed if found during pre-processing of the dataset
nltk.download('stopwords')
# Set the language of the stopwords dictionary to English
stop_words = set(stopwords.words('english'))

# Set the words to lookout for when searching for URL's
url = ["http", "https"]

# Create a dictionary for the spell check and set it to en_US (due to the majority of messages being sent from the US)
dictionary = Dictionary.from_files('en_US')

# Reading the dataset CSV file
data = open("fraud_email_.csv", 'r', encoding="utf8")

# Open a 2 CSV files called temp and processed
tempFile = open('temp.csv', 'w', encoding="utf8")
classFile = open('processed.csv', 'w', encoding="utf8")

# Set up the CSV attributes
sent = data.readline()
tempFile.write("ID,Text,Class,Percent Misspelled,Has URL,Fully Uppercase,Total Words\n")
classFile.write("Class,Percent Misspelled,Has URL,Fully Uppercase,Total Words\n")

# Set count to 0
count = 0
# Remove newline generated from data.readline()
string = sent[:len(sent)-2]

# Pre-processing the data
# Loop that outputs to the file as text, class
while True:
    URLFound = 0
    misspelled = []
    tempFile.write(str(count) + ",")
    count = count + 1

    # Shift variables to next lines values
    sent = data.readline()
    words = sent.split()
    string = words[:len(words) - 1]
    nums = sent[len(sent) - 2:]
    up = 0
    ress = []
    for sub in nums:
        ress.append(re.sub('\n', '', sub))

    # Loop to iterate and remove stop words
    for i in string:
        w = re.sub(r'[^\w\s]', ' ', i)
        # Add word to file if it is not a stop word
        if w not in stop_words:
            tempFile.write(w + " ")

        # Add misspelled words to the misspelled array
        if not dictionary.lookup(w):
            misspelled.append(w)

        # Check to see if there is any URL tags
        for i in url:
            if i in w:
                URLFound = 1

        # Checking the total number of letters that are in uppercase
        if w.isupper():
            up = up + 1

    missed = len(misspelled)
    total = len(words)

    if total > 0:
        # Calculate the average of misspelled words
        miss = missed / total
        # Calculate the average of upper case word
        upper = up / total
    else:
        miss = 0
        up = 0

    if not words:
        break

    tempFile.write("," + ress[0] + "," + str(miss) + "," + str(URLFound) +
                   "," + str(upper) + "," + str(total) + "\n")
    classFile.write(ress[0] + "," + str(miss) + "," + str(URLFound)
                    + "," + str(upper) + "," + str(total) + "\n")

# Close the files
tempFile.close()
classFile.close()
data.close()
print("Files Processed.")

# Read the CSV file
df = pd.read_csv('processed.csv', header=0, engine='python')

# Define X and Y features
X = df.drop('Class', axis=1)
Y = df['Class']
# Prepare configuration for cross validation test harness
# Set the seed value to 7, this will ensure that the results are reproducible
seed = 7
# Setup the model classifiers to test on
models = [('KNN', KNeighborsClassifier()), ('CART', DecisionTreeClassifier()), ('NB', GaussianNB()), ('SVM', SVC())]
# Measure each model in turn
results = []
names = []
scoring = 'accuracy'
models_dict = {}
for name, model in models:
    kfold = model_selection.KFold(n_splits=10, shuffle=True, random_state=seed)
    cv_results = model_selection.cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)
    models_dict[name] = {"binary": model,"avg_performance": cv_results.mean(),"std_performance": cv_results.std()}

# Set the best model to the model with teh highest accuracy
best_model_dict = list(models_dict.values())[np.argmax([x["avg_performance"] for x in list(models_dict.values())])]

# Fit the model so that it can be used with a different dataset later
best_model = best_model_dict["binary"].fit(X,Y)

# Boxplot algorithm comparison to visulise which classifier works best
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
# Set grid parameters to a 1x1 grid with first subplot
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()

# Show which classifier performed the best
print("From the results gathered we can see that ", best_model, "had the highest accuracy overall")

# Saving the best model for usage in the Heroku app
joblib.dump(best_model, 'model.pkl')
print("Model Saved.")
