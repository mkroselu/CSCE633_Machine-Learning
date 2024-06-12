# HW5 - SVM

## (a) Data pre-processing

Pre-process the data via removing the headers of each document (e.g., lines that start with `From:` and `Subject:`), eliminating any additional handles and URLs, and converting all words to lower-case. Note: You can use the nltk library from here: https://www.nltk.org/ to remove stop words. Regular expression may be helpful.


```python
import os
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import gensim
from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
```


```python
# Load stop words
nltk.download('stopwords')
nltk.download('punkt')
stop_words = set(stopwords.words('english'))

def preprocess(text):
    # Remove headers
    text = re.sub(r'^From:.*\n', '', text, flags=re.MULTILINE)
    text = re.sub(r'^Subject:.*\n', '', text, flags=re.MULTILINE)
    
    # Remove additional handles and URLs
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'http\S+', '', text)
    
    # Convert to lower case
    text = text.lower()
    
    # Remove stop words
    text_tokens = word_tokenize(text)
    tokens_without_sw = [token for token in text_tokens if not token in stop_words]
    filtered_text = ' '.join(tokens_without_sw)
    
    # Split the data based on the "newsgroup :" pattern
    documents = re.split(r'newsgroup : ', filtered_text, flags=re.MULTILINE)[1:]
    
    # Remove any leading or trailing white space from each document
    documents = [doc.strip() for doc in documents]
    
    return documents
```

    [nltk_data] Downloading package stopwords to C:\Users\MEI-KUEI
    [nltk_data]     LU\AppData\Roaming\nltk_data...
    [nltk_data]   Package stopwords is already up-to-date!
    [nltk_data] Downloading package punkt to C:\Users\MEI-KUEI
    [nltk_data]     LU\AppData\Roaming\nltk_data...
    [nltk_data]   Package punkt is already up-to-date!
    


```python
# read data 

with open('rec.autos.txt', errors='surrogateescape') as f:
    text = f.read()
auto = preprocess(text)

with open('sci.space.txt', errors='surrogateescape') as f:
    text = f.read()
space = preprocess(text)

with open('sci.med.txt', errors='surrogateescape') as f:
    text = f.read()
med = preprocess(text)

with open('comp.sys.mac.hardware.txt', errors='surrogateescape') as f:
    text = f.read()
hardware = preprocess(text)
```

## (b) Feature extraction via bag-of-words representation

Extract the term frequency-inverse document frequency (TF-IDF) for every document. TF-IDF evaluates the relevance of a word is to a given document in a collection of documents. It is computed via multiplying the number of occurrences of a word in a document and the inverse document frequency of the word across a set of documents. Note: You can use sklearn.feature_extraction.text.TfidfVectorizer.

Top 25 words in transportation document


```python
tfIdfVectorizer = TfidfVectorizer(use_idf=True)
tfIdf = tfIdfVectorizer.fit_transform(auto)
df = pd.DataFrame(tfIdf[0].T.todense(), index=tfIdfVectorizer.get_feature_names_out(), columns=["TF-IDF"])
df = df.sort_values('TF-IDF', ascending=False)
print (df.head(25))
```

                   TF-IDF
    profit       0.411988
    saturn       0.377378
    dealer       0.334974
    money        0.240662
    reducing     0.185197
    saving       0.172601
    2k           0.157370
    1000         0.145586
    car          0.124895
    expenses     0.123465
    minimize     0.123465
    competitors  0.123465
    fred         0.104913
    of           0.103054
    pocket       0.103054
    priced       0.103054
    reduce       0.099815
    believe      0.098875
    out          0.097057
    average      0.091553
    save         0.091553
    would        0.087814
    price        0.087041
    class        0.084502
    prices       0.083862
    

Top 25 words in space document


```python
tfIdfVectorizer = TfidfVectorizer(use_idf=True)
tfIdf = tfIdfVectorizer.fit_transform(space)
df = pd.DataFrame(tfIdf[0].T.todense(), index=tfIdfVectorizer.get_feature_names_out(), columns=["TF-IDF"])
df = df.sort_values('TF-IDF', ascending=False)
print (df.head(25))
```

                    TF-IDF
    existence     0.308429
    curved        0.271126
    dong          0.271126
    bruce         0.201005
    undefined     0.197374
    unobservable  0.197374
    observable    0.197374
    synonymous    0.197374
    tesla         0.175075
    physics       0.171151
    properties    0.159618
    theory        0.127572
    matter        0.117246
    tom           0.112934
    59497         0.105910
    unless        0.104162
    randale       0.098687
    filling       0.098687
    knell         0.098687
    refuse        0.098687
    bass          0.098687
    constructs    0.098687
    inferred      0.098687
    crb           0.098687
    curvature     0.093929
    

Top 25 words in medicine document


```python
tfIdfVectorizer = TfidfVectorizer(use_idf=True)
tfIdf = tfIdfVectorizer.fit_transform(med)
df = pd.DataFrame(tfIdf[0].T.todense(), index=tfIdfVectorizer.get_feature_names_out(), columns=["TF-IDF"])
df = df.sort_values('TF-IDF', ascending=False)
print (df.head(25))
```

                        TF-IDF
    gaucher           0.451486
    disease           0.263857
    genzyme           0.225743
    ceredase          0.225743
    costs             0.150072
    macrophages       0.112872
    brittle           0.112872
    enyzme            0.112872
    380               0.112872
    hieght            0.112872
    57110             0.112872
    osteopporosis     0.112872
    justifyably       0.112872
    biotech           0.112872
    biggy             0.112872
    glucocerebroside  0.112872
    netlanders        0.112872
    mutation          0.105179
    remarkable        0.105179
    relying           0.105179
    replacement       0.105179
    spleen            0.100111
    yr                0.100111
    enlarged          0.096326
    researched        0.096326
    

Top 25 words in hardware document


```python
tfIdfVectorizer = TfidfVectorizer(use_idf=True)
tfIdf = tfIdfVectorizer.fit_transform(hardware)
df = pd.DataFrame(tfIdf[0].T.todense(), index=tfIdfVectorizer.get_feature_names_out(), columns=["TF-IDF"])
df = df.sort_values('TF-IDF', ascending=False)
print (df.head(25))
```

                  TF-IDF
    regions     0.384205
    nutek       0.292488
    structure   0.281377
    patent      0.247461
    region      0.230523
    patents     0.230523
    apple       0.199586
    clone       0.198840
    mapinfo     0.164974
    data        0.157289
    pict        0.153682
    stored      0.140689
    internal    0.117967
    adb         0.103229
    files       0.100675
    opinions    0.094417
    believe     0.084370
    vice        0.082487
    implement   0.082487
    engineered  0.082487
    alverson    0.082487
    50418       0.082487
    patented    0.082487
    versa       0.082487
    dunno       0.082487
    

## (c) Feature extraction via word2vec representation 

Extract the word2vec representation for each document. Word2vec leverages a neural network and a large language corpus to output a vector representation of a document via learning word similarities and capturing the contextual meaning of the words. Note: You can use the word2vec model from the Genism library from here: https://pypi.org/project/gensim/.


```python
all_docs = auto + space + med + hardware
```


```python
# Train a Word2Vec model on your corpus of documents
model = Word2Vec(all_docs, vector_size=300, window=5, min_count=5, workers=4)

# Obtain the Word2Vec representation for each document in your corpus
doc_vectors = []
for doc in all_docs:
    words = doc.split()
    vectors = [model.wv.get_vector(word) for word in words if word in model.wv.key_to_index]
    if vectors:
        doc_vector = sum(vectors) / len(vectors)
    else:
        doc_vector = [0] * 300  # if no words in vocab, use zero vector
    doc_vectors.append(doc_vector)
```

## (d) Document classification with SVMs

Use a SVM to classify a document among the four considered topics using: (1) the TF-IDF features; and (2) the word2vec features.
Randomly split the data into a training (80%), validation (10%), and testing (10%) set. Using the training and the validation sets, experiment with different values of misclassifjication cost C via hyper-parameter tuning. After finding the misclassification cost C that provides the best accuracy on the validation set for each type of feature, report the accuracy of that model on the test set. Also present the confusion matrix on the test set. Compare and contrast the results with the two types of features.


```python
print(len(auto), len(space), len(med), len(hardware))
```

    1980 1974 1984 1922
    


```python
labels = [0] * 1980 + [1] * 1974 + [2] * 1984 + [3] * 1922
tfidf = tfIdfVectorizer.fit_transform(all_docs)
```


```python
df = pd.DataFrame(list(zip(all_docs, labels, tfidf, doc_vectors)), columns =['Text', 'Label', 'TF-IDF', 'word2vec']) 
df
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Text</th>
      <th>Label</th>
      <th>TF-IDF</th>
      <th>word2vec</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>rec.autos document_id : 101551 article ( fred ...</td>
      <td>0</td>
      <td>(0, 16702)\t0.04165758478647919\n  (0, 85)\t...</td>
      <td>[0.0876568, 0.11373603, -0.037397083, -0.15062...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>rec.autos document_id : 101552 article ( gary ...</td>
      <td>0</td>
      <td>(0, 37747)\t0.1357359477746361\n  (0, 16826)...</td>
      <td>[-0.10963901, 0.28646937, 0.01883296, -0.02246...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>rec.autos document_id : 101553 thanks responde...</td>
      <td>0</td>
      <td>(0, 36958)\t0.21527382825484603\n  (0, 26831...</td>
      <td>[-0.36591262, 0.13937807, 0.19431634, -0.50086...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>rec.autos document_id : 101554 subject says . ...</td>
      <td>0</td>
      <td>(0, 22975)\t0.15565840000429534\n  (0, 20531...</td>
      <td>[-0.117455155, 0.28166255, 0.24468397, -0.0938...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>rec.autos document_id : 101555 ( stephen wolfs...</td>
      <td>0</td>
      <td>(0, 10313)\t0.048182498373623046\n  (0, 1132...</td>
      <td>[-0.10389984, 0.24735743, 0.06250552, -0.04224...</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>7855</th>
      <td>comp.sys.mac.hardware document_id : 52442 frie...</td>
      <td>3</td>
      <td>(0, 251)\t0.15383515462477623\n  (0, 20432)\...</td>
      <td>[0.022504378, 0.16801114, 0.25965986, -0.00103...</td>
    </tr>
    <tr>
      <th>7856</th>
      <td>comp.sys.mac.hardware document_id : 52443 ( ja...</td>
      <td>3</td>
      <td>(0, 16404)\t0.12497859358476575\n  (0, 4390)...</td>
      <td>[0.0038019875, 0.27972633, -0.03631023, 0.0186...</td>
    </tr>
    <tr>
      <th>7857</th>
      <td>comp.sys.mac.hardware document_id : 52444 'm t...</td>
      <td>3</td>
      <td>(0, 4391)\t0.3357299259522026\n  (0, 37428)\...</td>
      <td>[-0.084446, 0.20517752, 0.29311243, -0.1522957...</td>
    </tr>
    <tr>
      <th>7858</th>
      <td>comp.sys.mac.hardware document_id : 52445 nort...</td>
      <td>3</td>
      <td>(0, 12546)\t0.19579946671388337\n  (0, 27401...</td>
      <td>[0.64342797, -0.06546467, 0.6932751, 0.6785109...</td>
    </tr>
    <tr>
      <th>7859</th>
      <td>comp.sys.mac.hardware document_id : 52446 ques...</td>
      <td>3</td>
      <td>(0, 36247)\t0.11441031173412056\n  (0, 24146...</td>
      <td>[-0.031914517, 0.2543327, 0.31849527, 0.119638...</td>
    </tr>
  </tbody>
</table>
<p>7860 rows × 4 columns</p>
</div>



Train with TF-IDF feature


```python
# In the first step we will split the data in training and remaining dataset
X_train, X_rem, y_train, y_rem = train_test_split(tfidf, labels, train_size=0.8)
X_valid, X_test, y_valid, y_test = train_test_split(X_rem, y_rem, test_size=0.5)

print(X_train.shape), print(np.array(y_train).shape)
print(X_valid.shape), print(np.array(y_valid).shape)
print(X_test.shape), print(np.array(y_test).shape)
```

    (6288, 41870)
    (6288,)
    (786, 41870)
    (786,)
    (786, 41870)
    (786,)
    




    (None, None)




```python
# Define the range of C values to try
c_values = [0.01, 0.1, 1, 10, 100]

# Train and evaluate the model for each value of C
best_accuracy = 0
best_c = None

for c in c_values:
    # Train the SVM classifier with the current value of C
    clf = SVC(C=c)
    clf.fit(X_train, y_train)

    # Evaluate the model on the validation set and record its accuracy
    accuracy = clf.score(X_valid, y_valid)

    # Update the best accuracy and best C if the current model is better
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_c = c

# Train the final model with the selected value of C
final_clf = SVC(C = best_c)
final_clf.fit(X_train, y_train)

# Evaluate the final model on the test set
test_accuracy = final_clf.score(X_test, y_test)
y_pred = final_clf.predict(X_test)
```


```python
# Print the results
print('Best C value: {}'.format(best_c))
print('Validation accuracy: {:.3f}'.format(best_accuracy))
print('Test accuracy: {:.3f}'.format(test_accuracy))
print('Confusion matrix on the test set:')
print(confusion_matrix(y_test, y_pred))
```

    Best C value: 1
    Validation accuracy: 1.000
    Test accuracy: 0.999
    Confusion matrix on the test set:
    [[196   0   0   0]
     [  0 206   0   0]
     [  0   1 216   0]
     [  0   0   0 167]]
    

Train with word2vec feature


```python
# In the first step we will split the data in training and remaining dataset
X_train, X_rem, y_train, y_rem = train_test_split(doc_vectors, labels, train_size=0.8)
X_valid, X_test, y_valid, y_test = train_test_split(X_rem, y_rem, test_size=0.5)

print(np.array(X_train).shape), print(np.array(y_train).shape)
print(np.array(X_valid).shape), print(np.array(y_valid).shape)
print(np.array(X_test).shape), print(np.array(y_test).shape)
```

    (6288, 300)
    (6288,)
    (786, 300)
    (786,)
    (786, 300)
    (786,)
    




    (None, None)




```python
# Define the range of C values to try
c_values = [0.01, 0.1, 1, 10, 100]

# Train and evaluate the model for each value of C
best_accuracy = 0
best_c = None

for c in c_values:
    # Train the SVM classifier with the current value of C
    clf = SVC(C=c)
    clf.fit(X_train, y_train)

    # Evaluate the model on the validation set and record its accuracy
    accuracy = clf.score(X_valid, y_valid)

    # Update the best accuracy and best C if the current model is better
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_c = c

# Train the final model with the selected value of C
final_clf = SVC(C = best_c)
final_clf.fit(X_train, y_train)

# Evaluate the final model on the test set
test_accuracy = final_clf.score(X_test, y_test)
y_pred = final_clf.predict(X_test)
```


```python
# Print the results
print('Best C value: {}'.format(best_c))
print('Validation accuracy: {:.3f}'.format(best_accuracy))
print('Test accuracy: {:.3f}'.format(test_accuracy))
print('Confusion matrix on the test set:')
print(confusion_matrix(y_test, y_pred))
```

    Best C value: 100
    Validation accuracy: 0.562
    Test accuracy: 0.555
    Confusion matrix on the test set:
    [[121  24  34  31]
     [ 28  95  55  14]
     [ 27  31 109  21]
     [ 26  23  36 111]]
    

**Ans**: The misclassification cost **C** providing the best accuracy on the validation set for TF-IDF and Word2Vec feature are 1 and 100 respectively. The **validation accuracy** for TF-IDF and Word2Vec feature are 1 and 0.562 respectively. After finding the best C, it was used in training model, tested on the test set. The **accuracy on the test set** for TF-IDF and Word2Vec feature are 0.999 and 0.555 respectively.

## (e) Obtaining additional insights with the SVMs

For the best performing SVM (i.e., among the different C values and feature types), report the number of the training samples that were: (1) misclassified; and (2) within the margin (i.e., correctly classified).


```python
# Use the trained SVM classifier to predict the labels for the training data
y_pred = final_clf.predict(X_train)

# Compute the number of misclassified and correctly classified samples
misclassified = (y_pred != y_train).sum()
within_margin = 6288 - misclassified

# Print the results 
print("Number of misclassified training samples: ", misclassified)
print("Number of training samples within margin: ", within_margin)
```

    Number of misclassified training samples:  1
    Number of training samples within margin:  6287
    


```python
print(confusion_matrix(y_train, y_pred))
```

    [[1580    0    0    0]
     [   0 1626    0    0]
     [   0    1 1567    0]
     [   0    0    0 1514]]
    


```python

```
