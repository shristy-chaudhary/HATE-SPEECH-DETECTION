# HATE-SPEECH-DETECTION
This project is a machine learning model for detecting hate speech in text. The model uses a combination of natural language processing techniques and machine learning algorithms to analyze text and determine if it contains hate speech. The model is trained on a dataset of labeled text (e.g. negative speech or positive speech or neutral speech) and then used to classify new text. The code uses the pandas library to read a CSV file containing labeled text, and then uses the sklearn library to train a model using the Naive Bayes algorithm. The performance of the model is evaluated using the accuracy metric, which is a value between 0 and 1 representing the percentage of test examples that the model correctly classified as negative( hate speech) or positive(non-hate speech).The code also takes input from the user in the form of text, and then predicts the label of the text whether it is negative(hate speech) or positive(non-hate speech). The goal of this code is to develop a machine learning model that can accurately identify hate speech in text, which can be useful for a variety of applications such as social media monitoring and content moderation.

Methodology(Step By Step)
The first step is to import the necessaryl ibraries such as pandas for data manipulation, matplotlib for data visualization , CountVectorizer and MultinomialNB from sklearn for machine learning , nltk for text preprocessing and use to download the stopwords corpus.

Next, we use pandas library to read a CSV file from a URL or local file and convert it into a pandas DataFrame. This dataset contains labeled text(hate speech or non- hate speech) which will be used to train the model.

Preprocess the text data by removing stop words and performing stemming using the “preprocess_text” function.

Print the columns of the dataframe and the first few rows to check the data.

Plot a pie chart to visualize the distribution of sentiments in the dataset.

After loading the dataset, we split the dataset into training and test sets. The training set is used to train the model and the test set is used to evaluate the performance of the model. We use 80% data for training and 20% for testing.

Next, we use the CountVectorizer to create a vocabulary of words from the training set and then convert the text into numerical form using the fit_transform method.

We use the MultinomialNB() class to create an instance of the Naive Bayes model. It's a probabilistic classifier based on Bayes' theorem with an assumption of independence between predictors.

We train the model using the fit method and passing the vectorized text and label of the text and the trained model is also used to calculate the accuracy.

Now, we take input from the user in the form of text and use the transform method to convert the text into numerical form.

We use the predict method to make predictions on the input text and pass the vectorized text as input.

Finally, we print the label of the text whether it is negative (hate speech) or positive ( non-hate speech).

The main goal of this code is to train a machine learning model to detect hate speech in text and then use this model to predict the label of new text. It uses a combination of natural language processing techniques and machine learning algorithms to analyze text and determine if it contains hate speech.

![image](https://github.com/shristy-chaudhary/HATE-SPEECH-DETECTION/assets/110960844/2631dafd-5693-43d8-ba26-511644a7637f)

Results:
The code will output the label of the input text whether it is negative(hate speech) or positive( non-hate speech). The accuracy of the model will be evaluated on the test set using the accuracy_score method and passing the true labels and predicted labels. The accuracy score will be a value between 0 and 1, representing the percentage of test examples that were correctly classified.

![image](https://github.com/shristy-chaudhary/HATE-SPEECH-DETECTION/assets/110960844/c568360b-3039-46b9-8508-204751792235)

Conclusion:
The code uses a combination of natural language processing techniques and machine learning algorithms to detect hate speech in text. The model is trained on a dataset of labeled text and is able to classify new text as hate speech or non-hate speech. The performance of the model depends on the specific dataset and the accuracy of the model can be evaluated using the accuracy_score method. The code takes input text as input from the user and then predict the output as hate speech or non hate speech with showing accuracy_score.

