# **Movie Review Sentiment Analysis**
This project implements a sentiment analysis tool that uses the NLTK movie reviews dataset. It predicts whether a movie review is positive or negative using a Naive Bayes classifier and machine learning techniques.

## **Project Overview**
The goal of this project is to classify movie reviews as either positive or negative based on the text content. The project demonstrates:

* Loading and preprocessing text data

* Using a Naive Bayes classifier for sentiment classification

* Evaluating the model's performance

* Visualizing the distribution of sentiments and the most common words in the reviews

* Generating a confusion matrix to assess model accuracy

## **Key Features**
**Sentiment Prediction** : Predicts whether a movie review is positive or negative.

**Model Evaluation** : Displays classification reports and accuracy metrics.

**Visualization** :

   * Sentiment distribution (positive vs. negative reviews).

   * Top 20 most frequent words in both positive and negative reviews.

   * Confusion matrix to evaluate model performance.

**Easy to Extend**: This project can be further extended by experimenting with different classifiers, adding more data preprocessing, or using other text vectorization techniques.

## **Libraries Used**
* **nltk** : Natural Language Toolkit for text processing and handling the movie reviews dataset.

* **pandas** : For managing and manipulating data.

* **scikit-learn** : For building the machine learning model and evaluating performance.

* **seaborn & matplotlib** : For visualizations and plotting the data.

## **Setup Instructions**
**1. Install Dependencies**
To run this project, you need to install the required libraries. You can install them using pip:
```bash
pip install nltk pandas scikit-learn seaborn matplotlib
```

**2. Download NLTK Data**
Run the following code in your Python script or Jupyter notebook to download the movie_reviews dataset from NLTK:
```bash
import nltk
nltk.download('movie_reviews')
```

**3. Run the Project**
The project is composed of several steps:

* **Load and Preprocess Data** : The movie reviews are loaded from NLTK's corpus and labeled as positive (pos) or negative (neg).

* **Vectorization** : The reviews are converted into a numerical format using CountVectorizer to create a matrix of token counts.

* **Train Naive Bayes Classifier** : The Multinomial Naive Bayes model is trained on the dataset.

* **Model Evaluation** : The model is tested, and performance metrics (accuracy and classification report) are displayed.

* **Custom Sentiment Prediction** : Custom reviews can be passed into the model to predict sentiment.

* **Visualization** :

     * The sentiment distribution (positive vs. negative reviews) is visualized using a count plot.

     * The top 20 most frequent words in both positive and negative reviews are plotted.

     * A confusion matrix is generated to display the model's performance.

```bash
# Example Usage
print(predict_sentiment("I absolutely loved this movie! It was fantastic."))
print(predict_sentiment("It was a terrible film. I hated it."))
print(predict_sentiment("The movie was okay, nothing special."))
```
## **Project Breakdown**
**Step 1: Data Preprocessing and Vectorization**
We load the NLTK movie reviews dataset and preprocess the text. We use the CountVectorizer to convert the text into numerical data (word counts), which is suitable for training a machine learning model.

**Step 2: Model Training**
We train a Multinomial Naive Bayes model using the training dataset. This model is simple and works well for text classification tasks.

**Step 3: Model Evaluation**
After training the model, we evaluate it by predicting the sentiment on the test dataset. We display the accuracy and generate a classification report that includes metrics such as precision, recall, and F1-score.

**Step 4: Predict Custom Reviews**
We also provide a function to predict the sentiment of custom movie reviews. You can input any review and see if it is classified as positive or negative.

**Step 5: Visualizations**
The project includes several visualizations:

* **Sentiment Distribution** : A count plot showing the distribution of positive and negative reviews.

* **Most Frequent Words** : The top 20 most common words used in positive and negative reviews.

* **Confusion Matrix**: A confusion matrix that shows how well the model performs by comparing the true labels to the predicted labels.
**Example Output**
```bash
Accuracy and Classification Report
plaintext
Copy
Edit
Accuracy: 0.80
Classification Report:
              precision    recall  f1-score   support
         neg       0.80      0.80      0.80       199
         pos       0.80      0.80      0.80       201
   accuracy                           0.80       400
  macro avg       0.80      0.80      0.80       400
weighted avg       0.80      0.80      0.80       400
```
**Sample Custom Review Predictions**
```bash
Predicted Sentiment for "I absolutely loved this movie! It was fantastic.": Positive
Predicted Sentiment for "It was a terrible film. I hated it.": Negative
Predicted Sentiment for "The movie was okay, nothing special.": Negative
```
## **Sentiment Distribution Plot**
A bar plot showing the count of positive vs. negative reviews.

## **Top 20 Words in Positive and Negative Reviews**
Two separate bar plots show the most frequent words in positive and negative reviews.

## **Confusion Matrix**
A confusion matrix with counts for True Positives, True Negatives, False Positives, and False Negatives.

## **Conclusion**
This sentiment analysis tool is a simple yet effective way to classify movie reviews into positive and negative sentiments. The project is fully scalable and can be extended with more advanced models, additional preprocessing, or a larger dataset. By visualizing the most common words and the model's performance through a confusion matrix, you gain deeper insights into how the model works.

## **Future Improvements** 
* Use a more sophisticated text vectorization method such as TF-IDF (Term Frequency-Inverse Document Frequency).

* Experiment with other classifiers like Logistic Regression, SVM, or Deep Learning models.

* Add more advanced preprocessing steps (like stemming, lemmatization, or removing stop words).

* Deploy the model as a web application for real-time sentiment analysis.# Movie Review Sentiment Analysis

* This project implements a basic sentiment analysis tool using the NLTK movie reviews dataset.
  
* It predicts whether a movie review is positive or negative using a machine learning model.

