# Twitter Sentiment Analysis

This project performs sentiment analysis on Twitter data using Natural Language Processing (NLP) techniques and machine learning. It analyzes the sentiments of tweets and visualizes the results.

## Requirements
- Python 3.8
- Conda (optional)
- Streamlit
- Matplotlib
- Numpy
- Pandas
- Seaborn
- NLTK
- TextBlob
- Scikit-learn
- Wordcloud

## Project Structure

- `sentimental.py`: Main Streamlit application file.
- `vaccination_tweets.csv`: Sample dataset containing Twitter data.

## Usage

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/Rajeshkumar-14/Twitter-Sentimental-Analysis.git
   cd twitter-sentiment-analysis
   ```
2. **Install Dependencies:**

  ```bash
  pip install streamlit matplotlib numpy pandas seaborn nltk textblob scikit-learn wordcloud
```
3. **Run the Streamlit application:**
  ```bash
  streamlit run sentimental.py
  ```
4. **Open the provided URL in your browser.**

## Features
- Data preprocessing using NLTK and TextBlob.
- Visualization of the distribution of sentiments.
- Word clouds for positive, negative, and neutral tweets.
- Training a sentiment analysis model using logistic regression.
- Model evaluation with accuracy score and confusion matrix.
- Saving and loading the model using Pickle.
- Additional model: Support Vector Machine (SVM) with Grid Search.

## Notes
- The project uses the vaccination_tweets.csv dataset, and you can replace it with your own dataset.
- Ensure that you have the required Python packages installed before running the application.

**Feel free to explore and analyze Twitter sentiments using this Streamlit application!**
