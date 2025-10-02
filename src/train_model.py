import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
import joblib
from src.preprocess import clean_text

def load_data(path):
    df = pd.read_csv(path)
    # normalize column names
    if 'comment_text' not in df.columns:
        possible = [c for c in df.columns if 'comment' in c.lower() or 'text' in c.lower()]
        if possible:
            df['comment_text'] = df[possible[0]]
        else:
            raise ValueError("Could not find a comment text column. Provide 'comment_text'.")
    if 'is_spam' not in df.columns:
        possible = [c for c in df.columns if 'spam' in c.lower() or 'label' in c.lower()]
        if possible:
            df['is_spam'] = df[possible[0]]
        else:
            raise ValueError("Could not find 'is_spam' column (0/1).")

    df = df.dropna(subset=['comment_text', 'is_spam']).copy()
    df['comment_text'] = df['comment_text'].astype(str).apply(clean_text)
    return df

def main(args):
    df = load_data(args.data)
    X = df['comment_text']
    y = df['is_spam'].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=10000, ngram_range=(1,2), stop_words='english')),
        ('clf', LogisticRegression(max_iter=1000))
    ])

    print('Training model...')
    pipeline.fit(X_train, y_train)

    preds = pipeline.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f'Accuracy on test set: {acc:.4f}')
    print('Classification report:')
    print(classification_report(y_test, preds))

    joblib.dump(pipeline, args.output)
    print(f'Saved trained model to: {args.output}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='data/youtube_comments.csv')
    parser.add_argument('--output', type=str, default='models/spam_classifier.pkl')
    args = parser.parse_args()
    main(args)
