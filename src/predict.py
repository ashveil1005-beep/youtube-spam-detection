import argparse
import joblib

def main(args):
    model = joblib.load(args.model)
    text = args.text
    if not text:
        text = input('Enter comment text: ')
    pred = model.predict([text])[0]
    print('SPAM' if pred == 1 else 'NOT SPAM')
    try:
        prob = model.predict_proba([text])[0]
        idx = list(model.classes_).index(1)
        print(f'Spam probability: {prob[idx]:.3f}')
    except Exception:
        pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='models/spam_classifier.pkl')
    parser.add_argument('--text', type=str, default=None)
    args = parser.parse_args()
    main(args)
