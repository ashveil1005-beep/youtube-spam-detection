# YouTube Spam Detection

## Overview
Machine Learning pipeline to classify YouTube comments as **spam** or **not spam**.
Includes scripts for preprocessing, training, and prediction.

## Quick start
1. Create a Python virtual environment and install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate   # or .\venv\Scripts\activate on Windows
   pip install -r requirements.txt
   ```
2. Put your dataset at `data/youtube_comments.csv`. CSV must contain `comment_text` and `is_spam` (0/1).
3. Train:
   ```bash
   python src/train_model.py --data data/youtube_comments.csv --output models/spam_classifier.pkl
   ```
4. Predict:
   ```bash
   python src/predict.py --model models/spam_classifier.pkl --text "Check out my channel for free gifts!"
   ```

## Project Structure
```
youtube-spam-detection/
├─ data/                # dataset (sample provided)
├─ notebooks/           # optional exploratory notebook
├─ src/                 # python scripts
├─ models/              # saved trained model (ignored from git by default)
├─ requirements.txt
└─ README.md
```

## Notes
- The repository scaffold contains a small sample dataset. Replace it with a larger labeled CSV for real training.
- If your trained model is large (>100 MB), use Git LFS or avoid committing it.
