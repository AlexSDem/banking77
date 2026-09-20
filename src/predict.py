"""Example: python src/predict.py 'My card has not arrived yet'."""
import argparse
from pathlib import Path
import joblib

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('text')
    args = p.parse_args()
    model = joblib.load(Path(__file__).resolve().parents[1] / 'results/model_combined.joblib')
    print(model.predict([args.text])[0])
