"""
Demo: SHAP explanations for FinBERT sentiment classification.

Model: ProsusAI/finbert
Shows word-level token contributions for positive and negative financial texts.

Run:
    python main.py
"""

import shap
from transformers import pipeline

MODEL_NAME = "ProsusAI/finbert"

print(f"Loading model: {MODEL_NAME}...")
pipe = pipeline("text-classification", model=MODEL_NAME, top_k=None)

print("Initializing SHAP explainer...")
explainer = shap.Explainer(pipe)

# Positive sentiment
text_positive = "The company reported record profits and strong revenue growth this quarter."
print(f"\n[Positive Example]\n{text_positive}")

pred_positive = pipe(text_positive)
print(f"Prediction: {pred_positive[0]}")

shap_values_positive = explainer([text_positive])

print("\nToken contributions (positive example):")
tokens = shap_values_positive.data[0]
label_names = [c["label"] for c in pred_positive[0]]
for i, label in enumerate(label_names):
    print(f"\n  [{label}]")
    for token, value in zip(tokens, shap_values_positive.values[0, :, i]):
        print(f"    {token:20s}  {value:+.4f}")

# Negative sentiment

text_negative = "The stock plummeted after the company missed earnings and cut its dividend."
print(f"\n[Negative Example]\n{text_negative}")

pred_negative = pipe(text_negative)
print(f"Prediction: {pred_negative[0]}")

shap_values_negative = explainer([text_negative])

print("\nToken contributions (negative example):")
tokens = shap_values_negative.data[0]
for i, label in enumerate(label_names):
    print(f"\n  [{label}]")
    for token, value in zip(tokens, shap_values_negative.values[0, :, i]):
        print(f"    {token:20s}  {value:+.4f}")


print("\nSaving SHAP reports...")

with open("report_positive.html", "w") as f:
    f.write(shap.plots.text(shap_values_positive[0], display=False))

with open("report_negative.html", "w") as f:
    f.write(shap.plots.text(shap_values_negative[0], display=False))

print("Reports saved: report_positive.html, report_negative.html")
