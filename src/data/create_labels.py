import pandas as pd

# Load cleaned dataset
df = pd.read_csv("data/processed/clean_employee_reviews.csv")

# Convert ratings to sentiment labels
def rating_to_sentiment(rating):
    if rating <= 2:
        return 0      # Negative
    elif rating == 3:
        return 1      # Neutral
    else:
        return 2      # Positive

df["label"] = df["overall-ratings"].apply(rating_to_sentiment)

# Save dataset
df.to_csv("data/processed/labeled_employee_reviews.csv", index=False)

print("=" * 60)
print("Label Distribution")
print("=" * 60)
print(df["label"].value_counts())

print("\nSaved Successfully!")