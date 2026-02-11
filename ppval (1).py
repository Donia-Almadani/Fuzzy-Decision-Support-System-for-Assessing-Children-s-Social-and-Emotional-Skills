import pandas as pd

# Read fuzzy results with feedback
df = pd.read_excel("fuzzy_results_with_feedback.xlsx")

# Function to classify level
def classify_level(score):
    if score < 4:
        return 'Needs Support'
    elif score < 7:
        return 'Developing'
    else:
        return 'Strong'

df["level"] = df["social_skill_result"].apply(classify_level)

# ---- Extract one example of each case ----
case_strong = df[df["level"] == "Strong"].head(1)
case_developing = df[df["level"] == "Developing"].head(1)
case_needs = df[df["level"] == "Needs Support"].head(1)

cases = {
    "Strong Case": case_strong,
    "Developing Case": case_developing,
    "Needs Support Case": case_needs
}

# ---- Print all cases clearly ----
for name, case in cases.items():
    if case.empty:
        print(f"\n No {name} found in the dataset.")
        continue

    row = case.iloc[0]

    print(f"\n==========================")
    print(f"{name}")
    print(f"==========================")
    print(f"Empathy           : {row['empathy']:.2f}")
    print(f"Cooperation       : {row['cooperation']:.2f}")
    print(f"Emotional Reg     : {row['emotional_reg']:.2f}")
    print(f"Social Belonging  : {row['social_belonging']:.2f}")
    print(f"Fuzzy Output    : {row['social_skill_result']:.3f}")
    print(f"Level           : {row['level']}")
    print(f"Feedback        : {row['feedback']}")
