import pandas as pd
import matplotlib.pyplot as plt

# Read Data
df = pd.read_excel("fuzzy_results_final.xlsx")

# Print Columns
print("Columns in file:")
print(df.columns)

# -----------------------------
# Basic stats for social_skill_result

scores = df["social_skill_result"]

print("\nBasic statistics for social_skill_result:")
print(scores.describe())

# -----------------------------
# Create level column (Needs Support / Developing / Strong)

def classify_level(score):
    if score < 4:
        return 'Needs Support'
    elif score < 7:
        return 'Developing'
    else:
        return 'Strong'

df["social_skill_level"] = df["social_skill_result"].apply(classify_level)

# -----------------------------
# Number of students in each level

levels_order = ["Needs Support", "Developing", "Strong"]

level_counts = df["social_skill_level"].value_counts().reindex(levels_order, fill_value=0)
total = level_counts.sum()
percentages = (level_counts / total) * 100

print("\nCounts per level:")
print(level_counts)

print("\nPercentages per level:")
print(percentages.round(2))

# -----------------------------
#  Bar Chart

plt.figure()
plt.bar(levels_order, level_counts)
plt.title("Number of students in each social skill level")
plt.xlabel("Social skill level")
plt.ylabel("Number of students")
plt.tight_layout()
plt.show()

# -----------------------------
#  Pie Chart

plt.figure()
plt.pie(percentages, labels=levels_order, autopct="%.2f%%", startangle=90)
plt.title("Distribution of social skill levels")
plt.axis("equal")
plt.tight_layout()
plt.show()


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
    print(f"School Belonging  : {row['school_Belonging']:.2f}")
    print(f"Fuzzy Output    : {row['social_skill_result']:.3f}")
    print(f"Level           : {row['level']}")
    print(f"Feedback        : {row['feedback']}")