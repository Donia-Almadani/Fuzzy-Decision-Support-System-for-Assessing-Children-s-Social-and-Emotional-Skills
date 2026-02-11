import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt
import pandas as pd

# ----- Load dataset -----
data = pd.read_excel("INT_01_ST_SELSkills (1).xlsx")

# Select key features
subset = data[['Empathy_Mean', 'Cooperation_Mean', 'EmotionalReg_Mean', 'School Belonging']].dropna()

# Normalize to 0–10 scale
subset = subset.apply(lambda x: 10 * (x - x.min()) / (x.max() - x.min()))

# Rename columns for easier handling
subset = subset.rename(columns={
    'Empathy_Mean': 'empathy',
    'Cooperation_Mean': 'cooperation',
    'EmotionalReg_Mean': 'emotional_reg',
    'School Belonging': 'school_Belonging'
})

# ----- Define fuzzy variables -----
empathy = ctrl.Antecedent(np.arange(0, 11, 1), 'empathy')
cooperation = ctrl.Antecedent(np.arange(0, 11, 1), 'cooperation')
emotional_reg = ctrl.Antecedent(np.arange(0, 11, 1), 'emotional_reg')
school_Belonging = ctrl.Antecedent(np.arange(0, 11, 1), 'school_Belonging')
social_skill = ctrl.Consequent(np.arange(0, 11, 1), 'social_skill')

# Membership functions for inputs
for var in [empathy, cooperation, emotional_reg, school_Belonging]:
    var['low'] = fuzz.trimf(var.universe, [0, 0, 5])
    var['medium'] = fuzz.trimf(var.universe, [2, 5, 8])
    var['high'] = fuzz.trimf(var.universe, [5, 10, 10])

# Membership functions for output
social_skill['needs_support'] = fuzz.trimf(social_skill.universe, [0, 0, 4])
social_skill['developing'] = fuzz.trimf(social_skill.universe, [3, 5, 7])
social_skill['strong'] = fuzz.trimf(social_skill.universe, [6, 10, 10])


# ----- Define fuzzy rules -----

# =========================================================
# STRONG
# =========================================================

# STRONG
rule1  = ctrl.Rule(empathy['high'] & cooperation['high'], social_skill['strong'])
rule2  = ctrl.Rule(school_Belonging['high'] & emotional_reg['high'], social_skill['strong'])
rule3  = ctrl.Rule(emotional_reg['high'] & empathy['medium'], social_skill['strong'])
rule4  = ctrl.Rule(emotional_reg['high'] & cooperation['medium'], social_skill['strong'])
rule5  = ctrl.Rule(school_Belonging['high'] & empathy['high'], social_skill['strong'])

# DEVELOPING
rule6  = ctrl.Rule(empathy['medium'] & emotional_reg['medium'], social_skill['developing'])
rule7  = ctrl.Rule(school_Belonging['low'] & empathy['medium'], social_skill['developing'])
rule8  = ctrl.Rule(cooperation['medium'] & emotional_reg['medium'], social_skill['developing'])
rule9  = ctrl.Rule(empathy['medium'] & cooperation['medium'], social_skill['developing'])
rule10 = ctrl.Rule(school_Belonging['medium'] & emotional_reg['medium'], social_skill['developing'])

# MIXED CASE → Developing
rule11 = ctrl.Rule(empathy['high'] & cooperation['low'], social_skill['developing'])
rule12 = ctrl.Rule(empathy['low'] & cooperation['high'], social_skill['developing'])

# NEEDS SUPPORT
rule13 = ctrl.Rule(empathy['low'] & cooperation['low'], social_skill['needs_support'])
rule14 = ctrl.Rule(emotional_reg['low'] & empathy['low'], social_skill['needs_support'])
rule15 = ctrl.Rule(emotional_reg['low'] & cooperation['low'], social_skill['needs_support'])

# Build fuzzy control system
social_ctrl = ctrl.ControlSystem([
    rule1, rule2, rule3, rule4, rule5,
    rule6, rule7, rule8,
    rule9, rule10, rule11, rule12,
    rule13, rule14, rule15
])
# ----- Compute fuzzy results safely -----
results = []

for i, row in subset.iterrows():
    social_sim = ctrl.ControlSystemSimulation(social_ctrl)
    try:
        # Clip all values to stay in [0, 10]
        e = float(np.clip(row['empathy'], 0, 10))
        c = float(np.clip(row['cooperation'], 0, 10))
        r = float(np.clip(row['emotional_reg'], 0, 10))
        s = float(np.clip(row['school_Belonging'], 0, 10))

        social_sim.input['empathy'] = e
        social_sim.input['cooperation'] = c
        social_sim.input['emotional_reg'] = r
        social_sim.input['school_Belonging'] = s

        # Compute fuzzy output
        social_sim.compute()

        if 'social_skill' in social_sim.output:
            results.append(social_sim.output['social_skill'])
        else:
            results.append(np.nan)
            print(f"No output for record {i}: {e}, {c}, {r}, {s}")
    except Exception as ex:
        results.append(np.nan)
        print(f" Error at record {i}: {ex}")

# Add results to dataset
subset['social_skill_result'] = results

# Visualize membership functions
empathy.view()
social_skill.view()
plt.show()

# ----- Save fuzzy results -----
subset.to_excel("fuzzy_results_final.xlsx", index=False)
print("Saved fuzzy results to fuzzy_results_final.xlsx")

# ----- Chatbot-style feedback system -----
# ----- Chatbot-style feedback system -----
def feedback_message(score):
    if np.isnan(score):
        return "Undetermined: The student's social-emotional profile does not match any defined patterns. Consider reviewing individual skills and providing personalized support."
    if score < 4:
        return "Needs Support: Encourage teamwork, empathy exercises, and social interaction."
    elif score < 7:
        return "Developing: Student shows average skills; promote group projects and emotional awareness."
    else:
        return "Strong: Student demonstrates strong social-emotional maturity. Maintain peer leadership roles."

# Apply feedback
subset['feedback'] = subset['social_skill_result'].apply(feedback_message)
subset.to_excel("fuzzy_results_with_feedback.xlsx", index=False)
print("Feedback saved to fuzzy_results_with_feedback.xlsx")