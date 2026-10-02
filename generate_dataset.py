# generate_dataset.py
"""Generate a richer synthetic dataset for experimentation with balanced placement labels."""

import pandas as pd
import numpy as np
import os

np.random.seed(42)
N = 2000  # total students

branches = ["CSE", "IT", "ECE", "MECH", "CIVIL", "EEE"]
gender = ["M", "F"]
job_roles = ["Data Scientist", "Software Engineer", "QA", "DevOps", "Business Analyst", "Embedded Engineer"]

rows = []

for i in range(N):
    cgpa = np.round(np.clip(np.random.normal(7.2, 0.8), 4.0, 10.0), 2)
    intern = np.random.choice([0, 1], p=[0.6, 0.4])
    certs = np.random.poisson(0.7)
    comm = np.round(np.clip(np.random.normal(60, 12), 20, 100), 0)
    aptitude = np.round(np.clip(np.random.normal(55, 15), 10, 100), 0)
    branch = np.random.choice(branches, p=[0.25, 0.2, 0.2, 0.15, 0.1, 0.1])
    gender_ = np.random.choice(gender, p=[0.6, 0.4])
    skills = np.random.choice(
        ["Python;SQL", "C/C++", "Embedded C", "Java;SQL", "None", "Python;ML"],
        p=[0.25, 0.15, 0.1, 0.2, 0.15, 0.15]
    )

    # Placement probability score
    score = 0.4 * cgpa + 0.3 * (intern * 1.0) + 0.05 * certs + 0.2 * (comm / 100) + 0.2 * (aptitude / 100)
    placed = 1 if score + np.random.normal(0, 0.2) > 6.0 else 0

    # Force balance: half placed, half not placed
    if i < N // 2:
        placed = 1
    else:
        placed = 0

    salary = np.nan
    job = np.nan
    if placed:
        base = 300000 + (cgpa - 6.0) * 50000 + certs * 5000 + (comm - 50) * 1000
        salary = max(150000, int(base + np.random.normal(0, 20000)))
        job = np.random.choice(job_roles)

    rows.append({
        "StudentID": f"S{i+1:04d}",
        "Gender": gender_,
        "Branch": branch,
        "CGPA": cgpa,
        "Internship": intern,
        "Certifications": certs,
        "Communication": comm,
        "Aptitude": aptitude,
        "Skills": skills,
        "Placed": placed,
        "Salary": salary,
        "JobRole": job
    })

df = pd.DataFrame(rows)

# Shuffle rows so placed/not placed are mixed
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

os.makedirs("data", exist_ok=True)
df.to_csv("data/student_placement_dataset.csv", index=False)
print("✅ Balanced dataset saved to data/student_placement_dataset.csv")
print(df["Placed"].value_counts())
