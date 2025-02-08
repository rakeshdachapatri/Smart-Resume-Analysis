import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from pdfminer.high_level import extract_text
import tkinter as tk
from tkinter import filedialog

# Load jobs data from CSV
job_data = pd.read_csv("test.csv")
job_data.fillna('missing', inplace=True)

# Vectorize job requirements text
vectorizer = CountVectorizer()
job_matrix = vectorizer.fit_transform(job_data["Requirements"])

# Train the classifier
classifier = MultinomialNB()
classifier.fit(job_matrix, job_data["Job"])

# Function to load and process PDF
def load_pdf():
    file_path = filedialog.askopenfilename(filetypes=[("PDF Files", "*.pdf")])
    if not file_path:
        return

    # Extract text from resume PDF
    resume_text = extract_text(file_path)
    
    # Vectorize the resume text
    resume_matrix = vectorizer.transform([resume_text])

    # Predict probabilities for job requirements
    probabilities = classifier.predict_proba(resume_matrix)

    # Get top N predictions
    n = 3
    top_n_predictions = sorted(zip(classifier.classes_, probabilities[0]), key=lambda x: x[1], reverse=True)[:n]

    # Display top N job recommendations in the GUI
    for i, prediction in enumerate(top_n_predictions):
        job_label = tk.Label(root, text=f"{i+1}. {prediction[0]}", font=("Arial", 14), fg="black", padx=10, pady=2)
        job_label.pack()

# Create the GUI
root = tk.Tk()
root.geometry("500x500")
root.title("Job Recommendation System")

recommendations_label = tk.Label(root, text="Job Recommendation System", font=("Arial", 20), bg="yellow")
recommendations_label.pack(pady=10)

txt_label = tk.Label(root, text="We provide you Job Recommendations based on your Resume", font=("Arial", 16), bg="black", fg="white")
txt_label.pack(padx=10, pady=50)

browse_button = tk.Button(root, text="Browse Resume", height=2, width=30, font=("Arial", 12), bg="#3c8dbc", fg="#fff", command=load_pdf)
browse_button.pack(padx=20, pady=100)

txt1_label = tk.Label(root, text="Job Recommendations:", font=("Arial", 16), bg="black", fg="white")
txt1_label.pack(padx=10, pady=3)

root.configure(bg='white')
root.mainloop()

