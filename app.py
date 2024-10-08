from flask import Flask, jsonify, request, send_file
import os
import openai
from dotenv import load_dotenv
import PyPDF2
import io
import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer


load_dotenv()

app = Flask(__name__)

openai.api_key = os.getenv('OPENAI_API_KEY')
if not openai.api_key:
    raise ValueError("API key is missing. Ensure you have an OPENAI_API_KEY in your .env file.")

print("API key loaded successfully.")

#helper fuunction
def extract_text_from_pdf(pdf_file):
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return " ".join(text.split()) 
    except Exception as e:
        raise Exception(f"Error reading PDF: {str(e)}")
    

@app.route("/")
def index():
    return send_file("index.html")


@app.route("/api/generate-flashcards", methods = ['POST'])
def generate_flashcards():
    #check if received data
    print("Received data:", request.files, request.form)

    if "file" in request.files:
        file  = request.files['file']
        if file.filename.endswith('.pdf'):
            text = extract_text_from_pdf(file.stream)
        else:
            return jsonify({"error": "Unsupported file type. Please upload a PDF file"}), 400
    else:
        text = request.form.get("text")

    if not text:
        return jsonify({"error": "No text provided"}), 400
    
    try:
        prompt = f"""
        Given the following text, generate 5 flashcards in the format:
        Question: [Question here]
        Answer: [Answer here]

        Text: {text[:4000]} # limit text to 4000 charaters

        Please provide the output in JSON format.
        """
        print("Prompt sent to OpenAI: ", prompt)
        response = openai.chat.completions.create(
            model = "gpt-3.5-turbo",
            messages = [
                {"role": "system", "content": "You are a helpful assistant that generates flashcards from the given text."},
                {"role": "user", "content": prompt}
            ],
            max_tokens = 1000
        )

        print("OpenAI response: ", response)
        
        flashcards_json = response.choices[0].message.content
        #flashcards_json = flashcards_json.strip("```json").strip("```").strip()
        flashcards_json = flashcards_json.split('```json', 1)[-1].split('```', 1)[0].strip()
        try:
            flashcards = json.loads(flashcards_json)  # Parse the response into JSON
            print(flashcards_json)

        except json.JSONDecodeError:
            print(f"JSON decode error: {e}")
            print(f"Problematic JSON: {flashcards_json}")
            return jsonify({"error": "Failed to parse OpenAI response into JSON."}), 500

        return jsonify({"flashcards": flashcards})

    except Exception as e:
        print(f"Error: {str(e)}")  # Print the exact error for debugging
        return jsonify({"error": str(e)}), 500
    
@app.route('/api/check-answer', methods = ["POST"])
def check_answer():
    data = request.json
    students_answer = data["studentsAnswer"]
    corect_answer = data["correctAnswer"]

    #calculate cousine similarity
    documents = [students_answer, corect_answer]
    vectorizer = TfidfVectorizer().fit_transform(documents)
    vectors = vectorizer.toarray()

    cosine_sim = cosine_similarity(vectors)
    similarity_score = cosine_sim[0][1]

    #feedback
    feedback = f"Similarity Score: {similarity_score:.2f}."
    if similarity_score >= 90:
        feedback += "Correct Answer!"
    elif similarity_score >= 50:
        feedback += "Partially Correct"
    else:
        feedback += "Incorrect Answer"

    return jsonify({"feedback": feedback})


    
if __name__ == "__main__":
    app.run(debug= True)