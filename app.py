from flask import Flask, request, jsonify
import openai
import os
import re
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from dotenv import load_dotenv

# تحميل متغيرات البيئة
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

model_name = "gpt-3.5-turbo"
analyzer = SentimentIntensityAnalyzer()

app = Flask(__name__)

# ---- GENERATE QUESTIONS ----
def generate_questions(role):
    prompt = (
        f"Generate 3 behavioral and 2 technical interview questions for a {role} role. "
        "Please list only the questions, numbered."
    )
    response = openai.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=1.0,
        top_p=1.0,
        max_tokens=1000,
    )
    content = response.choices[0].message.content.strip()
    lines = content.split("\n")
    questions = [line.strip() for line in lines if line.strip() and "?" in line]
    return questions

@app.route('/start_interview', methods=['POST'])
def start_interview():
    data = request.get_json()
    job_role = data.get('job_role', 'Data Scientist')
    questions = generate_questions(job_role)
    return jsonify({'questions': questions})

@app.route('/submit_answer', methods=['POST'])
def submit_answer():
    data = request.get_json()
    question = data.get('question')
    answer = data.get('answer')
    
    sentiment = analyzer.polarity_scores(answer)
    
    feedback_prompt = (
        f"Evaluate how well the following answer responds to the interview question "
        f"in terms of relevance, completeness, and clarity.\n\n"
        f"Question: {question}\n"
        f"Answer: {answer}\n\n"
        f"Provide detailed feedback, then add a score out of 10 using this format:\n"
        f"Rating: X/10"
    )

    feedback_response = openai.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": feedback_prompt}],
        temperature=0.7,
        top_p=1.0,
        max_tokens=500,
    )

    feedback_text = feedback_response.choices[0].message.content.strip()
    
    rating = None
    if "Rating:" in feedback_text:
        rating_line = [line for line in feedback_text.split('\n') if "Rating:" in line]
        if rating_line:
            rating_str = rating_line[0].split(":")[1].strip()
            match = re.search(r'\d+', rating_str)
            if match:
                extracted = int(match.group())
                if 0 <= extracted <= 10:
                    rating = extracted
                else:
                    rating = None

    return jsonify({
        'feedback': feedback_text,
        'rating': rating
    })

# يضيف ملف التطبيق ليعمل كـ "Serverless Function"
if __name__ == "__main__":
    app.run(debug=True)
