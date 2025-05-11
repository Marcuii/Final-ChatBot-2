from flask import Flask, request, jsonify
from flask_cors import CORS
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

# القائمة البيضاء للمواقع المسموح لها بالوصول
ALLOWED_ORIGINS = [
    "https://vocanova.vercel.app",
    "https://vocanova.vercel.app/mockup-interview"
]

# تكوين CORS بدقة
CORS(app, resources={
    r"/*": {
        "origins": ALLOWED_ORIGINS,
        "methods": ["POST", "OPTIONS"],
        "allow_headers": ["Content-Type"],
        "supports_credentials": True
    }
})

@app.before_request
def check_origin():
    """التحقق من أن الطلب قادم من مصدر مسموح به"""
    if request.method == 'OPTIONS':
        return
    
    origin = request.headers.get('Origin')
    if origin not in ALLOWED_ORIGINS:
        return jsonify({"error": "الوصول غير مسموح به لهذا المصدر"}), 403

@app.after_request
def add_cors_headers(response):
    """إضافة رؤوس CORS للاستجابات"""
    if request.headers.get('Origin') in ALLOWED_ORIGINS:
        response.headers.add('Access-Control-Allow-Origin', request.headers.get('Origin'))
        response.headers.add('Access-Control-Allow-Credentials', 'true')
        response.headers.add('Access-Control-Expose-Headers', 'Content-Type')
    return response

# ---- وظيفة توليد الأسئلة ----
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

@app.route('/start-interview', methods=['POST', 'OPTIONS'])
def start_interview():
    if request.method == 'OPTIONS':
        return _build_preflight_response()
    
    data = request.get_json()
    job_role = data.get('job_role', 'Data Scientist')
    questions = generate_questions(job_role)
    return _build_actual_response(jsonify({'questions': questions}))

@app.route('/submit-answer', methods=['POST', 'OPTIONS'])
def submit_answer():
    if request.method == 'OPTIONS':
        return _build_preflight_response()
    
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

    return _build_actual_response(jsonify({
        'feedback': feedback_text,
        'rating': rating
    }))

def _build_preflight_response():
    """بناء استجابة Preflight لطلبات OPTIONS"""
    response = jsonify({'status': 'ok'})
    response.headers.add('Access-Control-Allow-Origin', ', '.join(ALLOWED_ORIGINS))
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
    response.headers.add('Access-Control-Allow-Methods', 'POST, OPTIONS')
    return response

def _build_actual_response(response):
    """إضافة رؤوس CORS للاستجابات الفعلية"""
    origin = request.headers.get('Origin')
    if origin in ALLOWED_ORIGINS:
        response.headers.add('Access-Control-Allow-Origin', origin)
        response.headers.add('Access-Control-Allow-Credentials', 'true')
    return response

if __name__ == "__main__":
    app.run(debug=True)
