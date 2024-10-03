from flask import Flask, request, jsonify
from flask_cors import CORS
import create_query_engine

app = Flask(__name__)
CORS(app)

@app.route("/")
def home():
    return "Hello, Flask!"

@app.route('/generate', methods=['GET'])
def test():
    prompt = request.args.get('prompt')
    response = create_query_engine.take_prompt(prompt)
    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000)