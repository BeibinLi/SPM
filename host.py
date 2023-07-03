import platform

from flask import Flask, jsonify, request

from inference import answer

app = Flask(__name__)

# def answer(question):
#     # here we simply reverse the string, you should replace it with your own logic
#     return f"I received your question {question}. Nice!!!"


@app.route('/question', methods=['POST'])
def get_answer():
    data = request.get_json()

    # check if question field exists
    if 'question' not in data:
        return jsonify({'message': 'No question field'}), 400

    question = data['question']
    answer_str = answer(question)

    return jsonify({'answer': answer_str})


if __name__ == '__main__':
    app.run(debug=False, host=platform.node().lower(), port=5000)
