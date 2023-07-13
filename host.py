import platform

from flask import Flask, jsonify, request

#from inference import answer

app = Flask(__name__)

# def answer(question):
#     # here we simply reverse the string, you should replace it with your own logic
#     return f"I received your question {question}. Nice!!!"


@app.route('/question', methods=['GET', 'POST'])
def get_answer():
    print("RRR")
    # data = request.get_json()

    # # check if question field exists
    # if 'question' not in data:
    #     return jsonify({'message': 'No question field'}), 400

    # question = data['question']
    # answer_str = answer(question)

    # return jsonify({'answer': answer_str})

    return jsonify({'answer': "yes!"})

if __name__ == '__main__':
    app.run(debug=False, host="localhost", port=5000)
