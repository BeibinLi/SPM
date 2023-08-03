from flask import Flask, jsonify

#from inference import answer

app = Flask(__name__)


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
