from flask import Flask, request, jsonify
import os

dir_path = os.path.dirname(os.path.realpath(__file__))
app = Flask(__name__, root_path=dir_path)

@app.route('/', methods=['GET'])
def home():
    return jsonify({'message': 'Service Running'})


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)

