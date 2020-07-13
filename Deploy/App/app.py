from flask import Flask, request, jsonify
import os
import sys
import Classifier

sys.path.append(os.path.join(os.path.dirname(__file__), 'App'))
dir_path = os.path.dirname(os.path.realpath(__file__))
app = Flask(__name__, root_path=dir_path)

@app.route('/', methods=['GET'])
def home():
    return jsonify({'message': 'Service Running'})

@app.route('/predict', methods=['POST'])
def predict():
    res = None
    data = request.get_json()
    if list(data.keys())[0] == 'value':
        model = Classifier.LSTMPipeline(data['value'])
        res = model.predict()
    else:
        res = 'Wrong Key'
    return jsonify({'result' : res})
    
if __name__ == "__main__":    
    app.run(debug=True, host='0.0.0.0', port=5000)

