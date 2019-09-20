import sys
sys.path.remove("/data")
sys.path.append("/home/jupyter/stormbreaker/Deep_Learning_Models")

import json
from flask import Flask, jsonify, make_response, request
from inferencing.deep_matching.inference import Inference

app = Flask(__name__)
inf = None


@app.route('/predict', methods=['POST'])
def predict():
    input_data = request.json['input_data']
    if inf is not None:
        if isinstance(input_data, dict):
            output = inf.predict(input_data)
        elif isinstance(input_data, list):
            output = inf.predict_batch(input_data)
        else:
            output = {"status" : 0, "response" : "Input data not in correct format"}
        return make_response(jsonify(json.dumps({'output': output})))
    return make_response(jsonify(json.dumps({'output': {"status" : 0, "response" : "Model could not be instantiated"}})))


if __name__ == '__main__':
    inf = Inference()
    app.run(host='0.0.0.0', port=5000)