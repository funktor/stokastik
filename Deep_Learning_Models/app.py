import json
from flask import Flask, jsonify, make_response, request
import inferencing.deep_matching.inference as DM_INF
import inferencing.color_extraction.inference as CL_INF

app = Flask(__name__)
dm_inf, cl_inf = None, None

@app.route('/deep_matching/predict', methods=['POST'])
def dm_predict():
    input_data = request.json['input_data']
    if dm_inf is not None:
        if isinstance(input_data, dict):
            output = dm_inf.predict(input_data)
        elif isinstance(input_data, list):
            output = dm_inf.predict_batch(input_data)
        else:
            output = {"status" : 0, "response" : "Input data not in correct format"}
        return make_response(jsonify(json.dumps({'output': output})))
    return make_response(jsonify(json.dumps({'output': {"status" : 0, "response" : "Model could not be instantiated"}})))


@app.route('/color_extraction/predict', methods=['POST'])
def cl_predict():
    if request.files.get("image"):
        image = request.files["image"].read()
    else:
        return make_response(jsonify(json.dumps({'output': {"status" : 0, "response" : "Image not uploaded"}})))
        
    if cl_inf is not None:
        output = cl_inf.predict(image)
        return make_response(jsonify(json.dumps({'output': output})))
    
    return make_response(jsonify(json.dumps({'output': {"status" : 0, "response" : "Model could not be instantiated"}})))


if __name__ == '__main__':
    dm_inf = DM_INF.Inference()
    cl_inf = CL_INF.Inference()
        
    app.run(host='0.0.0.0', port=5000)