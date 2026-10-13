from flask import Flask, request, jsonify
from flask_cors import CORS

from predict import predict_patient

app = Flask(__name__)
CORS(app)


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "message": "CardioVision API is running"})


@app.route("/api/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(silent=True)

        if not isinstance(data, dict):
            return jsonify({"error": "Send patient details as JSON"}), 400

        result = predict_patient(data)
        return jsonify(result)

    except Exception as error:
        app.logger.exception("Prediction failed")
        return jsonify({"error": str(error)}), 500


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
