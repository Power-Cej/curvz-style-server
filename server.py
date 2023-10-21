from flask import Flask, request, jsonify
from flask_cors import CORS
import base64
import motionCapture

app = Flask(__name__)
CORS(app)


@app.route('/myServer', methods=['POST'])
def myServer():
    data = request.get_json()
    image_data = data.get('imageData')

    # Decode base64 data to bytes
    image_bytes = base64.b64decode(image_data.split(',')[1])

    # Save the bytes to a file
    with open('testing.png', 'wb') as f:
        f.write(image_bytes)

    result = motionCapture.capture_motion('testing.png')
    return jsonify(result)


if __name__ == '__main__':
    app.run(debug=True)
