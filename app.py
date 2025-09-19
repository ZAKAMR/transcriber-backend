import os
import tempfile
from flask import Flask, request, jsonify
from flask_cors import CORS
import whisper

app = Flask(__name__)
CORS(app)  # allow frontend to connect

# Cache loaded models in memory so they don’t reload every request
whisper_models = {}

def get_whisper_model(name="small"):
    """Load whisper model into cache (CPU only)."""
    if name not in whisper_models:
        print(f"Loading Whisper model: {name} ...")
        whisper_models[name] = whisper.load_model(name)
    return whisper_models[name]

@app.route("/api/transcribe", methods=["POST"])
def transcribe_audio():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    # Default to small if not provided
    model_name = request.form.get("model", "small")

    try:
        # Save file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            file.save(tmp.name)
            tmp_path = tmp.name

        # Load Whisper model
        model = get_whisper_model(model_name)

        # Transcribe
        result = model.transcribe(tmp_path, task="transcribe")

        # Build response
        segments = []
        for i, seg in enumerate(result.get("segments", []), start=1):
            segments.append({
                "id": i,
                "startTime": seg["start"],
                "endTime": seg["end"],
                "text": seg["text"].strip()
            })

        response = {
            "language": result.get("language", "unknown"),
            "segments": segments
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass

import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)

