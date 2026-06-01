from flask import Flask, render_template, request
import whisper
from transformers import pipeline

app = Flask(__name__)

# Speech-to-Text Model
model = whisper.load_model("base")

# Summarization Model
summarizer = pipeline(
    "summarization",
    model="facebook/bart-large-cnn"
)

@app.route("/", methods=["GET", "POST"])
def home():

    transcript = ""
    summary = ""

    if request.method == "POST":

        audio_file = request.files["audio"]

        audio_file.save(audio_file.filename)

        result = model.transcribe(
            audio_file.filename
        )

        transcript = result["text"]

        summary_result = summarizer(
            transcript,
            max_length=150,
            min_length=40,
            do_sample=False
        )

        summary = summary_result[0][
            "summary_text"
        ]

    return render_template(
        "index.html",
        transcript=transcript,
        summary=summary
    )

if __name__ == "__main__":
    app.run(debug=True)
