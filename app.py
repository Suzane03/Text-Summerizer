from flask import Flask, render_template, request as req
from transformers import BartForConditionalGeneration, BartTokenizer
import logging

app = Flask(__name__)

# Setting up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Load pre-trained BART model and tokenizer
try:
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
    model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
except Exception as e:
    logger.error("Error loading model or tokenizer: %s", e)

@app.route("/", methods=["GET", "POST"])
def Index():
    return render_template("index.html")

@app.route("/Summarize", methods=["GET", "POST"])
def Summarize():
    if req.method == "POST":
        data = req.form["data"]
        maxL = int(req.form["maxL"])
        minL = maxL // 4

        try:
            # Tokenize the input text
            inputs = tokenizer([data], max_length=1024, truncation=True, return_tensors="pt")

            # Generate summary
            summary_ids = model.generate(inputs["input_ids"], max_length=maxL, min_length=minL, num_beams=4, length_penalty=2.0, early_stopping=True)

            # Decode the generated summary
            output = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

            return render_template("index.html", result=output)
        except Exception as e:
            logger.error("Error generating summary: %s", e)
            return render_template("index.html", result="Error generating summary. Please try again.")
    else:
        return render_template("index.html")

if __name__ == '__main__':
    app.run(debug=True)
