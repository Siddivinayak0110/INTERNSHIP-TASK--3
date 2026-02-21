from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os
import uuid
from nst_model import run_style_transfer

app = Flask(__name__)

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
OUTPUT_FOLDER = os.path.join(BASE_DIR, "static", "outputs")
STYLE_FOLDER = os.path.join(BASE_DIR, "static", "styles")

# Create folders automatically
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(STYLE_FOLDER, exist_ok=True)

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":

        file = request.files.get("content_image")
        style_name = request.form.get("style")

        if file and allowed_file(file.filename):

            # Create unique filename
            ext = file.filename.rsplit(".", 1)[1].lower()
            unique_filename = str(uuid.uuid4()) + "." + ext

            content_path = os.path.join(UPLOAD_FOLDER, unique_filename)
            file.save(content_path)

            style_path = os.path.join(STYLE_FOLDER, style_name)

            output_filename = "output_" + unique_filename
            output_path = os.path.join(OUTPUT_FOLDER, output_filename)

            # Run style transfer
            run_style_transfer(content_path, style_path, output_path)

            return render_template(
                "index.html",
                output_image=f"outputs/{output_filename}"
            )

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)