import os
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from uuid import uuid4

app = Flask(__name__, static_folder="public",
            static_url_path="/public", template_folder="views")

app.config["UPLOAD_FOLDER"] = os.path.join(os.getcwd(), "uploads")
app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024

ALLOWED_EXTENSIONS = ["image/png", "image/jpg", "image/jpeg"]

if not os.path.exists(app.config["UPLOAD_FOLDER"]):
    os.makedirs(app.config["UPLOAD_FOLDER"])


@app.get('/')
def index():
    return render_template('index.html')


@app.get("/about")
def about():
    return render_template('about.html')


@app.get("/how-to")
def howto():
    return render_template('how-to.html')


@app.post("/get-pred")
def get_pred():
    f = request.files['image']
    if not f:
        return "No file selected."
    elif f.mimetype not in ALLOWED_EXTENSIONS:
        return "Only PNG and JPG/JPEG files are allowed."
    else:
        fileExtension = f.mimetype.split("/")[1]
        f.save(os.path.join(
            app.config['UPLOAD_FOLDER'], secure_filename(uuid4().hex+"."+fileExtension)))
        return render_template('pred.html')
