import os
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from uuid import uuid4
from utils.PredictionHelper import PredictionHelper


app = Flask(__name__, static_folder="public",
            static_url_path="/public", template_folder="views")

app.config["UPLOAD_FOLDER"] = os.path.join(os.getcwd(), "uploads")
app.config["MAX_CONTENT_LENGTH"] = 4 * 1024 * 1024
app.config["TESTING"] = False

ALLOWED_EXTENSIONS = ["image/png", "image/jpg", "image/jpeg"]

if not os.path.exists(app.config["UPLOAD_FOLDER"]):
    os.makedirs(app.config["UPLOAD_FOLDER"])

ph = PredictionHelper()


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/about")
def about():
    return render_template("about.html")


@app.get("/how-to")
def howto():
    return render_template("how-to.html")


@app.post("/get-pred")
def get_pred():
    f = request.files["image"]
    if not f:
        return "No file selected."
    elif f.mimetype not in ALLOWED_EXTENSIONS:
        return "Only PNG and JPG/JPEG files are allowed."
    else:
        fileExtension = f.mimetype.split("/")[1]
        fileName = secure_filename(uuid4().hex+"."+fileExtension)
        filePath = os.path.join(app.config["UPLOAD_FOLDER"], fileName)
        f.save(filePath)

        ph.set_image_path(filePath)
        res = ph.run()

        os.remove(filePath)

        return render_template("pred.html", svm=res[0], cnn=res[1])


if __name__ == "__main__":
    app.run(debug=False)
