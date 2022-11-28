from flask import Flask,  render_template

app = Flask(__name__, static_folder="public",
            static_url_path="/public", template_folder="views")


@app.get('/')
def index():
    return render_template('index.html')


@app.get("/about")
def about():
    return render_template('about.html')


@app.get("/how-to")
def howto():
    return render_template('how-to.html')
