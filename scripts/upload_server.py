from flask import Flask, request, render_template_string
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = script_dir

HTML = """
<!doctype html>
<title>Upload</title>
<h1>Upload File with Progress</h1>
<form id="uploadForm" method="post" enctype="multipart/form-data" onsubmit="return false;">
  <input type="file" id="fileInput" name="file"><br><br>
  <button onclick="startUpload()">Start Upload</button>
  <button onclick="cancelUpload()">Cancel Upload</button><br><br>
  <progress id="progressBar" value="0" max="100" style="width:300px;"></progress>
  <div id="status"></div>
</form>

<script>
let xhr = null;

function startUpload() {
  const file = document.getElementById("fileInput").files[0];
  if (!file) {
    alert("Please select a file first.");
    return;
  }

  xhr = new XMLHttpRequest();
  xhr.upload.addEventListener("progress", function(e) {
    if (e.lengthComputable) {
      document.getElementById("progressBar").value = Math.round((e.loaded / e.total) * 100);
    }
  }, false);

  xhr.addEventListener("load", function() {
    document.getElementById("status").innerHTML = "Upload complete.";
  }, false);

  xhr.addEventListener("error", function() {
    document.getElementById("status").innerHTML = "Upload failed.";
  }, false);

  xhr.addEventListener("abort", function() {
    document.getElementById("status").innerHTML = "Upload canceled.";
  }, false);

  xhr.open("POST", "/upload");
  const formData = new FormData();
  formData.append("file", file);
  xhr.send(formData);
}

function cancelUpload() {
  if (xhr) {
    xhr.abort();
  }
}
</script>
"""

@app.route('/', methods=['GET'])
def index():
    return render_template_string(HTML)

@app.route('/upload', methods=['POST'])
def upload():
    f = request.files['file']
    f.save(os.path.join(app.config['UPLOAD_FOLDER'], f.filename))
    return 'OK'

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)