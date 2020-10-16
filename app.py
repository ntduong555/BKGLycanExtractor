#!./bin/python

# HACK! to allow this into the background!

import sys, werkzeug._internal


def demi_logger(type, message, *args, **kwargs):
    print(message, args, kwargs, file=sys.stderr)


werkzeug._internal._log = demi_logger

import os, secrets
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from BKGLycanExtractor.annotatePDF import annotatePDFGlycan
import traceback, time, shutil

app = Flask(__name__)

# 1MB max
app.config['MAX_CONTENT_PATH'] = 1024 * 1024 * 10

base = os.getcwd()
base_configs = base + "/BKGLycanExtractor/configs/"


@app.route('/')
def upload_file():
    base_example = os.path.join("static/pdfexample")
    return render_template('upload.html', basedir=base_example)


@app.route('/upload_handler', methods=['POST'])
def upload_handler():
    f = request.files['file']
    origfilename = secure_filename(f.filename)
    token = secrets.token_hex(nbytes=16)
    os.makedirs(os.path.join("static/files", token, "input"))
    os.makedirs(os.path.join("static/files", token, "output"))
    workdir = os.path.join("static/files", token)
    infilename = os.path.join(workdir, "input", origfilename)
    outfilename = os.path.join(workdir, "output", "annotated_" + origfilename)
    f.save(infilename)
    print(f"Working directory: {workdir}")
    work_dict = {
        "token":token,
        "workdir":workdir,
        "infilename":infilename,
        "outfilename":outfilename,
        "base_configs":base_configs
    }
    annotatePDFGlycan(work_dict)

    print(f"Output file: {outfilename}")
    return render_template('result.html', filename=origfilename, infileurl=infilename,
                           outfileurl=outfilename)

@app.route('/retrieve', methods=['POST','GET'])
def retrieve():
    if request.method =="POST":
        jobid = request.form.get("jobid")
        print(f"request job id {jobid}")
        return render_template('retrieve.html')
    else:
        return render_template('retrieve.html')

if __name__ == '__main__':
    app.run(port=10982, host="0.0.0.0", threaded=True)
