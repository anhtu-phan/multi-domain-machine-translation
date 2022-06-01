import os
import torch
from flask import Flask, request, render_template, redirect, url_for
import argparse

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/', methods=['POST'])
def index_post():
    input_sentence = request.form['input_sentence']

    return render_template('index.html')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Transformer text recognition demo")
    parser.add_argument("--port", default=9595)
    parser.add_argument("--model_folder", default="./checkpoints")

    app.run('0.0.0.0', port=9595)
