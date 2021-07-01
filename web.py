###########
# @TroyZhang
# 6/30/2021
###########

from flask import Flask, render_template, request, make_response
from process import ProcessImage
import os

IMG_FOLDER = os.path.join('static' , 'imgs')

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = IMG_FOLDER

@app.route('/')
def index():
	return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
	PI = ProcessImage()
	if request.files.get('photo'):
		f = request.files['photo']
		file_path = os.path.join(app.config['UPLOAD_FOLDER'], f.filename)
		f.save(file_path)
		# read
		data = PI.run_app(file_path)
		# response picture
		return render_template('index.html',
			data = data,
			usr_img = file_path)



if __name__ == '__main__':
	app.run(debug=True)

