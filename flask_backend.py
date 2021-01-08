from flask import Flask, request
from flask_restful import Api, Resource
import word_generator
import pdf_png_text_extraction

app = Flask(__name__)
api = Api(app)


class test(Resource):
    def post(self):
        if request.files:
            result = pdf_png_text_extraction.process_file_post(request.files['filename'])
        return result

api.add_resource(test, '/')

if __name__ == "__main__":
    app.run(debug = True)