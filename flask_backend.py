from flask import Flask, request
from flask_restful import Api, Resource
import word_generator
import pdf_png_text_extraction

app = Flask(__name__)
api = Api(app)


class test(Resource):
    def get(self, message):
        tokenizer, model = word_generator.load_model_tokenizer_GPT2()
        prob_word_dic = word_generator.next_word_prediction(tokenizer, model, message, num_results = 3)
        return prob_word_dic

    def post(self, message):
        if request.files:
            result = pdf_png_text_extraction.process_pdf(request.files['filename'])
        return result

api.add_resource(test, '/nwp/<string:message>')

if __name__ == "__main__":
    app.run(debug = True)