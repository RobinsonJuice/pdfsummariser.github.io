from django.shortcuts import render
from flask import Flask, render_template, request, send_file
from flask_wtf import FlaskForm
from wtforms import FileField, SubmitField
from werkzeug.utils import secure_filename
import os
from wtforms.validators import InputRequired
import pdfplumber
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import spacy
import pytextrank
from gtts import gTTS
from playsound import playsound
from fpdf import FPDF

app = Flask(__name__,template_folder='template')
app.config['SECRET_KEY'] = 'supersecretkey'
app.config['UPLOAD_FOLDER'] = 'static/files'

class UploadFileForm(FlaskForm):
    file = FileField("File", validators=[InputRequired()])
    submit = SubmitField("Upload File")

summary = ""
phrases_and_ranks = ""
sumToPdf = FPDF()

@app.route('/', methods=['GET','POST'])
@app.route('/home', methods=['GET','POST'])
def home():
    global summary
    global phrases_and_ranks
    form = UploadFileForm()
    if form.validate_on_submit():
        file = form.file.data 
        file.save(os.path.join(os.path.abspath(os.path.dirname(__file__)),app.config['UPLOAD_FOLDER'],secure_filename(file.filename))) # Then save the file
        pdfToString = ""

        with pdfplumber.open(file) as pdf:
            for page in pdf.pages:
                pdfToString += page.extract_text()

        tokenizer = AutoTokenizer.from_pretrained("nsi319/legal-led-base-16384")  
        model = AutoModelForSeq2SeqLM.from_pretrained("nsi319/legal-led-base-16384")

        padding = "max_length" 

        input_tokenized = tokenizer.encode(pdfToString, return_tensors='pt',padding=padding,pad_to_max_length=True, max_length=6144,truncation=True)
        summary_ids = model.generate(input_tokenized,
                                        num_beams=4,#4
                                        no_repeat_ngram_size=3,#3
                                        length_penalty=2, #2
                                        min_length=50,
                                        max_length=200)
        summary = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summary_ids][0]

        nlp = spacy.load("en_core_web_lg")
        nlp.add_pipe("textrank")

        text = pdfToString

        doc = nlp(text)

        phrases_and_ranks = [ 
            (phrase.chunks[0], phrase.rank) for phrase in doc._.phrases
        ]

        
        summary = (summary.rsplit('. ', 1)[0] + ".")

        return f"<h1>Summary</h1> <form method='POST'><input type='submit' name='submit_button' value='Text to Speech'></form>" + f"<p>{summary}</p>" + f"<h2>Word Frequency (Top 10 Most Frequent Words)</h2>" + f"<ol><li>{str(phrases_and_ranks[0][0]) + ' : ' + str(phrases_and_ranks[0][1])}</li><li>{str(phrases_and_ranks[1][0]) + ' : ' + str(phrases_and_ranks[1][1])}</li><li>{str(phrases_and_ranks[2][0]) + ' : ' + str(phrases_and_ranks[2][1])}</li><li>{str(phrases_and_ranks[3][0]) + ' : ' + str(phrases_and_ranks[3][1])}</li><li>{str(phrases_and_ranks[4][0]) + ' : ' + str(phrases_and_ranks[4][1])}</li><li>{str(phrases_and_ranks[5][0]) + ' : ' + str(phrases_and_ranks[5][1])}</li><li>{str(phrases_and_ranks[6][0]) + ' : ' + str(phrases_and_ranks[6][1])}</li><li>{str(phrases_and_ranks[7][0]) + ' : ' + str(phrases_and_ranks[7][1])}</li><li>{str(phrases_and_ranks[8][0]) + ' : ' + str(phrases_and_ranks[8][1])}</li><li>{str(phrases_and_ranks[8][0]) + ' : ' + str(phrases_and_ranks[8][1])}</li><ol>" + f"<form method='POST'><input type='submit' name='submit_button' value='Convert Summarized Text To PDF'></form>"

    if request.method == 'POST' and request.form['submit_button'] == 'Text to Speech':
        myobj = gTTS(text=summary, lang='en', slow=False)
        myobj.save('summary_tts.mp3')
        playsound('summary_tts.mp3')
        return f"<h1>Summary</h1> <form method='POST'><input type='submit' name='submit_button' value='Text to Speech'></form>" + f"<p>{summary}</p>" + f"<h2>Word Frequency (Top 10 Most Frequent Words)</h2>" + f"<ol><li>{str(phrases_and_ranks[0][0]) + ' : ' + str(phrases_and_ranks[0][1])}</li><li>{str(phrases_and_ranks[1][0]) + ' : ' + str(phrases_and_ranks[1][1])}</li><li>{str(phrases_and_ranks[2][0]) + ' : ' + str(phrases_and_ranks[2][1])}</li><li>{str(phrases_and_ranks[3][0]) + ' : ' + str(phrases_and_ranks[3][1])}</li><li>{str(phrases_and_ranks[4][0]) + ' : ' + str(phrases_and_ranks[4][1])}</li><li>{str(phrases_and_ranks[5][0]) + ' : ' + str(phrases_and_ranks[5][1])}</li><li>{str(phrases_and_ranks[6][0]) + ' : ' + str(phrases_and_ranks[6][1])}</li><li>{str(phrases_and_ranks[7][0]) + ' : ' + str(phrases_and_ranks[7][1])}</li><li>{str(phrases_and_ranks[8][0]) + ' : ' + str(phrases_and_ranks[8][1])}</li><li>{str(phrases_and_ranks[8][0]) + ' : ' + str(phrases_and_ranks[8][1])}</li><ol>" + f"<form method='POST'><input type='submit' name='submit_button' value='Convert Summarized Text To PDF'></form>"

    if request.method == 'POST' and request.form['submit_button'] == 'Convert Summarized Text To PDF':
        sumToPdf.add_page()
        sumToPdf.set_font("Arial", size = 15)
        sumToPdf.cell(200, 10, txt = "Summary",ln = 1, align = 'C')
        sumToPdf.multi_cell(200, 5, txt = summary, align = 'C')
        sumToPdf.cell(200, 10, txt = "Word Frequency (Top 10 Most Frequent Words)", ln = 4, align = 'C')
        sumToPdf.cell(200, 10, txt = "1. " + str(phrases_and_ranks[0][0]) + ' : ' + str(phrases_and_ranks[0][1]),ln = 5, align = 'C')
        sumToPdf.cell(200, 10, txt = "2. " + str(phrases_and_ranks[1][0]) + ' : ' + str(phrases_and_ranks[1][1]),ln = 6, align = 'C')
        sumToPdf.cell(200, 10, txt = "3. " + str(phrases_and_ranks[2][0]) + ' : ' + str(phrases_and_ranks[2][1]),ln = 7, align = 'C')
        sumToPdf.cell(200, 10, txt = "4. " + str(phrases_and_ranks[3][0]) + ' : ' + str(phrases_and_ranks[3][1]),ln = 8, align = 'C')
        sumToPdf.cell(200, 10, txt = "5. " + str(phrases_and_ranks[4][0]) + ' : ' + str(phrases_and_ranks[4][1]),ln = 9, align = 'C')
        sumToPdf.cell(200, 10, txt = "6. " + str(phrases_and_ranks[5][0]) + ' : ' + str(phrases_and_ranks[5][1]),ln = 10, align = 'C')
        sumToPdf.cell(200, 10, txt = "7. " + str(phrases_and_ranks[6][0]) + ' : ' + str(phrases_and_ranks[6][1]),ln = 11, align = 'C')
        sumToPdf.cell(200, 10, txt = "8. " + str(phrases_and_ranks[7][0]) + ' : ' + str(phrases_and_ranks[7][1]),ln = 12, align = 'C')
        sumToPdf.cell(200, 10, txt = "9. " + str(phrases_and_ranks[8][0]) + ' : ' + str(phrases_and_ranks[8][1]),ln = 13, align = 'C')
        sumToPdf.cell(200, 10, txt = "10. " + str(phrases_and_ranks[9][0]) + ' : ' + str(phrases_and_ranks[9][1]),ln = 14, align = 'C')
        sumToPdf.output("SummaryTOPDF.pdf")
        return f"<h1>Summary</h1> <form method='POST'><input type='submit' name='submit_button' value='Text to Speech'></form>" + f"<p>{summary}</p>" + f"<h2>Word Frequency (Top 10 Most Frequent Words)</h2>" + f"<ol><li>{str(phrases_and_ranks[0][0]) + ' : ' + str(phrases_and_ranks[0][1])}</li><li>{str(phrases_and_ranks[1][0]) + ' : ' + str(phrases_and_ranks[1][1])}</li><li>{str(phrases_and_ranks[2][0]) + ' : ' + str(phrases_and_ranks[2][1])}</li><li>{str(phrases_and_ranks[3][0]) + ' : ' + str(phrases_and_ranks[3][1])}</li><li>{str(phrases_and_ranks[4][0]) + ' : ' + str(phrases_and_ranks[4][1])}</li><li>{str(phrases_and_ranks[5][0]) + ' : ' + str(phrases_and_ranks[5][1])}</li><li>{str(phrases_and_ranks[6][0]) + ' : ' + str(phrases_and_ranks[6][1])}</li><li>{str(phrases_and_ranks[7][0]) + ' : ' + str(phrases_and_ranks[7][1])}</li><li>{str(phrases_and_ranks[8][0]) + ' : ' + str(phrases_and_ranks[8][1])}</li><li>{str(phrases_and_ranks[8][0]) + ' : ' + str(phrases_and_ranks[8][1])}</li><ol>" + f"<form method='POST'><input type='submit' name='submit_button' value='Convert Summarized Text To PDF'></form>"


    return render_template('index.html', form=form)

if __name__== '__main__':
    app.run(debug=True)