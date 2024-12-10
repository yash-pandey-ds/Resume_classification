import re
import PyPDF2
import docx2txt
import pdfplumber
import pandas as pd
import streamlit as st

import en_core_web_sm

nlp = en_core_web_sm.load()
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer

st.title('Resume Classification')
# st.markdown('<style>h1{color: red;}</style>', unsafe_allow_html=True)


def extract_skills(resume_text):
    nlp_text = nlp(resume_text)
    noun_chunks = nlp_text.noun_chunks
    tokens = [token.text for token in nlp_text if
              not token.is_stop]

    data = pd.read_csv(r"skills.csv")
    skills = list(data.columns.values)
    skillset = []

    for token in tokens:
        if token.lower() in skills:
            skillset.append(token)

    for token in noun_chunks:
        token = token.text.lower().strip()
        if token in skills:
            skillset.append(token)
    return [i.capitalize() for i in set([i.lower() for i in skillset])]


def getText(filename):
    fullText = ''  # Create empty string
    if filename.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        doc = docx2txt.process(filename)
        for para in doc:
            fullText = fullText + para
    else:
        with pdfplumber.open(filename) as pdf_file:
            pdoc = PyPDF2.PdfFileReader(filename)
            number_of_pages = pdoc.getNumPages()
            page = pdoc.pages[0]
            page_content = page.extractText()
        for paragraph in page_content:
            fullText = fullText + paragraph
    return fullText


def display(doc_file):
    resume = []
    if doc_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        resume.append(docx2txt.process(doc_file))
    else:
        with pdfplumber.open(doc_file) as pdf:
            pages = pdf.pages[0]
            resume.append(pages.extract_text())
    return resume


def preprocess(sentence):
    sentence = str(sentence)
    sentence = sentence.lower()
    sentence = sentence.replace('{html}', "")
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, '', sentence)
    rem_url = re.sub(r'http\S+', '', cleantext)
    rem_num = re.sub('[0-9]+', '', rem_url)
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(rem_num)
    filtered_words = [w for w in tokens if len(w) > 2 if not w in stopwords.words('english')]
    lemmatizer = WordNetLemmatizer()
    lemma_words = [lemmatizer.lemmatize(w) for w in filtered_words]
    return " ".join(lemma_words)


file_type = pd.DataFrame([], columns=['Uploaded File', 'Predicted Profile', 'Skills', ])
filename = []
predicted = []
skills = []

import pickle as pk

model = pk.load(open(r"modelDT.sav", 'rb'))
Vectorizer = pk.load(open(r"vector.pkl", 'rb'))

upload_file = st.file_uploader('Upload Your Resumes...', type=['docx', 'pdf'], accept_multiple_files=True)

output = ''

for doc_file in upload_file:
    if doc_file is not None:
        filename.append(doc_file.name)
        cleaned = preprocess(display(doc_file))
        prediction = model.predict(Vectorizer.transform([cleaned]))[0]
        predicted.append(prediction)
        extText = getText(doc_file)
        skills.append(extract_skills(extText))

    output = ''
    if predicted[-1] == 1:
        output = 'React Developer'
    elif predicted[-1] == 0:
        output = 'Peoplesoft'
    elif predicted[-1] == 2:
        output = 'SQL Developer'
    else:
        output = 'Workday'

    file_type['Uploaded File'] = filename
    file_type['Skills'] = skills
    file_type['Predicted Profile'] = output
    # st.table(file_type.style.format())

    predicted_profile = file_type['Predicted Profile'][0]
    st.success('The predicted profile for the resume is: '+predicted_profile)

