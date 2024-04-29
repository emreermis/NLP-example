
from flask import Flask, url_for, render_template, redirect, request
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk import pos_tag
import string, spacy
nlp = spacy.load('en_core_web_sm')

example = "Improve him believe opinion offered met and end cheered forbade. Friendly as stronger speedily by recurred. Son interest wandered sir addition end say. Manners beloved affixed picture men ask. Explain few led parties attacks picture company. On sure fine kept walk am in it. Resolved to in believed desirous unpacked weddings together. Nor off for enjoyed cousins herself. Little our played lively she adieus far sussex. Do theirs others merely at temper it nearer."
sentences = sent_tokenize(example)

tokens = word_tokenize(example)

filtered_tokens = [word for word in tokens if word not in string.punctuation]
#print(filtered_tokens)

stop_words = set(stopwords.words('english'))
filtered_words = [word for word in filtered_tokens if word.lower() not in stop_words]
#print(filtered_words)

stemmer = PorterStemmer()
stemmed_words = [stemmer.stem(word) for word in filtered_words]
#print(stemmed_words)

#lemmatizer = WordNetLemmatizer()
#lemmatized_words = [lemmatizer.lemmatize(word) for word in filtered_words]
#print(lemmatized_words)
filtered_text = ' '.join(str(element) for element in filtered_words)
doc = nlp(filtered_text)
lemmatized_words = [token.lemma_ for token in doc]
#print(doc)

pos_tags = pos_tag(filtered_words)
print(pos_tags)

app = Flask(__name__)


@app.route("/")
def home_page():
    return render_template("home.html", sentences=sentences, tokens=tokens, filtered_tokens=filtered_tokens, filtered_words=filtered_words, stemmed_words=stemmed_words, lemmatized_words=lemmatized_words, pos_tags=pos_tags)
@app.route('/try')
def try_page():
    return render_template("try.html")
@app.route('/process', methods=['POST'])
def process():
    if request.method == 'POST':
        user_input = request.form['text']
        sentences = sent_tokenize(user_input)
        tokens = word_tokenize(user_input)
        filtered_tokens = [word for word in tokens if word not in string.punctuation]
        stop_words = set(stopwords.words('english'))
        filtered_words = [word for word in filtered_tokens if word.lower() not in stop_words]
        stemmer = PorterStemmer()
        stemmed_words = [stemmer.stem(word) for word in filtered_words]
        filtered_text = ' '.join(str(element) for element in filtered_words)
        doc = nlp(filtered_text)
        lemmatized_words = [token.lemma_ for token in doc]
        pos_tags = pos_tag(filtered_words)

        return render_template('result.html',sentences=sentences, tokens=tokens, filtered_tokens=filtered_tokens, filtered_words=filtered_words, stemmed_words=stemmed_words, lemmatized_words=lemmatized_words, pos_tags=pos_tags, user_input=user_input)

if __name__ == '__main__':
    app.run(debug=True)
