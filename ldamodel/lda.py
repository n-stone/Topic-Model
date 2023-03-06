import gensim
from gensim import corpora, models
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
import pyLDAvis.gensim_models as gensimvis
import pyLDAvis
import nltk

nltk.download('stopwords')

tokenizer = RegexpTokenizer(r'\w+')
stop_words = set(stopwords.words('english'))
documents = ["이것은 예쁜 고양이다.",
             "그것은 둥근 책상입니다.",
             "이것은 귀여운 강아지다."]

texts = []
for doc in documents:
    raw = doc.lower()
    tokens = tokenizer.tokenize(raw)
    stopped_tokens = [t for t in tokens if not t in stop_words]
    texts.append(stopped_tokens)

dictionary = corpora.Dictionary(texts)
corpus = [dictionary.doc2bow(text) for text in texts]
tfidf = models.TfidfModel(corpus)

lda_model = gensim.models.ldamodel.LdaModel(corpus=tfidf[corpus], id2word=dictionary, num_topics=14, passes=10)

topics = lda_model.print_topics(num_words=5)
for topic in topics:
    print(topic)

vis_data = gensimvis.prepare(lda_model, corpus, dictionary, mds='mmds')
pyLDAvis.save_html(vis_data, "lda_model.html")