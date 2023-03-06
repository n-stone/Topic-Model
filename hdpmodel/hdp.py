from gensim.utils import simple_preprocess
from gensim.corpora import Dictionary
from gensim.models import HdpModel
from transformers import BertTokenizer
import torch

import pyLDAvis.gensim
import pyLDAvis.gensim_models

documents = ["이것은 예쁜 고양이다.",
             "그것은 둥근 책상입니다.", 
             "이것은 귀여운 강아지다."]

texts = [[word for word in simple_preprocess(document)] for document in documents]
dictionary = Dictionary(texts)
corpus = [dictionary.doc2bow(text) for text in texts]

hdp_model = HdpModel(corpus=corpus, id2word=dictionary, alpha=0.1, gamma=0.1, eta=0.1, T=100)

for index, topic in hdp_model.show_topics(formatted=False, num_words=30):
    print('Topic: {} \nWords: {}'.format(index, [w[0] for w in topic]))
    
vis = pyLDAvis.gensim.prepare(hdp_model, corpus, dictionary, mds='mmds')
pyLDAvis.save_html(vis, "hdp_model.html") 