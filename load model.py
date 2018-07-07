# Classification template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('ANN.csv',encoding = "ISO-8859-1",nrows=50000)

#input from user 

question1=input("enter question 1:")
question2=input("enter question 2:")
#calculation of features from the input question
len_1=len(question1)
len_2=len(question2)
diff_len=len_1-len_2
len_char_q1=len(''.join(set(str(question1).replace(' ', ''))))
len_char_q2=len(''.join(set(str(question2).replace(' ', ''))))
len_word_q1=len(str(question1).split())
len_word_q2=len(str(question2).split())
common_words=len(set(str(question1).lower().split()).intersection(set(str(question2).lower().split())))

#fuzzy
from fuzzywuzzy import fuzz
fuzz_qratio=fuzz.QRatio(str(question1), str(question2))
fuzz_WRatio=fuzz.WRatio(str(question1), str(question2))
fuzz_partial_ratio=fuzz.partial_ratio(str(question1), str(question2))
fuzz_partial_token_set_ratio=fuzz.partial_token_set_ratio(str(question1), str(question2))
fuzz_partial_token_sort_ratio=fuzz.partial_token_sort_ratio(str(question1), str(question2))
fuzz_token_set_ratio=fuzz.token_set_ratio(str(question1), str(question2))
fuzz_token_sort_ratio=fuzz.token_sort_ratio(str(question1), str(question2))

#wmd
import gensim
model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin.gz', binary=True)
#sen2vec
import scipy
question1_vectors = scipy.sparse.lil_matrix((dataset.shape[0], 300))
question2_vectors = scipy.sparse.lil_matrix((dataset.shape[0], 300))
error_count = 0
from nltk import word_tokenize
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
def sent2vec(s):
    words = str(s).lower() # .decode('utf-8') ahile kaam lagdaina
    words = word_tokenize(words)
    words = [w for w in words if not w in stop_words]
    words = [w for w in words if w.isalpha()]
    M = []
    for w in words:
        try:
            M.append(model[w])
        except:
            continue
    M = np.array(M)
    v = M.sum(axis=0)
    return v / np.sqrt((v ** 2).sum())
question1_vector=sent2vec(question1)
question2_vector=sent2vec(question2)

#cosine distances
from scipy.spatial.distance import cosine, cityblock, jaccard, canberra, euclidean, minkowski, braycurtis
cosine_distance= cosine(question1_vector,question2_vector)
cityblock_distance=cityblock(question1_vector,question2_vector)
jaccard_distance=jaccard(question1_vector,question2_vector)
canberra_distance=canberra(question1_vector,question2_vector)
euclidean_distance=euclidean(question1_vector,question2_vector)
minkowski_distance=minkowski(question1_vector,question2_vector,3)
braycurtis_distance=braycurtis(question1_vector,question2_vector)

#prediction with classifier
question=np.array([[diff_len,common_words,fuzz_qratio,fuzz_WRatio,fuzz_partial_ratio,fuzz_partial_token_set_ratio,fuzz_partial_token_sort_ratio,
          fuzz_token_set_ratio,fuzz_token_sort_ratio,cosine_distance,cityblock_distance,canberra_distance,euclidean_distance,minkowski_distance,
          braycurtis_distance]])
from keras.models import load_model
model = load_model('ANNmodel.h5')
result=model.predict(question)