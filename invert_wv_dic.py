from gensim.models import KeyedVectors
import json
import hashlib
wv_file="embeddings/vectors.kv"
wv = KeyedVectors.load(wv_file, mmap='r')
inverted_wv = dict()
wv.init_sims()

for word in wv.vocab :
    vect = ["{:0.3f}".format(x) for x in wv.word_vec(word, use_norm=True)]
    #vect = [str(x)[:4] for x in vect]
    hashed  = hashlib.sha1(str.encode(str(vect)))
    inverted_wv[hashed.hexdigest()] = word

refs = json.dumps(inverted_wv,indent=4)
f = open("models/inverted_wv.json", "w")
f.write(refs)
f.close()