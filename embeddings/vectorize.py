from imports import *
from preprocess.converters import Converter
from imports import Word2Vec,KeyedVectors

class Vectorize(object):
	def __init__(self):
		self.bow_model = None
		self.tf_idf_model = None
		self.w2v_model = None
		# self.converter = Converter()


	def to_count_vector(self, text_rows):
		vectorizer = CountVectorizer(analyzer='word',
									tokenizer=None,
									preprocessor=None,
									stop_words=None,
									max_features = 5000)
		vec_tfms = vectorizer.fit_transform(text_rows)
		return vec_tfms


	def to_tfidf_vector(self, text_rows):
		vectorizer = TfidfVectorizer(analyzer='word',
									tokenizer=None,
									preprocessor=None,
									stop_words=None,
									max_features = 5000)

		vec_tfms = vectorizer.fit_transform(text_rows)
		return vec_tfms


	def to_w2v(self, words, model, vocab, num_features, remove_stopwords):
		feature_vec = np.zeros((num_features,),dtype='float32')
		nwords = 0
		for wd in words:
			if wd in vocab:
				nwords += 1
				_vec = model.wv[wd]
				feature_vec = np.add(feature_vec,_vec)
		feature_vec = np.divide(feature_vec,nwords)
		return feature_vec


	def to_word_vectors(self, words_rows, model_path=None, num_features=300, remove_stopwords=True):
		model = Word2Vec.load(model_path)
		vocab = model.wv.index2word
		clean_train_rows = []
		for i,row in enumerate(words_rows):
			review_feature_vec = self.to_w2v(row,model,vocab,num_features,remove_stopwords)
			clean_train_rows.append(review_feature_vec)
			if ((i + 1) % 1000 == 0):
				print(f"{i+1} train rows vectorized...")
		return np.array(clean_train_rows)


	def to_glove_vectors(self, words_rows, model_path=None, num_features=300, remove_stopwords=True):
		model = KeyedVectors.load_word2vec_format(model_path, binary=False)
		vocab = model.wv.index2word
		clean_train_rows = []
		for i,row in enumerate(words_rows):
			review_feature_vec = self.to_w2v(row,model,vocab,num_features,remove_stopwords)
			clean_train_rows.append(review_feature_vec)
			if ((i + 1) % 1000 == 0):
				print(f"{i+1} train rows vectorized...")
		return np.array(clean_train_rows)


	def to_fasttext_vectors(self, words_rows, model_path=None, num_features=300, remove_stopwords=True):
		model = KeyedVectors.load_word2vec_format(model_path, binary=False)
		vocab = model.wv.index2word
		clean_train_rows = []
		for i,row in enumerate(words_rows):
			review_feature_vec = self.to_w2v(row,model,vocab,num_features,remove_stopwords)
			clean_train_rows.append(review_feature_vec)
			if ((i + 1) % 1000 == 0):
				print(f"{i+1} train rows vectorized...")
		return np.array(clean_train_rows)