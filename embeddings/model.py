from imports import word2vec,FastText

class W2VModel(object):
	
	def __init__(self, num_features = 300, min_word_count = 40, 
				num_workers = 4, context = 10, downsampling = 1e-3):

		# Set values for various parameters
		self.num_features = num_features    # Word vector dimensionality                      
		self.min_word_count = min_word_count   # Minimum word count                        
		self.num_workers = num_workers       # Number of threads to run in parallel
		self.context = context          # Context window size
		self.downsampling = downsampling   # Downsample setting for frequent words
		self.model = None

	def train(self, sentences, init_sims=True, loglevel='INFO'):
		import logging
		if loglevel == 'INFO':
			lv = logging.INFO
		logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',\
			level=lv)
		print("Training W2V model....")
		model = word2vec.Word2Vec(sentences, workers=self.num_workers,
								size=self.num_features, min_count=self.min_word_count,
								window=self.context, sample=self.downsampling)
		
		# If we don't plan to train the model any further, calling 
		# init_sims will make the model much more memory-efficient.
		model.init_sims(replace=init_sims)
		self.model = model
		return model

	def save(self, model_name):
		# model_name = "toxic_300features_40minwords_10context"
		self.model.save(model_name)


class FastTextModel(object):
	
	def __init__(self, num_features = 300, min_word_count = 40, 
				num_workers = 4, context = 10, downsampling = 1e-3):

		# Set values for various parameters
		self.num_features = num_features    # Word vector dimensionality                      
		self.min_word_count = min_word_count   # Minimum word count                        
		self.num_workers = num_workers       # Number of threads to run in parallel
		self.context = context          # Context window size
		self.downsampling = downsampling   # Downsample setting for frequent words
		self.model = None

	def train(self, sentences, loglevel='INFO'):
		import logging
		if loglevel == 'INFO':
			lv = logging.INFO
		logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',\
			level=lv)
		print("Training FastText model....")
		model = FastText(size=self.num_features, window=self.context, min_count=self.min_word_count)
		model.build_vocab(sentences=sentences)
		model.train(sentences=sentences, total_examples=len(sentences), epochs=10)
		# If we don't plan to train the model any further, calling 
		# init_sims will make the model much more memory-efficient.

		# model.init_sims(replace=init_sims)
		self.model = model
		return model

	def save(self, model_name):
		# model_name = "toxic_300features_40minwords_10context"
		self.model.save(model_name)

