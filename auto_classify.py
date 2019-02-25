from preprocess.converters import Converter
from embeddings.vectorize import Vectorize
from embeddings.model import W2VModel,FastTextModel
from utils.read_write import Read, Write
from classifiers.model import ClassificationModels
from imports import train_test_split,np


def classify_using_bow(train_data_path,
						text_field_name,
						target_col,
						test_data_path=None,
						split_frac=0.10):

	reader = Read()

	train, test = reader.read_from_csv(train_path,test_path)

	converter = Converter(train,test,target_col,text_field_name)

	clean_train_rows = converter.preprocess_countbased()

	vectorizer = Vectorize()
	
	train_data_features = vectorizer.to_count_vector(clean_train_rows)

	X_train,X_test,y_train,y_test = train_test_split(train_data_features[:15000],train[target_col][:15000].values,
													 test_size=split_frac,
													 random_state=0)

	clf_models = ClassificationModels()

	clf_models.create_random_forest()

	clf_models.train(X_train, y_train)
	
	preds = clf_models.predict(X_test)

	cm = clf_models.confusion_matrix(y_test,preds)
	print(cm)


def classify_using_tfidf(train_data_path,
						text_field_name,
						target_col,
						test_data_path=None,
						split_frac=0.10):

	reader = Read()

	train, test = reader.read_from_csv(train_path,test_path)

	converter = Converter(train,test,target_col,text_field_name)

	clean_train_rows = converter.preprocess_countbased()

	vectorizer = Vectorize()

	train_data_features = vectorizer.to_tfidf_vector(clean_train_rows)

	X_train,X_test,y_train,y_test = train_test_split(train_data_features[:15000],train[target_col][:15000].values,
													 test_size=split_frac,
													 random_state=0)

	clf_models = ClassificationModels()
	clf_models.create_random_forest()

	clf_models.train(X_train, y_train)
	
	preds = clf_models.predict(X_test)

	cm = clf_models.confusion_matrix(y_test,preds)
	print(cm)


def classify_using_w2v(pretrained,
						train_data_path,
						text_field_name,
						target_col,
						model_path,
						n_features,
						test_data_path=None,
						split_frac=0.10):

	reader = Read()

	train, test = reader.read_from_csv(train_path,test_path)

	converter = Converter(train,test,target_col,text_field_name)
	
	if not pretrained:
		w2v_trn_data = converter.preprocess_w2v_train()
		w2v_model = W2VModel()
		w2v_model.train(w2v_trn_data)
		w2v_model.save(model_path)
	
	words_rows = converter.preprocess_w2v()

	vectorizer = Vectorize()

	train_data_features = vectorizer.to_word_vectors(words_rows, model_path, n_features)

	X_train,X_test,y_train,y_test = train_test_split(train_data_features[:15000],train[target_col][:15000].values,
													 test_size=split_frac,
													 random_state=0)

	train_mask = np.all(np.isnan(X_train) | np.isinf(X_train), axis=1)

	test_mask = np.all(np.isnan(X_test) | np.isinf(X_test), axis=1)

	X_train,y_train = X_train[~train_mask],y_train[~train_mask]
	X_test,y_test = X_test[~test_mask],y_test[~test_mask]

	clf_models = ClassificationModels()
	clf_models.create_random_forest()

	clf_models.train(X_train, y_train)
	
	preds = clf_models.predict(X_test)

	cm = clf_models.confusion_matrix(y_test,preds)
	print(cm)

def classify_using_glove(pretrained,
						train_data_path,
						text_field_name,
						target_col,
						model_path,
						n_features,
						test_data_path=None,
						split_frac=0.10):
	
	reader = Read()

	train, test = reader.read_from_csv(train_path,test_path)

	converter = Converter(train,test,target_col,text_field_name)

	model_path = '/home/chandrashekhar/Applied/text-models/models/glove.twitter.27B/glove.twitter.27B.25d.txt'
	if not pretrained:
		w2v_trn_data = converter.preprocess_w2v_train()
		from embeddings.model import W2VModel
		w2v_model = W2VModel()
		w2v_model.train(w2v_trn_data)
		w2v_model.save(model_path)
	

	words_rows = converter.preprocess_w2v()

	vectorizer = Vectorize()

	train_data_features = vectorizer.to_glove_vectors(words_rows,model_path,n_features)

	X_train,X_test,y_train,y_test = train_test_split(train_data_features[:15000],train[target_col][:15000].values,
													 test_size=split_frac,
													 random_state=0)

	train_mask = np.all(np.isnan(X_train) | np.isinf(X_train), axis=1)

	test_mask = np.all(np.isnan(X_test) | np.isinf(X_test), axis=1)

	X_train,y_train = X_train[~train_mask],y_train[~train_mask]
	X_test,y_test = X_test[~test_mask],y_test[~test_mask]

	clf_models = ClassificationModels()
	clf_models.create_random_forest()

	clf_models.train(X_train, y_train)
	
	preds = clf_models.predict(X_test)

	cm = clf_models.confusion_matrix(y_test,preds)
	print(cm)


def classify_using_fasttext(pretrained,
							supervised,
							train_data_path,
							text_field_name,
							target_col,
							model_path,
							n_features,
							test_data_path=None,
							split_frac=0.10):

	reader = Read()

	train, test = reader.read_from_csv(train_path,test_path)

	converter = Converter(train,test,target_col,text_field_name)
	
	if not pretrained:
		ft_trn_data = converter.preprocess_w2v_train()
		ft_model = FastTextModel()
		ft_model.train(ft_trn_data)
		ft_model.save(model_path)

	words_rows = converter.preprocess_w2v()

	vectorizer = Vectorize()

	train_data_features = vectorizer.to_fasttext_vectors(words_rows,
														supervised=supervised,
														model_path=model_path,
														num_features=n_features)

	X_train,X_test,y_train,y_test = train_test_split(train_data_features[:15000],train[target_col][:15000].values,
													 test_size=split_frac,
													 random_state=0)

	train_mask = np.all(np.isnan(X_train) | np.isinf(X_train), axis=1)

	test_mask = np.all(np.isnan(X_test) | np.isinf(X_test), axis=1)

	X_train,y_train = X_train[~train_mask],y_train[~train_mask]
	X_test,y_test = X_test[~test_mask],y_test[~test_mask]

	clf_models = ClassificationModels()
	clf_models.create_random_forest()

	clf_models.train(X_train, y_train)
	
	preds = clf_models.predict(X_test)

	cm = clf_models.confusion_matrix(y_test,preds)
	print(cm)


if __name__ == '__main__':

	train_path = '/home/chandrashekhar/Applied/auto_text_classif/data/jigsaw-toxic-comment-classification-challenge/train.csv'
	test_path = '/home/chandrashekhar/Applied/auto_text_classif/data/jigsaw-toxic-comment-classification-challenge/test.csv'
	text_field_name = 'comment_text'
	target_col = 'toxic'
	
	# train_path = '/home/chandrashekhar/Applied/auto_text_classif/data/tweet_sent/training.1600000.processed.noemoticon.csv'
	# test_path = '/home/chandrashekhar/Applied/auto_text_classif/data/jigsaw-toxic-comment-classification-challenge/test.csv'
	# text_field_name = 'Col6'
	# target_col = 'Col1'

	# split_frac = 0.25
	# classify_using_bow(train_path,text_field_name,target_col,test_path,split_frac)
	# classify_using_tfidf(train_path,text_field_name,target_col,test_path,split_frac)
	
	## Using W2V
	# model_path = '/home/chandrashekhar/Applied/text-models/models/fasttext/wiki-news-300d-1M.vec'
	# pretrained = True
	# split_frac = 0.25
	# num_features = 300
	# classify_using_w2v(pretrained,train_path,text_field_name,target_col,num_featuresmodel_path,split_frac)

	# model_path = '/home/chandrashekhar/Applied/text-models/models/fasttext/wiki-news-300d-1M.vec'
	# pretrained = False
	# split_frac = 0.25
	# num_features = 300
	# classify_using_w2v(pretrained,train_path,text_field_name,target_col,num_featuresmodel_path,split_frac)

	## Using glove
	# model_path = '/home/chandrashekhar/Applied/text-models/models/glove.twitter.27B/glove.twitter.27B.25d.txt'
	# pretrained = True
	# split_frac = 0.25
	# num_features = 25
	# classify_using_glove(pretrained,train_path,text_field_name,target_col,num_features,model_path,split_frac)

	## Below, not implemented yet
	# model_path = '/home/chandrashekhar/Applied/text-models/models/fasttext/wiki-news-300d-1M.vec'
	# pretrained = False
	# split_frac = 0.25
	# num_features = 25
	# classify_using_glove(pretrained,train_path,text_field_name,target_col,num_features,model_path,split_frac)

	## Using fasttext
	# model_path = '/home/chandrashekhar/Applied/text-models/models/fasttext/wiki-news-300d-1M.vec'
	# pretrained = True
	# supervised = False
	# split_frac = 0.25
	# num_features = 300
	# classify_using_fasttext(pretrained,supervised,train_path,text_field_name,target_col,num_features,model_path,split_frac)
	
	## Below doesn't work
	# model_path = '/home/chandrashekhar/Applied/text-models/models/fasttext/crawl-300d-2M-subword/crawl-300d-2M-subword.bin'
	# pretrained = True
	# supervised = False
	# split_frac = 0.25
	# num_features = 300
	# classify_using_fasttext(pretrained,supervised,train_path,text_field_name,target_col,num_features,model_path,split_frac)

	# model_path = '/home/chandrashekhar/Applied/opensource/toxic_ft.bin'
	# pretrained = False
	# supervised = True
	# split_frac = 0.25
	# num_features = 300
	# classify_using_fasttext(pretrained,supervised,train_path,text_field_name,target_col,num_features,model_path,split_frac)

	# model_path = '/home/chandrashekhar/Applied/opensource/toxic_ft.bin'
	# pretrained = True
	# supervised = True
	# split_frac = 0.25
	# num_features = 300
	# classify_using_fasttext(pretrained,supervised,train_path,text_field_name,target_col,num_features,model_path,split_frac)