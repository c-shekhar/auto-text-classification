from preprocess.converters import Converter
from embeddings.vectorize import Vectorize
from utils.read_write import Read, Write
from classifiers.model import ClassificationModels
from imports import train_test_split,np


def classify_using_bow():

	# train_path = '/home/chandrashekhar/Applied/auto_text_classif/data/jigsaw-toxic-comment-classification-challenge/train.csv'
	# test_path = '/home/chandrashekhar/Applied/auto_text_classif/data/jigsaw-toxic-comment-classification-challenge/test.csv'
	# text_field_name = 'comment_text'
	# target_col = 'toxic'


	train_path = '/home/chandrashekhar/Applied/auto_text_classif/data/tweet_sent/training.1600000.processed.noemoticon.csv'
	test_path = '/home/chandrashekhar/Applied/auto_text_classif/data/jigsaw-toxic-comment-classification-challenge/test.csv'
	text_field_name = 'Col6'
	target_col = 'Col1'

	split_frac = 0.25
	reader = Read()

	train, test = reader.read_from_csv(train_path,test_path)

	converter = Converter(train,target_col,text_field_name)

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

def classify_using_tfidf():

	# train_path = '/home/chandrashekhar/Applied/auto_text_classif/data/jigsaw-toxic-comment-classification-challenge/train.csv'
	# test_path = '/home/chandrashekhar/Applied/auto_text_classif/data/jigsaw-toxic-comment-classification-challenge/test.csv'
	# text_field_name = 'comment_text'
	# target_col = 'toxic'


	train_path = '/home/chandrashekhar/Applied/auto_text_classif/data/tweet_sent/training.1600000.processed.noemoticon.csv'
	test_path = '/home/chandrashekhar/Applied/auto_text_classif/data/jigsaw-toxic-comment-classification-challenge/test.csv'
	text_field_name = 'Col6'
	target_col = 'Col1'

	split_frac = 0.25
	reader = Read()

	train, test = reader.read_from_csv(train_path,test_path)

	converter = Converter(train,target_col,text_field_name)

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


def classify_using_w2v():
	pretrained = True

	train_path = '/home/chandrashekhar/Applied/auto_text_classif/data/jigsaw-toxic-comment-classification-challenge/train.csv'
	test_path = '/home/chandrashekhar/Applied/auto_text_classif/data/jigsaw-toxic-comment-classification-challenge/test.csv'
	text_field_name = 'comment_text'
	target_col = 'toxic'


	# train_path = '/home/chandrashekhar/Applied/auto_text_classif/data/tweet_sent/training.1600000.processed.noemoticon.csv'
	# test_path = '/home/chandrashekhar/Applied/auto_text_classif/data/jigsaw-toxic-comment-classification-challenge/test.csv'
	# text_field_name = 'Col6'
	# target_col = 'Col1'

	split_frac = 0.25
	reader = Read()

	train, test = reader.read_from_csv(train_path,test_path)

	converter = Converter(train,test,target_col,text_field_name)
	
	model_path = 'toxic_300features_40minwords_10context'
	if not pretrained:
		w2v_trn_data = converter.preprocess_w2v_train()
		from embeddings.model import W2VModel
		w2v_model = W2VModel()
		w2v_model.train(w2v_trn_data)
		w2v_model.save(model_path)
	
	words_rows = converter.preprocess_w2v()

	vectorizer = Vectorize()

	train_data_features = vectorizer.to_word_vectors(words_rows, model_path)

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

def classify_using_glove():
	pretrained = True

	train_path = '/home/chandrashekhar/Applied/auto_text_classif/data/jigsaw-toxic-comment-classification-challenge/train.csv'
	test_path = '/home/chandrashekhar/Applied/auto_text_classif/data/jigsaw-toxic-comment-classification-challenge/test.csv'
	text_field_name = 'comment_text'
	target_col = 'toxic'


	# train_path = '/home/chandrashekhar/Applied/auto_text_classif/data/tweet_sent/training.1600000.processed.noemoticon.csv'
	# test_path = '/home/chandrashekhar/Applied/auto_text_classif/data/jigsaw-toxic-comment-classification-challenge/test.csv'
	# text_field_name = 'Col6'
	# target_col = 'Col1'

	split_frac = 0.25
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

	train_data_features = vectorizer.to_glove_vectors(words_rows, model_path,25)

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

def classify_using_fasttext():
	pretrained = True

	train_path = '/home/chandrashekhar/Applied/auto_text_classif/data/jigsaw-toxic-comment-classification-challenge/train.csv'
	test_path = '/home/chandrashekhar/Applied/auto_text_classif/data/jigsaw-toxic-comment-classification-challenge/test.csv'
	text_field_name = 'comment_text'
	target_col = 'toxic'


	# train_path = '/home/chandrashekhar/Applied/auto_text_classif/data/tweet_sent/training.1600000.processed.noemoticon.csv'
	# test_path = '/home/chandrashekhar/Applied/auto_text_classif/data/jigsaw-toxic-comment-classification-challenge/test.csv'
	# text_field_name = 'Col6'
	# target_col = 'Col1'

	split_frac = 0.25
	reader = Read()

	train, test = reader.read_from_csv(train_path,test_path)

	converter = Converter(train,test,target_col,text_field_name)
	
	model_path = '/home/chandrashekhar/Applied/text-models/models/fasttext/wiki-news-300d-1M.vec'
	if not pretrained:
		w2v_trn_data = converter.preprocess_w2v_train()
		from embeddings.model import W2VModel
		w2v_model = W2VModel()
		w2v_model.train(w2v_trn_data)
		w2v_model.save(model_path)

	words_rows = converter.preprocess_w2v()

	vectorizer = Vectorize()

	train_data_features = vectorizer.to_fasttext_vectors(words_rows, model_path,300)

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

	# classify_using_bow()
	# classify_using_tfidf()
	# classify_using_w2v()
	# classify_using_glove()
	classify_using_fasttext()