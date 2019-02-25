from imports import bs,re,stopwords
import nltk

class Converter(object):
	def __init__(self, train_data, test_data, target_field_name, text_field_name):
		self.train_data = train_data
		self.test_data = test_data
		self.target_field = target_field_name
		self.text_field = text_field_name
		self.tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')


	def textblock_to_words(self, raw_text):
		review_soup = bs(raw_text)
		review_text = review_soup.get_text()
		review_letters_only = re.sub("[^a-zA-Z]"," ",raw_text)
		lower_case = review_letters_only.lower()
		words = lower_case.split()
		meaningful_words = [wd for wd in words if wd not in stopwords.words("english")]
		return(" ".join(meaningful_words))


	def textblock_to_wordslist(self, raw_text, remove_stopwords=False):
		review_soup = bs(raw_text)
		review_text = review_soup.get_text()
		review_letters_only = re.sub("[^a-zA-Z]"," ",raw_text)
		lower_case = review_letters_only.lower()
		words = lower_case.split()
		if remove_stopwords:
			meaningful_words = [wd for wd in words if wd not in stopwords.words("english")]
			return meaningful_words
		return words


	def textblock_to_sentences(self, raw_text, tokenizer, remove_stopwords=False):
		sentences = []
		tok_raw_text = tokenizer.tokenize(raw_text.strip())
		if len(tok_raw_text) > 0:
			for sent in tok_raw_text:
				sent_words = self.textblock_to_wordslist(sent, remove_stopwords)
				sentences.append(sent_words)
		return sentences


	def preprocess_countbased(self):
		clean_train_rows = []
		n_rows = self.train_data[self.text_field].size
		print(n_rows)
		print('-'*10)
		for i in range(n_rows):
			raw_text = self.train_data[self.text_field][i]
			clean_text = self.textblock_to_words(raw_text)
			clean_train_rows.append(clean_text)
			if ((i + 1) % 1000 == 0):
				print(f"{i+1} train rows cleaned...")
			if i == 15000:
				break
		return clean_train_rows


	def preprocess_w2v(self):
		clean_train_rows = []
		n_rows = self.train_data[self.text_field].size
		# print(n_rows)
		# print('-'*10)
		for i in range(n_rows):
			raw_text = self.train_data[self.text_field][i]
			clean_text = self.textblock_to_wordslist(raw_text,remove_stopwords=True)
			clean_train_rows.append(clean_text)
			if ((i + 1) % 1000 == 0):
				print(f"{i+1} train rows cleaned...")
			if i == 15000:
				break
		return clean_train_rows


	def preprocess_w2v_train(self):
		print('Preparing data for W2V/FastText...')
		tokenizer = self.tokenizer
		n_rows_train = self.train_data[self.text_field].size
		n_rows_test = self.test_data[self.text_field].size
		# print(n_rows)
		# print('-'*10)
		sentences = []
		for i in range(n_rows_train):
			raw_text = self.train_data[self.text_field][i]
			raw_text_sentences  = self.textblock_to_sentences(raw_text,tokenizer)
			sentences += raw_text_sentences
			# clean_train_reviews.append(review_feature_vec)
			if i == 15000:
				break
		
		for i in range(n_rows_test):
			raw_text = self.test_data[self.text_field][i]
			raw_text_sentences  = self.textblock_to_sentences(raw_text,tokenizer)
			sentences += raw_text_sentences
			# clean_train_reviews.append(review_feature_vec)
			if i == 15000:
				break

		return sentences

