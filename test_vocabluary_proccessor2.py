from tflearn.data_utils import VocabularyProcessor

vocab = {'hello':3, '.':5, 'world':20, '/' : 10}
sentences = ['hello world . / hello', 'hello']

vocab_processor = VocabularyProcessor(max_document_length=6, vocabulary=vocab)
encoded = list(vocab_processor.transform(sentences))
print(encoded)