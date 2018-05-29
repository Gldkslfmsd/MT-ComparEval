from collections import Counter
import math

from tensor2tensor.data_generators import text_encoder


def log(x):
	return math.log(x, 10)

class BaseSubwordEncoder:
	
	def encode(self, word):
		raise NotImplemented()

	def num_of_subwords(self, word):
#		return self.encode(word).count(" ")+1
		return len(self.encode(word))

	def avg_subwords_in_file(self, filename):
		subw = 0
		words = 0
		with open(filename, "r") as f:
			for line in f:
				line = line.split()
				words += len(line)
				subw += sum(self.num_of_subwords(w) for w in line)
		print(self.vocab_file, filename, subw, words)
		return subw/words

	def dict_from_file(self, filename):
		subword_dict = set()
		with open(filename, "r") as f:
			for line in f:
				line = line.split()
				subword_dict.update(set(line))
		return subword_dict




class SubwordTextEncoder(BaseSubwordEncoder):

	def __init__(self, vocab):
		self.vocab = text_encoder.SubwordTextEncoder(vocab)
		self.vocab_file = vocab

	def encode(self, word):
		return tuple(self.vocab.encode(word))
#		return "@@ ".join("".join(self.vocab._subtoken_ids_to_tokens([x])) for x in self.vocab.encode(word))


class NoEncoder(BaseSubwordEncoder):
	def encode(self, word):
		return word



def demo(vocabs):
	encoders = [ (voc,SubwordTextEncoder(voc)) for i,voc in enumerate(vocabs) ]
	for n,e in encoders:
		print(n)
		for w in "ŽŽŽTranslate ŽŽŽPOS ŽŽŽPERSON ORGANIZATION".split():
			x = e.encode(w)
			print(x)
			print(e.num_of_subwords(w))
		print()
		print()

def find_differences(v1, v2, filename):
	a = SubwordTextEncoder(v1)
	b = SubwordTextEncoder(v2)
	with open(filename, "r") as f:
		for line in f:
			line = line.split()
			for w in line:
#				sa = a.num_of_subwords(w) 
#				sb = b.num_of_subwords(w)
				sa = a.encode(w)
				sb = b.encode(w)
				if sa != sb:
					print(w, sa, sb)

def avg(v1, v2, filename):
	a = SubwordTextEncoder(v1)
	b = SubwordTextEncoder(v2)
	print(a.avg_subwords_in_file(filename), b.avg_subwords_in_file(filename))


if __name__ == "__main__":
	import sys
	#demo(sys.argv[1:])
#	find_differences(sys.argv[1], sys.argv[2], sys.argv[3])
	avg(sys.argv[1], sys.argv[2], sys.argv[3])
