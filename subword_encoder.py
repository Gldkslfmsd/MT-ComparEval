from tensor2tensor.data_generators import text_encoder
import sys
#e = vocab.encode("Dominik Macháček Macháčková")
#subt = [vocab._subtoken_ids_to_tokens([x]) for x in e]
#a = vocab._subtoken_ids_to_tokens(e)
#print(e,a, subt)
#print(len(e), len(a))
from functools import lru_cache

@lru_cache(maxsize=100000)
def subtokens(tok):
	tok = vocab.encode(tok)
	def d(t):
		if not t: return "_"
		return "+".join(t)
	r = " ".join(d(vocab._subtoken_ids_to_tokens([x])) for x in tok )
	if not r.endswith("_"):
		r += "_"
	return r

#	return " ".join("".join(vocab._subtoken_ids_to_tokens([x])) for x in vocab.encode(tok))
#	return " ".join(str(x) for x in vocab.encode(tok))

if __name__ == "__main__":
	vocab = text_encoder.SubwordTextEncoder(sys.argv[1])
	for line in sys.stdin:
		for tok in line.split():
			print(subtokens(tok), end=" ")
		print()
