import nltk
import sacrebleu as sb
from bleu_hook import bleu_wrapper 
from utils import *
from collections import defaultdict

def log(msg):
	print(msg)

class InsertSentences:

	UN = ""

	def __init__(self, conn):
		self.conn = conn

	def ngrams(self, seq, N):
		return list(zip(*tuple(seq[i:] for i in range(N))))

	

	def insert_sentences(self, task_id, task_path, ref_path):
		c = self.conn.cursor()
		ref = open(ref_path, "r")
		with open(task_path, "r") as translation:
			i = 1
			for r,h in zip(ref,translation):
				t = escape(h)
				cmd = """insert into translations (tasks_id, sentences_id, text)
				values (%d, %d, '%s');""" % (task_id, i, t)
				s = c.execute(cmd)
				i += 1
		ref.close()
		self.conn.commit()


		cmd = "".join("""select id from translations where tasks_id=%d
				and sentences_id=%d;""" % (task_id, j) for j in range(1,i-1))
		print(cmd[:1000])
		s = c.executescript(cmd)
		print("tady")
		sent_ids = [ j for j in range(1,i-1) ]
		print(sent_ids)

		ref = open(ref_path, "r")
		with open(task_path, "r") as translation:
			i = 1
			for r,h,sent_id in zip(ref,translation,sent_ids):
				t = escape(h)
				self.firmed_ngrams(sent_id, r,h, un="")
				self.firmed_ngrams(sent_id, r,h, un="un")
				if not (i%100):
					self.conn.commit()
				i += 1
		self.conn.commit()



	def firmed_ngrams(self, sent_id, ref, hyp, un=""):
		i = 0
		r = ref.split()
		h = hyp.split()
		for n in range(1,5):
			rngrams = set(self.ngrams(r,n))
			hngrams = self.ngrams(h,n)
			positions = defaultdict(lambda: 0)
			for hg in hngrams:
				if (not un and hg in rngrams) or (un and not hg in rngrams):
					cmd = """insert into %sconfirmed_ngrams
					(translations_id, text, length, position) values
					(%d, '%s', %d, %d);""" % (un, sent_id, 
							escape(" ".join(hg)),
							n, 
							positions[hg])
			#		print(cmd)
					c = self.conn.cursor()
					c.execute(cmd)
					i += 1
					positions[hg] += 1


		


class MetricBase:

	NAME = "abstract_metric"
	METRIC_ID = None
	CASED = True

	def __init__(self, conn):
		self.conn = conn

	def score(self, ref, hyp):
		raise NotImplemented()

	def read_process(self, fn):
		with open(fn, "r") as f:
			return f.read()
#			if self.CASED:
#				return [ s.lower() for s in f.readlines() ]
#			return [ s for s in f.readlines() ]

	def task_metric(self, task_id, trans_file, ref_file):
		score = self.score(trans_file, ref_file)
		log("counted %s for %s and %s: %2.2f" % (self.NAME, trans_file, ref_file, score))

		cmd = """insert into tasks_metrics (tasks_id, metrics_id, score)
		values (%s, %s, %2.2f);""" % (task_id, self.METRIC_ID, score)

		c = self.conn.cursor()
		c.execute(cmd)
		self.conn.commit()

	
class BleuMetric(MetricBase):

	NAME = "BLEU"
	METRIC_ID = 2
	CASED = False

	def score(self, ref, hyp):
		# nltk
#		n = len(ref)
#		b = sum(nltk.translate.bleu_score.sentence_bleu([r], h) for r,h in zip(ref, hyp))/n
#		return b*100

		# sacrebleu
#		b = sb.corpus_bleu(hyp, ref)
#		print(b)
#		return b.score

		# t2t bleu_hook:
		b = bleu_wrapper(ref, hyp, case_sensitive=self.CASED)
		return b*100

class BleuCasedMetric(BleuMetric):
	NAME = "BLEU-cased"
	METRIC_ID = 1
	CASED = True
