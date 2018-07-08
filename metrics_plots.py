
from metrics_table import *
import matplotlib.pyplot as plt 
import os

import seaborn as sns
#sns.set_palette(sns.color_palette("hls", 20))

from renaming import *


class ExperimentPlot(Experiment):

	img_dir_abs = "/home/d/Plocha/mff/diplomka/diplomka-text/img"
	save_to = None

	def x_y_pairs(self, task):
		raise NotImplemented()

	def plot(self):
		if self.task_names:
			print("tady",self.task_names)
			tasks = [ self.name_task_map[tn] for tn in self.task_names ]
		else:
			tasks = self.tasks

		for task in tasks:
				x,y = self.x_y_pairs(task)
				plt.plot(x,y, label=rename(task.name, spaces=True))
		plt.legend()
		if not self.save_to:
			plt.show()
		else:
			plt.savefig(self.save_to, bbox_inches='tight')

		
	
class PlotBleus(ExperimentPlot):

	DATA_CSV = "BLEU_cased_dev_trans.csv"
	
	def __init__(self, exp_path, epoch=True, task_names=None, save_to=None, save_as=None):
		super().__init__(exp_path)
		self.epoch = epoch
		self.task_names = task_names
		self.name_task_map = { t.name:t for t in self.tasks }
		print(self.name_task_map)
		self.save_to = save_to
		if save_as:
			self.save_to = os.path.join(self.img_dir_abs, save_as)

	def x_y_pairs(self, task):
		bleu_csv = os.path.join(task.path, self.DATA_CSV)
		x = []
		y = []
		with open(bleu_csv, "r") as f:
			for line in f.readlines()[1:]:
				_, step, score = line.split(",")
				if self.epoch:
					xv = int(step)/task.steps_in_epoch
#					if not "baseline" in task.name:
#						xv /= 2
#					else:
		#			if xv > 15.0:# and False:  # TODO
		#				break

				else:
					xv = int(step)

				x.append(xv)
				y.append(float(score))
		return x,y

class PlotAccuracy(PlotBleus):
	DATA_CSV =	"other_events-accuracy.csv"

tn = MISC
tn = BEST
def best_steps():
	tn = BEST
	exp = PlotBleus(sys.argv[1], epoch=False, task_names=tn)
	exp.plot()

def best_epochs():
#	tn = BEST
	exp = PlotBleus(sys.argv[1], epoch=True, task_names=tn)
	exp.plot()

def small_steps():
	tn = [
		'translate_decs_baseline',
		'translate_decs_mult_dtags500k',
		'translate_decs_baseline500k',
		'translate_decs_baseline_t500k',
	]
	di = [
		('translate_decs_baseline',"baseline, datasize:8.8M"),
		("translate_decs_mult_dtags500k","MT+Dtag, datasize:500k"),
		("translate_decs_baseline500k","baseline, datasize:500k"),
		('translate_decs_baseline_t500k',"baseline+TI mark, datasize:500k")
		]
	for t,n in di:
		rename_dict[t] = n

	exp = PlotBleus(sys.argv[1], epoch=False, task_names=tn, save_as="small_datasets_steps.pdf")
	exp.plot()

def general(tn):
	exp = PlotBleus(sys.argv[1], epoch=False, task_names=tn)
	exp.plot()

def voc_design():
	rel_fname = "img/voc-des-%s-steps.pdf"
	abs_path = "/home/d/Plocha/mff/diplomka/diplomka-text/"
	fname = abs_path + rel_fname
	latex_beg = r""""""
	subfigure = r"""\begin{subfigure}{.49\textwidth}
\centering
	\includegraphics[width=.95\linewidth]{%s}
	  \caption{%s}
		\label{fig:voc-des-%s}
		\end{subfigure}
"""
	latex_end = r"""\caption{plots of....}
\label{fig:fig}
\end{figure}
"""

	import matplotlib.pylab as pylab
	params = {'legend.fontsize': 'x-large',
		#	  'figure.figsize': (15, 5),
			   'axes.labelsize': 'x-large',
				'axes.titlesize':'x-large',
			 'xtick.labelsize':'x-large',
			  'ytick.labelsize':'x-large'}
	pylab.rcParams.update(params)

	right = False
	lat_final = latex_beg
	for n,runs in voc_design_name_runs:
		exp = PlotBleus(sys.argv[1], epoch=False, task_names=runs,
			save_to=fname % n)
		axes = plt.gca()
		axes.set_ylim([0.0,18.0])
		axes.set_xlim([0.0,10**6])
		exp.plot()
		plt.clf()

		lat_final += subfigure % ((rel_fname%n), n, n)
#		if not right:
#			right = True
#		else:
#	lat_final += latex_end

	with open(abs_path+"voc_figures.tex","w") as f:
		print(lat_final, file=f)




def best_steps_acc(base=PlotAccuracy,name="accuracy",data=ADEQ,epoch=False):
	import matplotlib.pylab as pylab
	params = {'legend.fontsize': 'x-large',
		#	  'figure.figsize': (15, 6),
			'figure.figsize': (10, 4),
			   'axes.labelsize': 'x-large',
				'axes.titlesize':'x-large',
			 'xtick.labelsize':'x-large',
			  'ytick.labelsize':'x-large'}
	pylab.rcParams.update(params)

	if base == PlotAccuracy:
		tn = [ b for b in data if b != 'translate_decs_baseline' and b!= 'translate_decs_baseline_t']
	else:
		tn = data

	print(tn)

	class _plotacc(base):
		def bold(self, task):
			if task.name in ['translate_decs_dummy_subwords',
'translate_decs_baseline_t',
'translate_decs_dummy_words',
'translate_decs_count_subwords',
'translate_decs_mult_demo_fbb100', 
'translate_decs_mult_repeat',]:
				return True#False
			else:
				return True

		def plot(self):
			if self.task_names:
				print("tady",self.task_names)
				tasks = [ self.name_task_map[tn] for tn in self.task_names ]
			else:
				tasks = self.tasks

			for task in tasks:
					x,y = self.x_y_pairs(task)
					if self.bold(task):
						lw = 2.5
						alpha = 0.9
					else:
						lw = 1
						alpha = 1
					plt.plot(x,y, label=rename(task.name, spaces=True), linewidth=lw, alpha=alpha)
#			plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
			plt.legend()
			if not self.save_to:
				plt.show()
			else:
				plt.savefig(self.save_to, bbox_inches='tight')

	exp = _plotacc(sys.argv[1], epoch=epoch, task_names=tn, save_as="adeq_%s.pdf" % name)
#	exp.save_to = False
	exp.plot()




def bleus_ne():
	d = ne_tasks()
	best_steps_acc(base=PlotBleus,name="bleus-ne",data=d)

def bleus_onmt():
	d = ONMT
	di = [
		('translate_decs_baseline',"T2T"),
		("onmtpy_baseline_lenfix","OpenNMT-py baseline, max len 1500, 4 shards"),
		("decs_baseline_lenwrong","OpenNMT-py, max len 50, 5k shards")
		]
	for t,n in di:
		rename_dict[t] = n
	best_steps_acc(base=PlotBleus,name="bleus-onmt",data=d,epoch=True)

def bleus_vocdesign_twice():
	d = VOCDESING_TWICE
	di = [
		('translate_decs_baseline',"T2T baseline"),
		("onmtpy_baseline_lenfix","OpenNMT-py baseline, max len 1500")
		]
	for t,n in di:
		rename_dict[t] = n
	best_steps_acc(base=PlotBleus,name="bleus-vocdesign-twice",data=d,epoch=False)

def bleus_onmt_multd(epoch=True):
	d = [
	'decs_baseline_lenwrong',
	'decs_dheadsofs_big',
	'decs_dtags_big',
	'decs_upos_big',
#	'translate_decs_baseline',
	]

	di = [
('decs_baseline_lenwrong', 'baseline, maxlen 50, shardsize 0.5MB'),
('decs_dheadsofs_big', 'MT+DheadsOffsets'),
('decs_dtags_big', 'MT+Dtags'),
('decs_upos_big', 'MT+UPOS'),

#		('translate_decs_baseline',"T2T"),
#		("onmtpy_baseline_lenfix","OpenNMT-py baseline, max len 1500"),
#		("onmtpy_baseline_lenwrong","OpenNMT-py, max len 50")
		]
	for t,n in di:
		rename_dict[t] = n
	if epoch:
		name = "bleus-onmt-multd" 
	else:
		name = "bleus-onmt-multd-step" 
	best_steps_acc(base=PlotBleus,name=name,data=d,epoch=epoch)


def bleus_dummy_base(name="2"):

	d = [
		"translate_decs_baseline",
		"translate_decs_baseline_t",
		"translate_decs_dummy_words",
		"translate_decs_mult_demo_fbb100",
		"translate_decs_dummy_subwords",
		"translate_decs_count_subwords",
		"translate_decs_voc_mult_repeat",
		]
	di = dummy_names()
#	di = dummy_names_mt()
	for t,n in di:
		rename_dict[t] = n
	best_steps_acc(base=PlotBleus,name="bleus-conll-fig"+n,data=d,epoch=False)

def bleus_dummy_eqlines(name="eqlines"):

	d = [
#		"translate_decs_baseline",
#		"translate_decs_baseline_t",
		"translate_decs_dummy_words",
		"translate_decs_mult_demo_fbb100",
		"translate_decs_dummy_subwords",
		#"translate_decs_count_subwords",
		"translate_decs_voc_mult_repeat",
		]
	di = dummy_names()

	for t,n in di:
		rename_dict[t] = n
	best_steps_acc(base=PlotBleus,name="bleus-conll-fig"+name,data=d,epoch=False)

def bleus_best_connl(epoch=True):

	di = [
		('translate_decs_baseline',"MT Baseline"),
		('translate_decs_mult_dheads',"MT+DepHeads"),
		('translate_decs_mult_dtags',"MT+DepLabels"),
		('translate_decs_mult_dht',"MT+DepHeads+DepLabels"),
	#	('translate_decs_mult_pos_full',"MT+POS"),
	#	('translate_decs_ner_underline',"MT+NE"),
		]
	for t,n in di:
		rename_dict[t] = n

	d = [ t for t,_ in di ]
	if epoch:
		e = "-epoch"
	else:
		e = ""
	best_steps_acc(base=PlotBleus,name="bleus-conll-fig3"+e,data=d,epoch=epoch)

def dummy_names():
	di = [
		('translate_decs_baseline',"MT Baseline"),
		('translate_decs_baseline_t',"MT TaskID"),
		("translate_decs_dummy_words","MT+EnumSrcWords"),
		("translate_decs_mult_demo_fbb100","MT+CountSrcWords"),
		("translate_decs_dummy_subwords","MT+EnumSrcSubwords"),
		("translate_decs_voc_mult_repeat","MT+RepeatSrc"),
		("translate_decs_count_subwords","MT+CountSrcSubwords"),
		]
	return di

def dummy_names_mt():
	di = [
		('translate_decs_baseline',"baseline"),
		('translate_decs_baseline_t',"baseline+TI marker"),
		("translate_decs_dummy_words","enum words"),
		("translate_decs_mult_demo_fbb100","count words"),
		("translate_decs_dummy_subwords","enum subwords"),
		("translate_decs_voc_mult_repeat","repeat"),
		("translate_decs_count_subwords","count subwords"),
		]
	return di


if __name__ == "__main__":

	onmt = [
	'decs_upos_big',
	'decs_dtags_big',
'decs_dheadsofs_big',
]

	
#	best_epochs()
#	small_steps()
#	general(DTAG)
#	general(POSsmall)
#	general(NE)
	#voc_design()

	#best_steps_acc()
	#ADEQ.remove('translate_decs_dtags_mtvoc')
	#ADEQ.remove('translate_decs_pos_full_mtvoc')

	#best_steps_acc(base=PlotBleus,name="bleus")
	#best_steps_acc(base=PlotBleus,name="bleus2",data=ADEQ[::-1])
#	best_steps_acc(base=PlotBleus,name="bleus-adapt",data=ADAPT)

#	best_steps_acc(base=PlotBleus,name="bleus-sampling",data=SAMPLING)


	#bleus_ne()
	#bleus_onmt()
	#bleus_onmt_multd()
#	bleus_onmt_multd(False)



	#bleus_vocdesign_twice()
	#bleus_dummy_eqlines()
	#bleus_dummy_base()
	#bleus_best_connl(False)
	#bleus_best_connl()
	best_steps_acc(data=onmt, name="acc-onmt")
