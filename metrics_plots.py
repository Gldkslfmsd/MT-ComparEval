
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
					if not "baseline" in task.name:
#						xv /= 2
						pass
					else:
						if xv > 4.0 and False:  # TODO
							break

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
	tn = SMALL
	exp = PlotBleus(sys.argv[1], epoch=False, task_names=tn)
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
			  'figure.figsize': (10, 5),
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
		pass
#		def plot(self):
#			for task in self.tasks:
#				if self.task_names and task.name in self.task_names:
#					x,y = self.x_y_pairs(task)
#					plt.plot(x,y, label=rename(task.name, spaces=True))
#	#		plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
#			plt.legend()
#			if not self.save_to:
#				plt.show()
#			else:
#				plt.savefig(self.save_to, bbox_inches='tight')

	exp = _plotacc(sys.argv[1], epoch=epoch, task_names=tn, save_as="adeq_%s.pdf" % name)
	exp.plot()




def bleus_ne():
	d = ne_tasks()
	best_steps_acc(base=PlotBleus,name="bleus-ne",data=d)

def bleus_onmt():
	d = ONMT
	di = [
		('translate_decs_baseline',"T2T"),
		("onmtpy_baseline_lenfix","OpenNMT-py")
		]
	for t,n in di:
		rename_dict[t] = n
	best_steps_acc(base=PlotBleus,name="bleus-onmt",data=d,epoch=True)

def bleus_vocdesign_twice():
	d = VOCDESING_TWICE
	di = [
		('translate_decs_baseline',"T2T"),
		("onmtpy_baseline_lenfix","OpenNMT-py")
		]
	for t,n in di:
		rename_dict[t] = n
	best_steps_acc(base=PlotBleus,name="bleus-vocdesign-twice",data=d,epoch=False)



if __name__ == "__main__":
	
#	best_epochs()
	#small_steps()
#	general(DTAG)
#	general(POSsmall)
#	general(NE)
	#voc_design()

#	best_steps_acc()
	#ADEQ.remove('translate_decs_dtags_mtvoc')
	#ADEQ.remove('translate_decs_pos_full_mtvoc')
	#best_steps_acc(base=PlotBleus,name="bleus")
#	best_steps_acc(base=PlotBleus,name="bleus-adapt",data=ADAPT)

#	best_steps_acc(base=PlotBleus,name="bleus-sampling",data=SAMPLING)


	#bleus_ne()
	#bleus_onmt()

	bleus_vocdesign_twice()
