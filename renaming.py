rename_dict = {
	'translate_decs_baseline': 'baseline',
'translate_decs_mult_dheads': "Dheads",
'translate_decs_mult_repeat': "repeat",
'translate_decs_mult_demo_fbb100': "count~words,~vocshr",
'translate_decs_voc_mult_dheads':"Dheads,~voc:MT,~+tags:no",
'translate_decs_voc_mult_dtags':"Dtag,~voc:MT,~+tags:no",
'translate_decs_mult_dht':"Dht,~vocshr",
'translate_decs_dheads_and_baseline1m':"adapt~Dheads~on~MT~1M",
'translate_decs_voc_mult_pos_full':"POSfull,~voc:MT,~+tags:no",
'translate_decs_voc_mult_demo':"count~words,~voc:MT,~+tags:no",
'translate_decs_voc_mult_dht':"Dht,~voc:MT,~+tags:no",
'translate_decs_vff_mult_pos_small':"POSsmall,~voc:MT,~+tags:yes",
'translate_decs_mult_dtags':"Dtag,~vocshr",
'translate_decs_vff_mult_dtags':"Dtag,~voc:MT,~+tags:yes",
'translate_decs_vff_mult_pos_full':"POSfull,~voc:MT,~+tags:yes",
'translate_decs_vf_mult_dheads':"Dheads,~voc:MT,~+tags:yes",
'translate_decs_vf_mult_pos_small':"xxxxxxxxxx~POSsmall~vf",
'translate_decs_mult_pos_small':"POSsmall,~vocshr",
'translate_decs_dheads_and_baseline100k':"adapt~Dheads~on~MT~100k",
'translate_decs_voc_mult_dheads_part':"Dheads,~voc:MT,~+tags:no,~1:2samp",
'translate_decs_voc_ner_underline':"NE~voc:MT,~+tags:no",
'translate_decs_vf_mult_pos_full':"xxxxxxxxPOSfull~vf",
'translate_decs_mult_dheads_part':"Dheads,~1:2samp",
'translate_decs_voc_mult_pos_small':"POSsmall,~voc:MT,~+tags:no",
'translate_decs_voc_mult_repeat':"repeat,~voc:MT,~+tags:no",
'translate_decs_mult_pos_full':"POSfull,~vocshr",
'translate_decs_ner_inline':"NE~enrich",
'translate_decs_ner':"NE,~sampling:no,~gold",
'translate_decs_ner_underline':"NE",
'translate_decs_baseline_t': "single-task~MT+marker", 
'translate_decs_mult_dtags500k': "Dtag,~datasize:500k",
'translate_decs_baseline_t500k': "baseline,~datasize:500k",
'translate_decs_pos_full_mtvoc': "single-task~POSfull,~voc:MT,~+tags:yes",
'translate_decs_dtags_mtvoc': "single-task~Dtag,~voc:MT,~+tags:yes",
}

def _rename_spaces(k):
	print(k)
	v = rename_dict[k]
	return v.replace("~"," ")



def rename(task_name, spaces=False):
	if task_name in rename_dict:
		if spaces:
			return _rename_spaces(task_name)
		return rename_dict[task_name]
	return task_name


	
ALL = ['translate_decs_mult_dheads',
'translate_decs_mult_repeat',
'translate_decs_mult_demo_fbb100',
'translate_decs_voc_mult_dheads',
'translate_decs_voc_mult_dtags',
'translate_decs_mult_dht',
'translate_decs_dheads_and_baseline1m',
'translate_decs_voc_mult_pos_full',
'translate_decs_baseline',
'translate_decs_voc_mult_demo',
'translate_decs_voc_mult_dht',
'translate_decs_vff_mult_pos_small',
'translate_decs_mult_dtags',
'translate_decs_vff_mult_dtags',
'translate_decs_vff_mult_pos_full',
'translate_decs_vf_mult_dheads',
'translate_decs_vf_mult_pos_small',
'translate_decs_mult_pos_small',
'translate_decs_dheads_and_baseline100k',
'translate_decs_voc_mult_dheads_part',
'translate_decs_voc_ner_underline',
'translate_decs_vf_mult_pos_full',
'translate_decs_mult_dheads_part',
'translate_decs_voc_mult_pos_small',
'translate_decs_voc_mult_repeat',
'translate_decs_mult_pos_full',
'translate_decs_ner_inline',
'translate_decs_ner',
'translate_decs_ner_underline']


POS =  [
#'translate_decs_mult_repeat',
#'translate_decs_mult_demo_fbb100',
#'translate_decs_voc_mult_dheads',
#'translate_decs_voc_mult_dtags',
#'translate_decs_mult_dht',
#'translate_decs_dheads_and_baseline1m',
'translate_decs_voc_mult_pos_full',
'translate_decs_baseline',
#'translate_decs_voc_mult_demo',
#'translate_decs_voc_mult_dht',
'translate_decs_vff_mult_pos_small',
#'translate_decs_mult_dtags',
#'translate_decs_vff_mult_dtags',
'translate_decs_vff_mult_pos_full',
#'translate_decs_vf_mult_dheads',
#'translate_decs_vf_mult_pos_small',
'translate_decs_mult_pos_small',
#'translate_decs_dheads_and_baseline100k',
#'translate_decs_voc_mult_dheads_part',
#'translate_decs_voc_ner_underline',
#'translate_decs_vf_mult_pos_full',
#'translate_decs_mult_dheads_part',
'translate_decs_voc_mult_pos_small',
#'translate_decs_voc_mult_repeat',
'translate_decs_mult_pos_full',
#'translate_decs_ner_inline',
#'translate_decs_ner',
#'translate_decs_ner_underline'
]

BEST = [

'translate_decs_mult_repeat',
'translate_decs_mult_demo_fbb100',
'translate_decs_mult_dheads',
#'translate_decs_voc_mult_dtags',
#'translate_decs_mult_dht',
#'translate_decs_dheads_and_baseline1m',
#'translate_decs_voc_mult_pos_full',
'translate_decs_baseline',
#'translate_decs_voc_mult_demo',
#'translate_decs_voc_mult_dht',
#'translate_decs_vff_mult_pos_small',
'translate_decs_mult_dtags',
#'translate_decs_vff_mult_dtags',
#'translate_decs_vff_mult_pos_full',
#'translate_decs_vf_mult_dheads',
#'translate_decs_vf_mult_pos_small',
'translate_decs_mult_pos_small',
#'translate_decs_dheads_and_baseline100k',
#'translate_decs_voc_mult_dheads_part',
#'translate_decs_voc_ner_underline',
#'translate_decs_vf_mult_pos_full',
#'translate_decs_mult_dheads_part',
#'translate_decs_voc_mult_pos_small',
#'translate_decs_voc_mult_repeat',
'translate_decs_mult_pos_full',
#'translate_decs_ner_inline',
#'translate_decs_ner',
'translate_decs_ner_underline'
]

SMALL = [
'translate_decs_baseline',
'translate_decs_mult_dtags500k',
'translate_decs_baseline_t500k',
]

MISC = [ "translate_decs_baseline_t",
'translate_decs_baseline',]

#DTAG = list(filter(lambda x:"dtag" in x,_rename.keys()))
DTAG = ['translate_decs_mult_dtags', 
#'translate_decs_mult_dtags500k',
#'translate_decs_dtags_mtvoc', 
'translate_decs_vff_mult_dtags',
'translate_decs_voc_mult_dtags']
#print(DTAG)


#POS = list(filter(lambda x:"pos" in x,_rename.keys()))
#print(POS)
POSfull = [#'translate_decs_pos_full_mtvoc', 
#'translate_decs_vff_mult_pos_small',
#'translate_decs_vf_mult_pos_full', 
#'translate_decs_vf_mult_pos_small',
#'translate_decs_mult_pos_small', 
#'translate_decs_voc_mult_pos_small',
'translate_decs_voc_mult_pos_full', 
'translate_decs_vff_mult_pos_full',
'translate_decs_mult_pos_full'
]

POSsmall = [#'translate_decs_pos_full_mtvoc', 
'translate_decs_vff_mult_pos_small',
#'translate_decs_vf_mult_pos_full', 
#'translate_decs_vf_mult_pos_small',
'translate_decs_mult_pos_small', 
'translate_decs_voc_mult_pos_small',
#'translate_decs_voc_mult_pos_full', 
#'translate_decs_vff_mult_pos_full',
#'translate_decs_mult_pos_full'
]

#NE = list(filter(lambda x:"ne" in x,_rename.keys()))
#print(NE)
NE = [ 
'translate_decs_ner_inline',
'translate_decs_ner',
#'translate_decs_voc_ner_underline',
'translate_decs_ner_underline',
'translate_decs_ner_gold',
'translate_decs_ner_auto',
]


#DHEADS = list(filter(lambda x:"dheads" in x,_rename.keys()))
DHEADS = [
'translate_decs_voc_mult_dheads_part',
#'translate_decs_dheads_and_baseline100k',
'translate_decs_voc_mult_dheads',
'translate_decs_vf_mult_dheads', 
#'translate_decs_dheads_and_baseline1m',
'translate_decs_mult_dheads', 
'translate_decs_mult_dheads_part']

#COUNT = list(filter(lambda x:"demo" in x,_rename.keys()))
COUNT = ['translate_decs_mult_demo_fbb100', 'translate_decs_voc_mult_demo']

#REPEAT = list(filter(lambda x:"repeat" in x,_rename.keys()))
REPEAT = ['translate_decs_voc_mult_repeat', 'translate_decs_mult_repeat']

#DHT = list(filter(lambda x:"dht" in x,_rename.keys()))
DHT = ['translate_decs_mult_dht', 'translate_decs_voc_mult_dht']



voc_design_name_runs = [
	("Dtag", DTAG),
	("POSfull", POSfull),
	("POSsmall", POSsmall),
	("NE", NE),
	("Dheads", DHEADS),
	("count", COUNT),
	("repeat", REPEAT),
	("Dht", DHT),
	]


ADEQ = ['translate_decs_mult_dheads', 
'translate_decs_ner_underline',
'translate_decs_mult_pos_full', 
'translate_decs_mult_dht',
'translate_decs_mult_dtags', 
'translate_decs_mult_repeat',
'translate_decs_mult_demo_fbb100', 
'translate_decs_mult_pos_small',
'translate_decs_baseline',
'translate_decs_baseline_t',
#'translate_decs_pos_full_mtvoc',
#'translate_decs_dtags_mtvoc',
]

ADAPT = [ 
'translate_decs_baseline',
'translate_decs_baseline_t',
'translate_decs_dheads_and_baseline1m',
'translate_decs_dheads_and_baseline100k',
'translate_decs_mult_dheads',
]



SAMPLING = [
#'translate_decs_baseline',
'translate_decs_baseline_t',
'translate_decs_voc_mult_dheads_part',
#'translate_decs_mult_dheads_part',
'translate_decs_voc_mult_dheads'
#'translate_decs_mult_dheads',
]



###############
# NE

def ne_tasks():
	d = [
		('translate_decs_baseline',"baseline"),
		('translate_decs_baseline_t',"MT,~task~determination"),
		('translate_decs_ner_inline', "MT,~enrich~source~NE~auto"),
		('translate_decs_ner',"MT+NE~gold,~171:1~sampling"),
		('translate_decs_ner_underline',"MT+NE~auto,~1:1~sampling"),
		("translate_decs_ner_gold","NER2 gold"),
		("translate_decs_ner_auto","NER2 auto"),
		]
	for t,n in d:
		rename_dict[t] = n
	d = [ x for x,_ in d ]
	return d

ONMT = [
"translate_decs_baseline",
"onmtpy_baseline_lenfix",
]


VOCDESING_TWICE = [
"translate_decs_baseline",
"translate_decs_vocdesign_sourcetwice",
"translate_decs_vocdesign_targettwice",
]
