# -*- coding: utf-8 -*-
import sys
import collections
import itertools
import numpy as np
from scipy.stats import ttest_1samp, wilcoxon, ttest_ind, mannwhitneyu

def Compute_2SampleWilcoxonStatSign(x1, x2):
	[stat, p_value] = mannwhitneyu(x1, x2)
	return p_value

def Compute_TtestStatSign(x1, x2):
	[stat, p_value] = ttest_ind(x1, x2)
	return p_value

def Compute_WilcoxonStatSign(x1, x2):
	[stat, p_value] = wilcoxon(x1, x2, 'wilcox', True)
	return p_value

print sys.argv[1]
print sys.argv[2]

inpf1 = open(sys.argv[1], 'rb')
inpf2 = open(sys.argv[2], 'rb')

x1 = []
x2 = []
for l in inpf1:
	if l.find('\t') == -1:
		continue

	l = l.strip()

	toks = l.split('\t')
	if len(toks) == 4:	
		x1.append(float(toks[3]))
inpf1.close()

for l in inpf2:
	if l.find('\t') == -1:
		continue

	l = l.strip()

	toks = l.split('\t')
	if len(toks) == 4:	
		x2.append(float(toks[3]))
inpf2.close()

print len(x1), len(x2)

#compare x1, x2
c = 0
for i in range(0, len(x2)):
	if x2[i] < x1[i]:
		c = c + 1

print 'better_pplx=' + str(c)

print "(Wilcoxon) p-value=" + str(Compute_WilcoxonStatSign(x1, x2))
print "(two-sample Wilcoxon) p-value=" + str(Compute_2SampleWilcoxonStatSign(x1, x2))
print "(T-test) p-value=" + str(Compute_TtestStatSign(x1, x2))

