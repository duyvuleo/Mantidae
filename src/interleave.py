import sys
import collections
import itertools

#------
#input: ...
#ref0: ...
#ref1: ...
#...
#output0: ...
#output1: ...
#...
#------

if len(sys.argv) < 2: 
	print "Invalid argument(s)!"
	exit()

fis = []
for i in range(len(sys.argv) - 1):
	fis.append(open(sys.argv[i+1], "rb"))

for l_input in fis[0]:
	print "0) " + l_input.strip()
	for i in range(1, len(fis)):
		print str(i) + ") " + fis[i].readline().strip()

	print "-----------------------------------------------------"

for i in range(len(fis)):
	fis[i].flush()
	fis[i].close()

