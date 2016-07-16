import os
import pdb

for (dirpath, dirnames, filenames) in os.walk('.'):
	print '---'
	print 'dirpath', dirpath
	print 'dirnames', dirnames
	print 'filenames', filenames
	#pdb.set_trace()
