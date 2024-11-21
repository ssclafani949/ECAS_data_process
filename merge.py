#!/usr/bin/env python
# 
# copyright  (C) 2010
# The Icecube Collaboration
# 
# $Id: merge.py 107762 2013-07-01 22:55:31Z nwhitehorn $
# 
# @version $Revision: 107762 $
# @date $LastChangedDate: 2013-07-01 17:55:31 -0500 (Mon, 01 Jul 2013) $
# @author Jakob van Santen <vansanten@wisc.edu> Last changed by: $LastChangedBy: nwhitehorn $
# 

# concatenate tables, keeping track of indexes and such automatically

from optparse import OptionParser
import os

parser = OptionParser()
parser.add_option("-f","--format",dest="format",help="format to output",default='hdf5')
parser.add_option("-z","--compress",dest="compress",help="compression level",default=1,type=int)
parser.add_option("-n","--frames",dest="nframes",help="number of frames to process",default=None,type=int)
parser.add_option("-o","--output",dest="outfile",help="name of the output file",default=None,type=str)


options,args = parser.parse_args()
if len(args) < 1:
    parser.error("You must supply at least one input file")
    
infiles = args
iformat = 'hdf5' # only service that supports reading at this point
oformat = options.format
if options.outfile is None:
    outfile = os.path.basename(infile) + '.' + options.format
else:
    outfile = options.outfile

from icecube import icetray,tableio
from icecube.tableio import I3TableTranscriber

# try to import the appropriate services
if 'hdf5' in [iformat,oformat]:
	from icecube.hdfwriter import I3HDFTableService
if 'root' in [iformat,oformat]:
	from icecube.rootwriter import I3ROOTTableService
if 'csv' in [iformat,oformat]:
	from icecube.textwriter import I3CSVTableService

if iformat == 'hdf5':
	inservices = [(I3HDFTableService,(infile,1,'r')) for infile in infiles]
elif iformat == 'root':
	inservices = [(I3ROOTTableService,(infile,'r')) for infile in infiles]
else:
	raise "Unknown input format '%s'" % iformat
	
if oformat == 'hdf5':
	outservice = I3HDFTableService(outfile,options.compress,'w')
elif oformat == 'root':
	outservice = I3ROOTTableService(outfile,options.compress)
elif oformat == 'csv':
	outservice = I3CSVTableService(outfile)
else:
	raise "Unknown out format '%s'" % oformat

for ctor,args in inservices:
    print('Merging %s'%args[0])
    inservice = ctor(*args)
    try:
        scribe = I3TableTranscriber(inservice,outservice)
    except RuntimeError as e:
        print(e)
        continue
    if options.nframes is not None:
        scribe.Execute(options.nframes)
    else:
        scribe.Execute()
    inservice.Finish()

outservice.Finish()
