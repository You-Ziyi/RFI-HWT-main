import prd_imp as prd
import sys,getopt
import argparse
parser = argparse.ArgumentParser()
parser.despeciption='Please enter at least the filename'
parser.add_argument("-file","--Fitsfile",help="please input the flename",type=str)
parser.add_argument("-time", "--inputA",help="This is the time for rfifind (in sec)",type=float,default="0.5")
parser.add_argument("-sigma", "--inputB",help="This is the threshold for masking RFI data grid.",type=float,default="0.5")

args=parser.parse_args()

#filename=sys.argv[1]
p=prd.Data_pro(args.Fitsfile)
p.readfile(pola='I')
time=args.inputA
threshold=args.inputB
p.exrfi(time=time,threshold=threshold)

