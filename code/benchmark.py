#!/usr/bin/python

import subprocess
import sys
import os
import os.path
import getopt
import math
import datetime
import random

import rutil

def usage(fname):
    
    ustring = "Usage: %s [-h][-Q] [-k K] [-b BENCHLIST] [-n NSTEP] [-u UPDATE] [-t THREADLIMIT] [-i ID] [-f OUTFILE]" % fname
    print ustring
    print "    -h            Print this message"
    print "    -Q            Quick mode: Don't compare with reference solution"
    print "    -k            Specify graph dimension"
    print "    -b BENCHLIST  Specify which benchmark(s) to perform as substring of 'ABCDEF'"
    print "    -n NSTEP      Specify number of steps to run simulations"
    print "    -u UPDATE     Specify update mode"
    print "       r: rat order"
    print "       s: synchronous"
    print "       b: batch"
    print "    -t THREADLIMIT Specify number of OMP threads"
    print "       If > 1, will run crun-omp.  Else will run crun-seq"
    print "    -i ID         Specify unique ID for distinguishing check files"
    print "    -f OUTFILE    Create output file recording measurements"
    print "         If file name contains field of form XX..X, will replace with ID having that many digits"
    sys.exit(0)

# General information
simProgram = "./crun-seq"
ompSimProgram = "./crun-omp"
refSimProgram = "./crun-soln"
dataDirectory = "./data"

outFile = None

doCheck = True
saveDirectory = "./check"

testFileName = ""
referenceFileName = ""

# Grading parameters
pointsPerRun = 15
lowerThreshold = 0.5
upperThreshold = 0.9

# How many mismatched lines warrant detailed report
mismatchLimit = 5

# Graph/rat combinations: name : (graphType, ratType)
benchmarkDict = {
    'A': ('t', 'u'),
    'B': ('h', 'u'),
    'C': ('v', 'u'),
    'D': ('p', 'u'),
    'E': ('v', 'r'),
    'F': ('p', 'd'),
    }

# k/load factor combinations
loadFactorDict = {12:4, 36:10, 60:25, 180:32}

defaultDimension = 180
defaultSteps = 100
defaultMode = "b"
defaultTests = "".join(sorted(benchmarkDict.keys()))
defaultThreadCount = 12

uniqueId = ""

def outmsg(s, noreturn = False):
    if len(s) > 0 and s[-1] != '\n' and not noreturn:
        s += "\n"
    sys.stdout.write(s)
    sys.stdout.flush()
    if outFile is not None:
        outFile.write(s)


def graphFileName(graphDimension, graphType):
    k = graphDimension
    return dataDirectory + "/g-%s%.3dx%.3d.gph" % (graphType, k, k)

def ratFileName(graphDimension, ratType):
    loadFactor = loadFactorDict[graphDimension]
    k = graphDimension
    return dataDirectory + "/r-%.3dx%.3d-%s%d.rats" % (k, k, ratType, loadFactor)

def testName(params):
    root = "%.3d-%s-%s-%.2d-%.2d-%s-%.2d" % params
    if uniqueId != "":
        root +=  ("-" + uniqueId)
    return root + ".txt"

def saveFileName(useRef, graphDimension, graphType, ratType, loadFactor, stepCount, updateFlag, seed = rutil.DEFAULTSEED):
    params = (graphDimension, graphType, ratType, loadFactor, stepCount, updateFlag, seed)
    return saveDirectory + "/" + ("ref" if useRef else "tst") + testName(params)

def trim(s):
    while len(s) > 0 and s[-1] in '\r\n':
        s = s[:-1]
    return s

def checkOutputs(referenceFile, testFile, testName):
    if referenceFile == None or testFile == None:
        return True
    badLines = 0
    lineNumber = 0
    while True:
        rline = referenceFile.readline()
        tline = testFile.readline()
        lineNumber +=1
        if rline == "":
            if tline == "":
                break
            else:
                badLines += 1
                outmsg("Test %s.  Mismatch at line %d.  Reference simulation ended prematurely" % (testName, lineNumber))
                break
        elif tline == "":
            badLines += 1
            outmsg("Test %s.  Mismatch at line %d.  Simulation ended prematurely\n" % (testName, lineNumber))
            break
        rline = trim(rline)
        tline = trim(tline)
        if rline != tline:
            badLines += 1
            if badLines <= mismatchLimit:
                outmsg("Test %s.  Mismatch at line %d.  Expected result:'%s'.  Simulation result:'%s'\n" % (testName, lineNumber, rline, tline))
    referenceFile.close()
    testFile.close()
    if badLines > 0:
        outmsg("%d total mismatches.\n" % (badLines))
    return badLines == 0


def runBenchmark(useRef, name, graphDimension, stepCount, updateType, threadCount, otherArgs):
    global referenceFileName, testFileName
    nodes = graphDimension * graphDimension
    gtype, rtype = benchmarkDict[name]
    load = loadFactorDict[graphDimension]
    gfname = graphFileName(graphDimension, gtype)
    rfname = ratFileName(graphDimension, rtype)
    params = [name, "%5d" % graphDimension, gtype, "%4d" % load, rtype, str(stepCount), updateType]
    cacheKey = ":".join(params)
    results = params + [str(threadCount)]
    prog = refSimProgram if useRef else simProgram if threadCount == 1 else ompSimProgram
    clist = ["-g", gfname, "-r", rfname, "-u", updateType, "-n", str(stepCount), "-t", str(threadCount)] + otherArgs
    simFile = subprocess.PIPE
    if not useRef:
        params = (graphDimension, gtype, rtype, load, stepCount, updateType, rutil.DEFAULTSEED)
        name = testName(params)
        print "+++++++++++++++++ Benchmark %s" % name
    if doCheck:
        if not os.path.exists(saveDirectory):
            try:
                os.mkdir(saveDirectory)
            except Exception as e:
                outMsg("Couldn't create directory '%s' (%s)" % (saveDirectory, str(e)))
                simFile = subprocess.PIPE
        clist += ["-i", str(stepCount)]
        fname = saveFileName(useRef, graphDimension, gtype, rtype, load, stepCount, updateType)
        try:
            simFile = open(fname, "w")
            if useRef:
                referenceFileName = fname
            else:
                testFileName = fname
        except:
            print "Couldn't open output file '%s'" % fname
            simFile = subprocess.PIPE
    else:
        clist += ["-q"]
    cmd = [prog] + clist
    cmdLine = " ".join(cmd)
    tstart = datetime.datetime.now()
    try:
        print "Running '%s'" % cmdLine
        simProcess = subprocess.Popen(cmd, stdout = simFile, stderr = subprocess.PIPE)
        simProcess.wait()
        if simFile != subprocess.PIPE:
            simFile.close()
        returnCode = simProcess.returncode
        # Echo any results printed by simulator on stderr onto stdout
        for line in simProcess.stderr:
            sys.stdout.write(line)
    except Exception as e:
        print "Execution of command '%s' failed. %s" % (cmdLine, e)
        return None
    if returnCode == 0:
        delta = datetime.datetime.now() - tstart
        secs = delta.seconds + 24 * 3600 * delta.days + 1e-6 * delta.microseconds
        rmoves = (nodes * load) * stepCount
        npm = 1e9 * secs/rmoves
        results.append("%.2f" % secs)
        results.append("%.2f" % npm)
        return results
    else:
        print "Execution of command '%s' gave return code %d" % (cmdLine, returnCode)
        return None

def score(npm, rnpm):
    if npm == 0.0:
        return 0
    ratio = rnpm/npm
    nscore = 0.0
    if ratio >= upperThreshold:
        nscore = 1.0
    elif ratio >= lowerThreshold:
        nscore = (ratio-lowerThreshold)/(upperThreshold - lowerThreshold)
    return int(math.ceil(nscore * pointsPerRun))

def formatTitle():
    ls = ["Name", "Dim", "gtype", "lf", "rtype", "steps", "update", "threads", "secs", "NPM"]
    if doCheck:
        ls += ["BNPM", "Ratio", "Pts"]
    return "\t".join(ls)

def sweep(testList, graphDimension, updateType, stepCount, threadCount, otherArgs):
    tcount = 0
    rcount = 0
    sum = 0.0
    refSum = 0.0
    resultList = []
    cresults = None
    totalPoints = 0
    for t in testList:
        ok = True
        nodes = graphDimension * graphDimension
        results = runBenchmark(False, t, graphDimension, stepCount, updateType, threadCount, otherArgs)
        if results is not None and doCheck:
            cresults = runBenchmark(True, t, graphDimension, stepCount, updateType, threadCount, otherArgs)
            if referenceFileName != "" and testFileName != "":
                try:
                    rfile = open(referenceFileName, 'r')
                except:
                    rfile = None
                    print "Couldn't open reference simulation output file '%s'" % referenceFileName
                    ok = False
                try:
                    tfile = open(testFileName, 'r')
                except:
                    tfile = None
                    print "Couldn't open test simulation output file '%s'" % testFileName
                    ok = False
                if rfile is not None and tfile is not None:
                    ok = checkOutputs(rfile, tfile, t)
        if results is not None:
            tcount += 1
            npm = float(results[-1])
            sum += npm
            if cresults is not None:
                rcount += 1
                cnpm = float(cresults[-1])
                refSum += cnpm
                ratio = cnpm/npm if npm > 0 else 0
                points = score(npm, cnpm)
                totalPoints += points
                results += [cresults[-1], "%.3f" % ratio, "%d" % points]
            resultList.append(results)
    print "+++++++++++++++++"
    outmsg(formatTitle())
    for r in resultList:
        outmsg("\t".join(r))
    if tcount > 0:
        avg = sum/tcount
        astring = "AVG:\t\t\t\t\t\t\t\t\t%.2f" % avg
        if refSum > 0:
            ravg = refSum/rcount
            astring += "\t%.2f" % ravg
        outmsg(astring)
        if doCheck:
            tstring = "TOTAL:\t\t\t\t\t\t\t\t\t\t\t\t%d" % totalPoints
            outmsg(tstring)

def generateFileName(template):
    global uniqueId
    myId = ""
    n = len(template)
    ls = []
    for i in range(n):
        c = template[i]
        if c == 'X':
            c = chr(random.randint(ord('0'), ord('9')))
        ls.append(c)
        myId += c
    if uniqueId == "":
        uniqueId = myId
    return "".join(ls) 

def run(name, args):
    global outFile, doCheck
    global uniqueId
    graphDimension = defaultDimension
    nstep = defaultSteps
    testList = list(defaultTests)
    updateType = defaultMode
    threadCount = defaultThreadCount
    optString = "hQk:b:n:u:t:i:f:"
    optlist, args = getopt.getopt(args, optString)
    otherArgs = []
    for (opt, val) in optlist:
        if opt == '-h':
            usage(name)
        elif opt == '-Q':
            doCheck = False
        elif opt == '-k':
            graphDimension = int(val)
            if graphDimension not in loadFactorDict:
                outmsg("Invalid graph dimension %d" % graphDimension)
                usage(name)
        elif opt == '-b':
            testList = list(val)
        elif opt == '-n':
            nstep = int(val)
        elif opt == '-i':
            uniqueId = val
        elif opt == '-f':
            fname = generateFileName(val)
            try:
                outFile = open(fname, "w")
            except Exception as e:
                outFile = None
                outmsg("Couldn't open output file '%s'" % fname)
        elif opt == '-u':
           updateType = val
        elif opt == '-t':
            threadCount = int(val)
        else:
            outmsg("Unknown option '%s'" % opt)
            usage(name)
    
    tstart = datetime.datetime.now()

    sweep(testList, graphDimension, updateType, nstep, threadCount, otherArgs)
    delta = datetime.datetime.now() - tstart
    secs = delta.seconds + 24 * 3600 * delta.days + 1e-6 * delta.microseconds
    print "Total test time = %.2f secs." % secs

if __name__ == "__main__":
    run(sys.argv[0], sys.argv[1:])
