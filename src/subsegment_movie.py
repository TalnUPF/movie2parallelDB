# -*- coding: utf-8 -*-
from optparse import OptionParser
import os
import sys
import re
import subprocess
import csv
import cPickle
import xml.etree.ElementTree as ET
import credentials
import speech_recognition as sr
from nltk.tokenize import RegexpTokenizer
from datetime import *

#CONSTANTS
TIME_ROUNDUP_FOR_AVCONV = 3
TEMP_SUB_WAV_BUFFER = 0.5  
SCRIBE_AVG_SCORE_THRESHOLD = 0.5
SENTENCE_END_BUFFER = 0
END_WORD_MISDETECTION_COMPANSATION = 0.30
MAX_SENTENCE_DURATION = 15.0
SENTENCE_END_MARKS = ['.', '?', '!', ':', '...']
PUNCTUATION_MARKS = [',', ';', '/', '"']
WAV_BITRATE = 16000

AVCONV_LOC = "avconv"

temptextfile="tmp_subtext.txt"
tempwavfile="tmp_subaudio.wav"
tempscribefile="tmp_scribe.xml"

def readSrt(subfile):
	'''
	read srt file line by line, each entry (seperated by empty line) has 4 lines:
	1 - entry id --> stored as id in dict
	2 - time values --> processed and stored as start and dur in dict
	3 - subtitle line 1 --> stored as line1 in dict
	4 - subtitle line 2 --> stored as line2 in dict
	'''
	subData = []
	
	with open(subfile) as f:
		lines = f.read().decode("utf-8-sig").encode("utf-8").splitlines()
		linetype=1		#1-id, 2-timestamps, 3..-text
				
		subEntry = {'id':0, 'start':"", 'end':"", 'duration':"", 'textline1':"", 'textline2':"", 'subtext':""}
		for line in lines:
			if not line:
				subEntry['subtext'] = subEntry['subtext'][1:]   #take out the extra whitespace at the beginning
				subData.append(subEntry)
				subEntry = {'id':0, 'start':"", 'end':"", 'duration':"", 'textline1':"", 'textline2':"", 'subtext':""}
				linetype=1
				continue
			elif linetype==1:
				subEntry['id']=line		
				linetype=linetype+1;
			elif linetype==2: 
				timestamp=line
				 
				[startstamp, endstamp] = timestamp.split(" --> ")
				[startstamp, endstamp] = [startstamp.replace(",", ":"), endstamp.replace(",", ":")]
				[s_hour,s_min,s_sec,s_mil]=startstamp.split(':')
				[e_hour,e_min,e_sec,e_mil]=endstamp.split(':')

				t_end = time(int(e_hour),int(e_min),int(e_sec),int(e_mil)*1000)
				dt_end = datetime.combine(date.today(), t_end)

				delta_start = timedelta(hours=int(s_hour), minutes=int(s_min), seconds=int(s_sec), microseconds=int(s_mil)*1000)

				t_dur = (dt_end - delta_start).time()
				[dur_sec, dur_mil] = [t_dur.second, t_dur.microsecond/1000]
								
				subEntry['start'] = "%s:%s:%s.%s"%(s_hour, s_min, s_sec, s_mil)
				subEntry['end'] = "%s:%s:%s.%s"%(e_hour, e_min, e_sec, e_mil)
				subEntry['duration'] = "%i.%i"%(dur_sec, dur_mil)

				subEntry['startSecs'] = round(timestamp2secs(startstamp),TIME_ROUNDUP_FOR_AVCONV)
				subEntry['endSecs'] = round(timestamp2secs(endstamp),TIME_ROUNDUP_FOR_AVCONV)
				subEntry['durSecs'] = round(subEntry['endSecs'] - subEntry['startSecs'],TIME_ROUNDUP_FOR_AVCONV)
				linetype=3
			else:
				subEntry['subtext'] += ' ' + line
	return subData

def timestamp2secs(timestamp):
	#converts timestamp in hh:mm:ss:mmm form to s.mmm
	[s_hour,s_min,s_sec,s_mil]=timestamp.split(':')
	seconds=int(s_hour)*60*60 + int(s_min)*60 + int(s_sec)
	milliseconds=float(s_mil)*0.001
	return seconds+milliseconds

def cleanSrtData(srtData):
	#preprocesses raw subtitle data
	for i in reversed(range(len(srtData))):
		entry = srtData[i]
		#1 - remove entries with non-speech information such as [LAUGHTER] [MOAN] (HORN HONKING)
		if re.search(r"(\(|\[)(.|\s)+(\]|\))", entry['subtext']):
			del srtData[i]

		#2 - take out informative marks
		if re.search(r"<.+>", entry['subtext']):
			entry['subtext'] = re.sub(r"<[a-z]>|</[a-z]>", "", entry['subtext'])

		#3 - clear speech dashes (happens when two speakers speak in the same sub entry)
		if re.match(r"^-.*", entry['subtext']):
			entry['subtext'] = re.sub(r"-\s", "", entry['subtext'])

		#4 - clear names from beginning (i.e. DON: blablabla)
		if re.search(r"[^|^-].+:", entry['subtext']):
			entry['subtext'] = re.sub(r"^.*:\s", "", entry['subtext'])

		#5 - take out the dots in Mr. Mrs. Dr. Ms. 
		entry['subtext'] = re.sub(r"Dr(\.\s|\s)", "Doctor ", entry['subtext'])
		if options.movielang == "eng":
			entry['subtext'] = re.sub(r"Mr(\.\s|\s)", "Mister ", entry['subtext'])
			entry['subtext'] = re.sub(r"Mrs(\.\s|\s)", "Mrs ", entry['subtext'])
			entry['subtext'] = re.sub(r"Ms(\.\s|\s)", "Miss ", entry['subtext'])
		if options.movielang == "spa":
			entry['subtext'] = re.sub(r"Sr(\.\s|\s)", "Señor ", entry['subtext'])
			entry['subtext'] = re.sub(r"Sra(\.\s|\s)", "Señora ", entry['subtext'])
			entry['subtext'] = re.sub(r"Ud(\.\s|\s)", "usted ", entry['subtext'])
			entry['subtext'] = re.sub(r"Uds(\.\s|\s)", "ustedes ", entry['subtext'])
	return srtData

def checkFile(filename, variable):
    if not filename:
        print "%s file not given"%variable
        sys.exit()
    else:
        if not os.path.isfile(filename):
            print "%s file %s does not exist"%(variable, filename)
            sys.exit()

def checkFolder(dir):
	if not os.path.exists(dir):
		print "Creating folder ./%s"%(dir)
		os.makedirs(dir)	

def segmentSubEntries(srtData):
	for srtEntry in srtData:
		fileId="%s%04d"%(options.movielang, int(srtEntry['id']))

		segmentAudioFile = "%s/%s.wav"%(options.outdir, fileId)
		subScriptFile = "%s/%s_sub.txt"%(options.outdir, fileId)
		dubScriptFile = "%s/%s_dub.txt"%(options.outdir, fileId)

		cutAudioWithAvconv(options.audio, srtEntry['startSecs'], srtEntry['durSecs'], segmentAudioFile)
		srtEntry['fileName'] = segmentAudioFile

		#write subtitle text to a separate file
		with open(subScriptFile, 'w') as f:
			f.write(srtEntry['subtext'])

		#transcribe the dubbed audio (different than subtitle)
		if options.transcribe_dub:
			dubtext = witAiRecognize(segmentAudioFile, credentials.WIT_AI_KEY)
			#dubtext = googleSpeechRecognize(segmentAudioFile, credentials.GOOGLE_API_KEY)
			srtEntry['dubtext'] = dubtext
			if dubtext:
				with open(dubScriptFile, 'w') as f:
					f.write(dubtext)


def cutAudioWithAvconv(audioFilename, start_time, cut_duration, outputAudioFilename):
	#if output file already exists, delete it
	if os.path.isfile(outputAudioFilename):
		os.remove(outputAudioFilename)
	command = "%s -i %s -ss %s -t %s -ac 1 -ar %s %s"%(AVCONV_LOC, audioFilename, start_time, cut_duration, WAV_BITRATE, outputAudioFilename)
	process = subprocess.Popen(command.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
	output, error = process.communicate()

def checkIfAvconvInstalled():
	command = "%s"%(AVCONV_LOC)
	try:
		process = subprocess.Popen(command.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
		output, error = process.communicate()
	except:
		print("avconv software not found. Either install it or set the AVCONV_LOC variable to its correct path.")
		sys.exit()

def srtData2Txt(srtData, filename):
	with open(filename, 'w') as datafile:
		for data in srtData:
			datafile.write("%s\t%s\t%s\t%s\t%s\t%s\n"%(data['id'], data['startSecs'], data['duration'], data['fileName'], data['subtext'], data['dubtext']))
		datafile.close()

def witAiRecognize(segment, WIT_AI_KEY):
    r = sr.Recognizer()
    with sr.AudioFile(segment) as source:
        audio = r.record(source) # read the entire audio file

    result = None
    try:
        result = r.recognize_wit(audio, key=WIT_AI_KEY)
        #print("*** " + r.recognize_wit(audio, key=WIT_AI_KEY))
    except sr.UnknownValueError:
        print("Wit.ai could not understand audio: %s"%segment)
    except sr.RequestError as e:
        print("Could not request results from Wit.ai service; {0}".format(e))

    return result

def googleSpeechRecognize(segment, GOOGLE_API_KEY):
	r = sr.Recognizer()
	with sr.AudioFile(segment) as source:
		audio = r.record(source) # read the entire audio file

	result = None
	try:
		# for testing purposes, we're just using the default API key
		# to use another API key, use `r.recognize_google(audio, key="GOOGLE_SPEECH_RECOGNITION_API_KEY")`
		# instead of `r.recognize_google(audio)`
		print("Google Speech Recognition thinks you said " + r.recognize_google(audio, language='en'))
	except sr.UnknownValueError:
		print("Google Speech Recognition could not understand audio")
	except sr.RequestError as e:
		print("Could not request results from Google Speech Recognition service; {0}".format(e))

	return result


def main(options):
	checkIfAvconvInstalled()
	checkFile(options.audio,"movie audio")
	checkFile(options.subfile,"subtitle file")
	checkFolder(options.outdir)

	print "Audio: %s\nSubtitles: %s\nLanguage: %s\nTranscription: %s"%(options.audio, options.subfile, options.movielang, options.transcribe_dub)
	print "Reading subtitles...",
	srtData = readSrt(options.subfile)
	srtData = cleanSrtData(srtData)
	print "done"

	print "Segmenting subtitle entries...",
	segmentSubEntries(srtData)
	print "done."
	#sentenceData2Txt(sentenceData, "%s/%s_sentenceData.csv"%(options.outdir, options.movielang))
	srtDataFile = "%s/%s_srtData.csv"%(options.outdir, options.movielang)
	srtData2Txt(srtData, srtDataFile)
	print "Srt data written to %s"%srtDataFile

if __name__ == "__main__":
    usage = "usage: %prog [-s infile] [option]"
    parser = OptionParser(usage=usage)
    parser.add_option("-a", "--audio", dest="audio", default=None, help="movie audio file to be segmented", type="string")
    parser.add_option("-s", "--sub", dest="subfile", default=None, help="subtitle file (srt)", type="string")
    parser.add_option("-o", "--output-dir", dest="outdir", default=None, help="Directory to output segments and sentences", type="string")
    parser.add_option("-l", "--lang", dest="movielang", default=None, help="Language of the movie audio (Three letter ISO 639-2/T code)", type="string")
    parser.add_option("-t","--transcribe", dest="transcribe_dub", action="store_true", default=False, help="send dubbed audio segments to wit.ai")

    (options, args) = parser.parse_args()

    main(options)