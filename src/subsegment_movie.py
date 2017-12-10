# -*- coding: utf-8 -*-
from optparse import OptionParser
import os
import sys
import re
import csv
import speech_recognition as sr
from datetime import datetime, date, time, timedelta
import pysrt
from pydub import AudioSegment
from proscript.proscript import Word, Proscript, Segment
from proscript.utilities import utils

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

#TODO: merge to normalize_transcript
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

#TODO: merge to other functions
def segmentSubEntries(srtData):
	for srtEntry in srtData:
		fileId="%s%04d"%(options.movielang, srtEntry.index)

		segmentAudioFile = "%s/%s.wav"%(options.outdir, fileId)
		subScriptFile = "%s/%s_sub.txt"%(options.outdir, fileId)
		dubScriptFile = "%s/%s_dub.txt"%(options.outdir, fileId)

		cutAudioWithAvconv(options.audiofile, subriptime_to_seconds(srtEntry.start), subriptime_to_seconds(srtEntry.duration), segmentAudioFile)
		wav_file_name = "%s.wav"%(fileId)

		#write subtitle text to a separate file
		with open(subScriptFile, 'w') as f:
			f.write(srtEntry.text_without_tags)

		#transcribe the dubbed audio (different than subtitle)
		# if options.transcribe_dub:
		# 	dubtext = witAiRecognize(segmentAudioFile, credentials.WIT_AI_KEY)
		# 	#dubtext = googleSpeechRecognize(segmentAudioFile, credentials.GOOGLE_API_KEY)
		# 	srtEntry['dubtext'] = dubtext
		# 	if dubtext:
		# 		with open(dubScriptFile, 'w') as f:
		# 			f.write(dubtext)

def checkArgument(argname, isFile=False, isDir=False, createDir=False):
	if not argname:
		return False
	else:
		if isFile and not os.path.isfile(argname):
			return False
		if isDir:
			if not os.path.isdir(argname):
				if createDir:
					print("Creating directory %s"%(argname))
					os.makedirs(argname)
				else:
					return False
	return True

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


def cutAudioWithPydub(audio_segment, start_time, end_time, outputfile):
	extract = audio_segment[start_time*1000:end_time*1000]
	extract.export(outputfile, format=options.audioformat)

def extract_audio_segments(proscript, audio_segment, output_dir, file_prefix=""):
	'''
	Cuts each segment and outputs as wav+transcript
	'''
	for segment in proscript.segment_list:
		fileId="%s%04d"%(file_prefix, segment.id)

		segmentAudioFile = "%s/%s_audio.wav"%(output_dir, fileId)
		subScriptFile = "%s/%s_sub.txt"%(output_dir, fileId)
		#dubScriptFile = "%s/%s_dub.txt"%(output_dir, fileId)

		#cutAudioWithAvconv(audiofile, segment.start_time, segment.get_duration(), segmentAudioFile)
		cutAudioWithPydub(audio_segment, segment.start_time, segment.end_time, segmentAudioFile)

		#write subtitle text to a separate file
		with open(subScriptFile, 'w') as f:
			f.write(segment.transcript)

def subriptime_to_seconds(srTime):
	'''
	Convert SubRipTime object to seconds
	'''
	t = datetime.combine(date.min, srTime.to_time()) - datetime.min
	return t.total_seconds()

def normalize_transcript(transcript):
	'''
	All text normalization here
	'''
	transcript = re.sub('\n', ' ', transcript)
	return transcript

def to_proscript(srt_data):
	proscript = Proscript()

	segment_count = 0
	first_utterance = True

	for index, srt_entry in enumerate(srt_data):
		start_time = subriptime_to_seconds(srt_entry.start)
		end_time = subriptime_to_seconds(srt_entry.end)

		transcript = srt_entry.text_without_tags.strip()

		if not transcript.isspace():
			if first_utterance:
				curr_seg = Segment()
				curr_seg.start_time = start_time
				curr_seg.end_time = end_time
				curr_seg.transcript += transcript
				first_utterance = False
			elif curr_seg.transcript[-1] in SENTENCE_END_MARKS:
				if curr_seg.transcript and not curr_seg.transcript.isspace():
					segment_count += 1
					curr_seg.id = segment_count
					curr_seg.transcript = normalize_transcript(curr_seg.transcript)
					proscript.add_segment(curr_seg)
					#curr_seg.to_string()
					#print("----====----")
				curr_seg = Segment()
				curr_seg.start_time = start_time
				curr_seg.end_time = end_time
				curr_seg.transcript += transcript
				#print("curr_seg:\n%s"%curr_seg.transcript)
			else:
				curr_seg.end_time = subriptime_to_seconds(srt_entry.end)
				curr_seg.transcript += ' ' + transcript
				#print("curr_seg:\n%s"%curr_seg.transcript)

		if index == len(srt_data) - 1:
			if curr_seg.transcript and not curr_seg.transcript.isspace():
				segment_count += 1
				curr_seg.id = segment_count
				curr_seg.transcript = normalize_transcript(transcript)
				proscript.add_segment(curr_seg)
				#curr_seg.to_string()
				#print("----====----")
	return proscript

def main(options):
	checkArgument(options.audiofile, isFile=True)
	checkArgument(options.subfile, isFile=True)
	checkArgument(options.outdir, isDir=True, createDir=True)

	print("Audio: %s\nSubtitles: %s\nLanguage: %s\nTranscription: %s"%(options.audiofile, options.subfile, options.movielang, options.transcribe_dub))
	print("Reading subtitles...", end="")
	#srtData = readSrt(options.subfile)
	#srtData = cleanSrtData(srtData)
	srtData = pysrt.open(options.subfile)
	print("done")

	audio = AudioSegment.from_file(options.audiofile, format=options.audioformat)

	movie_proscript = to_proscript(srtData)

	proscript_file = "%s/%s_proscript.csv"%(options.outdir, options.movielang)
	movie_proscript.segments_to_csv(proscript_file, ['id', 'start_time', 'end_time', 'transcript'], delimiter='|')

	extract_audio_segments(movie_proscript, audio, options.outdir, file_prefix=options.movielang)

	#print("Segmenting subtitle entries...", end="")
	#segmentSubEntries(srtData)
	#print("done.")
	#sentenceData2Txt(sentenceData, "%s/%s_sentenceData.csv"%(options.outdir, options.movielang))
	#srtDataFile = "%s/%s_srtData.csv"%(options.outdir, options.movielang)
	#srtData2Txt(srtData, srtDataFile)
	#print("Srt data written to %s"%srtDataFile)

if __name__ == "__main__":
    usage = "usage: %prog [-s infile] [option]"
    parser = OptionParser(usage=usage)
    parser.add_option("-a", "--audiofile", dest="audiofile", default=None, help="movie audio file to be segmented", type="string")
    parser.add_option("-s", "--sub", dest="subfile", default=None, help="subtitle file (srt)", type="string")
    parser.add_option("-o", "--output-dir", dest="outdir", default=None, help="Directory to output segments and sentences", type="string")
    parser.add_option("-l", "--lang", dest="movielang", default="", help="Language of the movie audio (Three letter ISO 639-2/T code)", type="string")
    parser.add_option("-t","--transcribe", dest="transcribe_dub", action="store_true", default=False, help="send dubbed audio segments to wit.ai")
    parser.add_option("-f", "--audioformat", dest="audioformat", default="wav", help="Audio format (wav, mp3 etc.)", type="string")

    (options, args) = parser.parse_args()

    main(options)