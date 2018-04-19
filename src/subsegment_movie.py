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
from shutil import copyfile
from proscript.proscript import Word, Proscript, Segment
from proscript.utilities import utils

#CONSTANTS
TIME_ROUNDUP_FOR_AVCONV = 3
TEMP_SUB_WAV_BUFFER = 0.5  
SCRIBE_AVG_SCORE_THRESHOLD = 0.5
SENTENCE_END_BUFFER = 0
END_WORD_MISDETECTION_COMPANSATION = 0.30
MAX_SENTENCE_DURATION = 30.0
MERGE_SEGMENT_MAX_SECONDS = 5.0
SENTENCE_END_MARKS = ['.', '?', '!', ':', '"']
PUNCTUATION_MARKS = [',', ';', '/', '"']
WAV_BITRATE = 16000
DEFAULT_SPEAKER_ID = "actor"

MFA_ALIGN_BINARY = "/Users/alp/extSW/montreal-forced-aligner/bin/mfa_align"
MFA_LEXICON_ENG = "/Users/alp/extSW/montreal-forced-aligner/pretrained_models/en.dict"
MFA_LM_ENG = "/Users/alp/extSW/montreal-forced-aligner/pretrained_models/english.zip"

MFA_LEXICON_SPA = "/Users/alp/extSW/montreal-forced-aligner/pretrained_models/spanish.dict"
MFA_LM_SPA = "/Users/alp/extSW/montreal-forced-aligner/pretrained_models/spanish.zip"

DELETE_TMP_WAV = False

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
		# if re.match(r"^-.*", entry['subtext']):
		# 	entry['subtext'] = re.sub(r"-\s", "", entry['subtext'])

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

def checkArgument(argname, isFile=False, isDir=False, createDir=False):
	if not argname:
		return False
	else:
		if isFile and not os.path.isfile(argname):
			return False
		if isDir or createDir:
			if not os.path.isdir(argname):
				if createDir:
					print("Creating directory %s"%(argname))
					os.makedirs(argname)
				else:
					return False
	return True

def sniff_file_encoding(filepath):
	from chardet.universaldetector import UniversalDetector
	detector = UniversalDetector()
	with open(filepath, 'rb') as f:
		for line in f.readlines():
			detector.feed(line)
			if detector.done: break
	detector.close()
	return detector.result['encoding']

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

def googleCloudRecognize(segment_file, recognizer, CREDENTIALS_JSON):
	with sr.AudioFile(segment_file) as source:
		audio = recognizer.record(source) # read the entire audio file

	result = None
	try:
		result = recognizer.recognize_google_cloud(audio, credentials_json=CREDENTIALS_JSON)
	except sr.UnknownValueError:
		print("Google Speech Recognition could not understand audio")
	except sr.RequestError as e:
		print("Could not request results from Google Speech Recognition service; {0}".format(e))

	return result

def cutAudioWithPydub(audio_segment, start_time, end_time, outputfile, output_audio_format='wav'):
	extract = audio_segment[int(start_time*1000):int(end_time*1000)]
	extract.export(outputfile, format=output_audio_format)

def extract_audio_segments(proscript, audiofile, output_dir, file_prefix="", output_audio_format='wav', segments_subdir='segments'):
	'''
	Cuts each segment and outputs as wav+transcript
	'''
	segments_output_dir = os.path.join(output_dir, segments_subdir)
	checkArgument(segments_output_dir, createDir = True)
	audio_segment = AudioSegment.from_file(audiofile, format='wav')
	for segment in proscript.segment_list:
		fileId="%s%04d"%(file_prefix, segment.id)
		segmentAudioFile = "%s/%s.wav"%(segments_output_dir, fileId)
		subScriptFile = "%s/%s.txt"%(segments_output_dir, fileId)

		cutAudioWithPydub(audio_segment, segment.start_time, segment.end_time, segmentAudioFile, output_audio_format)

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
	transcript = re.sub('’', "'", transcript)
	transcript = re.sub('"', ' ', transcript)
	transcript = re.sub('\n', ' ', transcript)
	transcript = re.sub('^-', '', transcript)
	transcript = re.sub(' -', ' - ', transcript)
	return transcript

def check_sentence_end(transcript):
	'''
	Returns true if the transcript ends a sentence.
	'''
	if transcript[-1] in SENTENCE_END_MARKS:
		if transcript[-1] == '.':
			#check if it's '...' then it's not sentence ending
			word_reversed = transcript[::-1]
			if re.search(r"^\W",word_reversed):
				punc = word_reversed[:re.search(r"\w", word_reversed).start()][::-1]
			if punc == '...':
				return False
		return True
	else:
		return False

def get_speaker_id():
	'''
	STUB
	'''
	return DEFAULT_SPEAKER_ID

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
				curr_seg.speaker_id = get_speaker_id()
				curr_seg.start_time = start_time
				curr_seg.end_time = end_time
				curr_seg.transcript += transcript
				first_utterance = False
			elif check_sentence_end(curr_seg.transcript) or start_time - curr_seg.end_time > MERGE_SEGMENT_MAX_SECONDS:
				if curr_seg.transcript and not curr_seg.transcript.isspace():
					segment_count += 1
					curr_seg.id = segment_count
					curr_seg.transcript = normalize_transcript(curr_seg.transcript)
					proscript.add_segment(curr_seg)
					#curr_seg.to_string()
					#print("----====----")
				curr_seg = Segment()
				curr_seg.speaker_id = get_speaker_id()
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

	process_list = []
	#Fill process list either from process list or from given audio, subtitle pair
	if options.list_of_files:
		with open(options.list_of_files) as f:
			for line in f:
				file_id = line.split('\t')[0]
				file_in_audio = line.split('\t')[1].strip()
				file_in_srt = line.split('\t')[2].strip()
				file_lang = line.split('\t')[3].strip()
				file_output_dir = os.path.join(options.outdir, file_id)
				if checkArgument(file_in_audio, isFile=True) and checkArgument(file_in_srt, isFile=True):
					output_dir = os.path.join(options.outdir, file_id, file_lang)
					checkArgument(output_dir, isDir=True, createDir=True)
					process_list.append({'file_id': file_id, 'file_in_audio':file_in_audio, 'file_in_srt':file_in_srt, 'output_dir':output_dir, 'lang':file_lang})	
	else:
		file_id = "movie"
		checkArgument(options.audiofile, isFile=True)
		checkArgument(options.subfile, isFile=True)
		checkArgument(options.outdir, isDir=True, createDir=True)
		process_list.append({'file_id': file_id, 'file_in_audio':options.audiofile, 'file_in_srt':options.subfile, 'output_dir':options.outdir, 'lang':options.movielang})	

	#Process files
	for process in process_list:
		movieid = process['file_id']
		audiofile = process['file_in_audio']
		subfile = process['file_in_srt']
		outdir = process['output_dir']
		movielang = process['lang']

		print("Audio: %s\nSubtitles: %s\nLanguage: %s\nTranscription: %s"%(audiofile, subfile, movielang, options.transcribe_dub))
		print("Reading subtitles...", end="")
		srt_encoding = sniff_file_encoding(subfile)
		print(" (encoding: %s) ..."%srt_encoding, end="")
		srtData = pysrt.open(subfile, encoding=srt_encoding)
		print("done")

		#Audio file needs to be stored as wav in the output folder temporarily
		audio = AudioSegment.from_file(audiofile, format=options.audioformat)
		tmp_audiopath = os.path.join(outdir, movieid + '_' + movielang + '.wav')
		cutAudioWithPydub(audio, 0, subriptime_to_seconds(srtData[-1].end), tmp_audiopath)

		movie_proscript = to_proscript(srtData)
		movie_proscript.id = movieid
		movie_proscript.audio = tmp_audiopath
		movie_proscript.duration = audio.duration_seconds
		movie_proscript.speaker_ids.append(DEFAULT_SPEAKER_ID)
		
		utils.proscript_segments_to_textgrid(movie_proscript, outdir, file_prefix="%s_%s"%(movieid, movielang), speaker_segmented=False)
		if movielang == 'eng':
			utils.mfa_word_align(outdir,  transcript_type="TextGrid", mfa_align_binary=MFA_ALIGN_BINARY, lexicon=MFA_LEXICON_ENG, language_model=MFA_LM_ENG)
		elif movielang == 'spa':
			utils.mfa_word_align(outdir,  transcript_type="TextGrid", mfa_align_binary=MFA_ALIGN_BINARY, lexicon=MFA_LEXICON_SPA, language_model=MFA_LM_SPA)
		utils.get_word_alignment_from_textgrid(movie_proscript)
		utils.split_multispeaker_segments(movie_proscript, default_speaker_id = DEFAULT_SPEAKER_ID)
		utils.assign_word_ids(movie_proscript)

		extract_audio_segments(movie_proscript, tmp_audiopath, outdir, file_prefix='%s_%s'%(movieid, movielang), output_audio_format='wav')

		#STORE DATA TO FILES
		segments_proscript_file = os.path.join(outdir, "%s_%s.segments-proscript.csv"%(movieid, movielang))
		movie_proscript.segments_to_csv(segments_proscript_file, ['id', 'start_time', 'end_time', 'transcript'], delimiter='|')

		words_proscript_file = os.path.join(outdir, "%s_%s.words-proscript.csv"%(movieid, movielang))
		movie_proscript.to_csv(words_proscript_file)

		if DELETE_TMP_WAV:
			os.remove(tmp_audiopath)

if __name__ == "__main__":
    usage = "usage: %prog [-s infile] [option]"
    parser = OptionParser(usage=usage)
    parser.add_option("-i", "--filelist", dest="list_of_files", default=None, help="list of files to process. Each line with id, wav, xml, lang (tab separated)", type="string")	#glissando files
    parser.add_option("-a", "--audiofile", dest="audiofile", default=None, help="movie audio file to be segmented", type="string")
    parser.add_option("-s", "--sub", dest="subfile", default=None, help="subtitle file (srt)", type="string")
    parser.add_option("-o", "--output-dir", dest="outdir", default=None, help="Directory to output segments and sentences", type="string")
    parser.add_option("-l", "--lang", dest="movielang", default="", help="Language of the movie audio (Three letter ISO 639-2/T code)", type="string")
    parser.add_option("-t", "--transcribe", dest="transcribe_dub", action="store_true", default=False, help="send dubbed audio segments to wit.ai")
    parser.add_option("-f", "--audioformat", dest="audioformat", default="mp3", help="Audio format (wav, mp3 etc.)", type="string")

    (options, args) = parser.parse_args()

    main(options)