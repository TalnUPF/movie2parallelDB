import os
import sys
import re
import csv
import copy
import nltk
import string
from optparse import OptionParser
from datetime import datetime, date, time, timedelta
from proscript.proscript import Word, Proscript, Segment
from proscript.utilities import utils
from subsegment_movie import *
from yandex_translate import YandexTranslate
import numpy as np
import gensim, logging

SURE_MATCH_CORRELATION_THRESHOLD = 70.0
OK_MATCH_CORRELATION_THRESHOLD = 30.0
PARTIAL_MATCH_CORRELATION_THRESHOLD = 8.0

UNK_TOKEN = 'UNKNOWN'
YANDEX_TRANSLATE_KEY = "trnsl.1.1.20161114T160229Z.04e03dd799ba5ad4.e6fec58810fcfd545da54bb125e149e71e76c8ee"
W2V_MODEL_EN_FILE = '/Users/alp/Documents/Corpora/OpenSubtitles2018/word2vec_en_mtready.model'

'''
Returns indexes of matching segments in proscripts. Proscript with less segments should be given as first argument (proscript_spa)
'''
def map_segments(proscript_spa, proscript_eng):
	matched = []	#spa-eng
	spa_index = 0
	eng_index = 0
	eng_end = False
	while spa_index < len(proscript_spa.segment_list):
		correlation = 0
		while eng_index <= len(proscript_eng.segment_list):
			if eng_index == len(proscript_eng.segment_list):
				eng_end = True
				break

			correlation = get_segments_correlation(proscript_spa.segment_list[spa_index], proscript_eng.segment_list[eng_index])
			if correlation >= SURE_MATCH_CORRELATION_THRESHOLD:
				#print("Match (%f) between spa:%i and eng:%i"%(correlation, proscript_spa.segment_list[spa_index].id, proscript_eng.segment_list[eng_index].id))
				matched.append(([spa_index], [eng_index]))
				eng_index += 1
				spa_index += 1
				break
			elif correlation >= PARTIAL_MATCH_CORRELATION_THRESHOLD:
				#print("Partial Correlation of %i between spa:%i and eng:%i"%(correlation, proscript_spa.segment_list[spa_index].id, proscript_eng.segment_list[eng_index].id))
				#Try merging with the next one and see if it gets better
				match_candidates = [([spa_index], [eng_index])]
				if spa_index + 1 < len(proscript_spa.segment_list) and proscript_spa.segment_list[spa_index].speaker_id == proscript_spa.segment_list[spa_index + 1].speaker_id:
					match_candidates.append(([spa_index, spa_index + 1], [eng_index]))

				if eng_index + 1 < len(proscript_eng.segment_list) and proscript_eng.segment_list[eng_index].speaker_id == proscript_eng.segment_list[eng_index + 1].speaker_id:
					match_candidates.append(([spa_index], [eng_index, eng_index + 1]))

				if spa_index + 1 < len(proscript_spa.segment_list) and eng_index + 1 < len(proscript_eng.segment_list) and proscript_spa.segment_list[spa_index].speaker_id == proscript_spa.segment_list[spa_index + 1].speaker_id and proscript_eng.segment_list[eng_index].speaker_id == proscript_eng.segment_list[eng_index + 1].speaker_id:
					match_candidates.append(([spa_index, spa_index + 1], [eng_index, eng_index + 1]))

				candidate_scores = []
				#print("candidates")
				for index, candidate in enumerate(match_candidates):
					merged_correlation = get_segments_correlation([proscript_spa.segment_list[spa_index] for spa_index in candidate[0]], 
																   [proscript_eng.segment_list[eng_index] for eng_index in candidate[1]])
					candidate_scores.append(merged_correlation)
					#print("%s - %s:%i"%([proscript_spa.segment_list[spa_index].id for spa_index in candidate[0]], [proscript_eng.segment_list[eng_index].id for eng_index in candidate[1]], merged_correlation))

				best_match_index = np.argmax(candidate_scores)
				best_match = match_candidates[best_match_index]
				#print("Best match %i of %i"%(best_match_index + 1, len(match_candidates)))
				if best_match_index > 0 and candidate_scores[best_match_index] > candidate_scores[0] and candidate_scores[best_match_index] >= SURE_MATCH_CORRELATION_THRESHOLD:
					#print("Merged match (%f) between spa:%s and eng:%s"%(candidate_scores[best_match_index], [proscript_spa.segment_list[spa_index].id for spa_index in best_match[0]], [proscript_eng.segment_list[eng_index].id for eng_index in best_match[1]]))
					matched.append(best_match)
					spa_index += len(best_match[0])
					eng_index += len(best_match[1])
				elif candidate_scores[0] >= OK_MATCH_CORRELATION_THRESHOLD:
					#print("OK Merged match (%f) between spa:%s and eng:%s"%(candidate_scores[best_match_index], [proscript_spa.segment_list[spa_index].id for spa_index in best_match[0]], [proscript_eng.segment_list[eng_index].id for eng_index in best_match[1]]))
					matched.append(match_candidates[0])
					spa_index += 1
					eng_index += 1
				else:
					#merge fail
					#print("No good merge for '%s' and '%s'"%(proscript_spa.segment_list[spa_index].transcript, proscript_eng.segment_list[eng_index].transcript))
					if proscript_spa.segment_list[spa_index].start_time < proscript_eng.segment_list[eng_index].start_time:
						#print("Missed SPA segment %i: %s"%(proscript_spa.segment_list[spa_index].id, proscript_spa.segment_list[spa_index].transcript))
						spa_index += 1
					else:
						eng_index += 1
			else:
				#make indexes catch up
				if proscript_spa.segment_list[spa_index].start_time < proscript_eng.segment_list[eng_index].start_time:
					#print("Catch up Missed SPA segment %i: %s"%(proscript_spa.segment_list[spa_index].id, proscript_spa.segment_list[spa_index].transcript))
					spa_index += 1
				else:
					eng_index += 1
		if eng_end:
			break
	return matched

'''
Returns indexes of matching segments in proscripts. Proscript with less segments should be given as first argument (proscript_spa)
'''
def map_segments_2(proscript_spa, proscript_eng, translator, w2v_model, stopwords = []):
	matched = []	#spa-eng
	spa_index = 0
	eng_index = 0
	eng_end = False
	while spa_index < len(proscript_spa.segment_list):
		correlation = 0
		while eng_index <= len(proscript_eng.segment_list):
			if eng_index == len(proscript_eng.segment_list):
				eng_end = True
				break

			correlation = get_segments_correlation(proscript_spa.segment_list[spa_index], proscript_eng.segment_list[eng_index])
			if correlation > MATCH_CORRELATION_THRESHOLD:
				print("Match (%f) between spa:%i and eng:%i"%(correlation, proscript_spa.segment_list[spa_index].id, proscript_eng.segment_list[eng_index].id))
				matched.append(([spa_index], [eng_index]))
				eng_index += 1
				spa_index += 1
				break
			elif correlation > PARTIAL_MATCH_CORRELATION_THRESHOLD:
				print("Partial Correlation of %i between spa:%i and eng:%i"%(correlation, proscript_spa.segment_list[spa_index].id, proscript_eng.segment_list[eng_index].id))

				match_candidates = [([spa_index], [eng_index])]
				if proscript_eng.segment_list[eng_index].get_duration() > proscript_spa.segment_list[spa_index].get_duration() and spa_index + 1 < len(proscript_spa.segment_list) and proscript_spa.segment_list[spa_index].speaker_id == proscript_spa.segment_list[spa_index + 1].speaker_id:
					match_candidates.append(([spa_index, spa_index + 1], [eng_index]))

				elif proscript_spa.segment_list[spa_index].get_duration() >= proscript_eng.segment_list[eng_index].get_duration() and eng_index + 1 < len(proscript_eng.segment_list) and proscript_eng.segment_list[eng_index].speaker_id == proscript_eng.segment_list[eng_index + 1].speaker_id:
					match_candidates.append(([spa_index], [eng_index, eng_index + 1]))

				print("candidates")
				candidate_scores = []
				for index, candidate in enumerate(match_candidates):
					transcript_spa = ' '.join([segment.transcript for segment in proscript_spa.segment_list[array_to_slice(candidate[0])]])
					transcript_eng = ' '.join([segment.transcript for segment in proscript_eng.segment_list[array_to_slice(candidate[1])]])
					transcript_spa_translated = translator.translate(transcript_spa, 'es-en')['text'].pop()
					similarity_score = get_sentence_similarity(transcript_spa_translated, transcript_eng, w2v_model, stopwords)
					candidate_scores.append(similarity_score)
					print("%i: %s (%s) - %s"%(index, transcript_spa, transcript_spa_translated, transcript_eng))
				
				#get best (minimum) of the scores and add that pair to matches
				best_match_index = np.argmin(candidate_scores)
				best_match = match_candidates[best_match_index]
				matched.append(best_match)
				spa_index += len(best_match[0])
				eng_index += len(best_match[1])
				print("%i Won."%(best_match_index))
				print("Match between spa:%s and eng:%s"%([proscript_spa.segment_list[spa_index].id for spa_index in best_match[0]], [proscript_eng.segment_list[eng_index].id for eng_index in best_match[1]]))
			else:
				#make indexes catch up
				if proscript_spa.segment_list[spa_index].start_time < proscript_eng.segment_list[eng_index].start_time:
					print("missed %i: "%(proscript_spa.segment_list[spa_index].id, proscript_spa.segment_list[spa_index].transcript))
					spa_index += 1

				else:
					eng_index += 1
		if eng_end:
			break
	return matched


'''
Outputs mapping to a text file
'''
def mapping_to_file(mapping, file_path, proscript_spa, proscript_eng):
	with open(file_path, "w") as f:
		aligned_segment_index = 1
		for matching_segment_indexes in mapping:
			spa_indexes = [proscript_spa.segment_list[segment_index].id for segment_index in matching_segment_indexes[0]]
			eng_indexes = [proscript_eng.segment_list[segment_index].id for segment_index in matching_segment_indexes[1]]
			spa_transcript = ' '.join([proscript_spa.segment_list[segment_index].transcript for segment_index in matching_segment_indexes[0]])
			eng_transcript = ' '.join([proscript_eng.segment_list[segment_index].transcript for segment_index in matching_segment_indexes[1]])
			f.write("%i:%s-%s:%s|%s\n"%(aligned_segment_index, spa_indexes, eng_indexes, spa_transcript, eng_transcript))
			aligned_segment_index += 1

'''
Returns the percentage of time correlation between two segment sets
'''
def get_segments_correlation(segments1, segments2):
	if type(segments1) == list:
		segments1_list = segments1
	else:
		segments1_list = [segments1]
	if type(segments2) == list:
		segments2_list = segments2
	else:
		segments2_list = [segments2]
	#print("correlating: %f - %f"%(max(segments1_list[0].start_time, segments2_list[0].start_time), min(segments1_list[-1].end_time, segments2_list[-1].end_time)))
	#print("span: %f - %f"%(min(segments1_list[0].start_time, segments2_list[0].start_time), max(segments1_list[-1].end_time, segments2_list[-1].end_time) ))
	correlating = min(segments1_list[-1].end_time, segments2_list[-1].end_time) - max(segments1_list[0].start_time, segments2_list[0].start_time)
	span = max(segments1_list[-1].end_time, segments2_list[-1].end_time) - min(segments1_list[0].start_time, segments2_list[0].start_time)
	return round(max(0, correlating/span * 100))

def vectorize_sentence(sentence, w2v_model, stopwords = []):
	sentence_tokenized = nltk.word_tokenize(sentence)
	sentence_tokenized = [token for token in sentence_tokenized if token not in stopwords]
	emb_vector_size = w2v_model.layer1_size
	vectorized_sentence = np.zeros(shape = (len(sentence_tokenized), emb_vector_size), dtype='float32')

	for token_index in range(len(sentence_tokenized)):
		token = sentence_tokenized[token_index]
		try:
			word_vector = w2v_model.wv[token]
		except KeyError as e:
			word_vector = w2v_model.wv[UNK_TOKEN]
		vectorized_sentence[token_index] = word_vector

	return np.average(vectorized_sentence, axis=0)

def get_sentence_similarity(sentence_1, sentence_2, w2v_model, stopwords = []):
	vector_1 = vectorize_sentence(sentence_1, w2v_model, stopwords)
	vector_2 = vectorize_sentence(sentence_2, w2v_model, stopwords)

	return np.linalg.norm(vector_1 - vector_2)

def array_to_slice(index_array):
	if len(index_array) == 1:
		return slice(index_array[0], index_array[0] + 1)
	else:
		return slice(index_array[0], index_array[1] + 1)


'''
Creates one segment from a list of adjacent segments given in order in a list
'''
def merge_segments_to_new_segment(segment_list, new_segment_id, new_speaker_id = None, proscript_ref = None):
	assert len(segment_list) > 0, "Given segment list is empty"
	#assert speaker id's are the same
	new_segment = Segment()
	new_segment.id = new_segment_id
	new_segment.start_time = segment_list[0].start_time
	new_segment.end_time = segment_list[-1].end_time
	if new_speaker_id:
		new_segment.speaker_id = new_speaker_id
	else:
		new_segment.speaker_id = segment_list[0].speaker_id
	new_segment.transcript = ' '.join([segment.transcript for segment in segment_list])
	if proscript_ref:
		new_segment.proscript_ref = proscript_ref
	#fill in words
	for segment in segment_list:
		for word in segment.word_list:
			new_segment.add_word(word)
	return new_segment

def get_aligned_proscripts(mapping_list, proscript_spa, proscript_eng, copy_speaker_info_from='0'):
	aligned_proscript_spa = Proscript()
	aligned_proscript_spa.id = proscript_spa.id + "_aligned"
	aligned_proscript_spa.audio_file = proscript_spa.audio_file
	aligned_proscript_spa.duration = proscript_spa.duration

	aligned_proscript_eng = Proscript()
	aligned_proscript_eng.id = proscript_eng.id + "_aligned"
	aligned_proscript_eng.audio_file = proscript_eng.audio_file
	aligned_proscript_eng.duration = proscript_eng.duration

	new_spa_segment_id = 1
	new_eng_segment_id = 1
	for mapping in mapping_list:
		new_segment_eng = merge_segments_to_new_segment( [proscript_eng.segment_list[segment_index] for segment_index in mapping[1]], 
														 new_eng_segment_id, 
														 proscript_ref = aligned_proscript_eng)
		new_segment_spa = merge_segments_to_new_segment( [proscript_spa.segment_list[segment_index] for segment_index in mapping[0]], 
														 new_spa_segment_id, 
														 new_speaker_id = new_segment_eng.speaker_id, 
														 proscript_ref = aligned_proscript_spa)
		aligned_proscript_spa.add_segment(new_segment_spa)
		aligned_proscript_eng.add_segment(new_segment_eng)

		new_spa_segment_id += 1
		new_eng_segment_id += 1

	utils.assign_word_ids(aligned_proscript_spa)
	utils.assign_word_ids(aligned_proscript_eng)

	aligned_proscript_spa.populate_speaker_ids()
	aligned_proscript_eng.populate_speaker_ids()

	return aligned_proscript_spa, aligned_proscript_eng

def main(options):
	translator = YandexTranslate(YANDEX_TRANSLATE_KEY)
	stopwords = nltk.corpus.stopwords.words('english')
	stopwords.extend(string.punctuation)
	stopwords.append('')

	w2v_model_en = gensim.models.Word2Vec.load(W2V_MODEL_EN_FILE)

	process_list_eng = fill_task_list_from_file(options.list_of_files_eng, options.output_dir)
	process_list_spa = fill_task_list_from_file(options.list_of_files_spa, options.output_dir)

	#print(process_list_spa)
	#print(process_list_eng)

	assert len(process_list_eng) == len(process_list_spa), "Process lists are not the same length"

	
	for task_index, (proscript_eng, proscript_spa) in enumerate(zip(process_tasks(process_list_eng, options.input_audio_format, skip_mfa=options.skip_mfa), process_tasks(process_list_spa, options.input_audio_format, skip_mfa=options.skip_mfa))):
		proscript_mapping_2 = map_segments(proscript_spa, proscript_eng)
		#proscript_mapping_2 = map_segments_2(proscript_spa, proscript_eng, translator, w2v_model_en, stopwords)

		aligned_proscript_spa, aligned_proscript_eng = get_aligned_proscripts(proscript_mapping_2, proscript_spa, proscript_eng)
		utils.assign_acoustic_means(aligned_proscript_spa)
		utils.assign_acoustic_means(aligned_proscript_eng)

		#Determine paths for parallel data
		task_output_path = process_list_eng[task_index]['output_dir']
		parallel_output_path = os.path.join(task_output_path, '..', 'spa-eng')
		checkArgument(parallel_output_path, createDir = True)

		#Fill in acoustic features
		# if options.annotate_prosody:
		# 	acoustic_files_path_eng = os.path.join(parallel_output_path, 'acoustics_eng')
		# 	acoustic_files_path_spa = os.path.join(parallel_output_path, 'acoustics_spa')
		# 	utils.assign_acoustic_feats(aligned_proscript_eng, acoustic_files_path_eng)
		# 	utils.assign_acoustic_feats(aligned_proscript_spa, acoustic_files_path_spa)


		#write mapping to file
		mapping_file_path = os.path.join(parallel_output_path, '%s_mapping.txt'%proscript_eng.id)
		mapping_to_file(proscript_mapping_2, mapping_file_path, proscript_spa, proscript_eng)
		print("Mapping extracted to %s"%mapping_file_path)

		print("spanish audio: %s"%aligned_proscript_spa.audio_file)
		print("english audio: %s"%aligned_proscript_eng.audio_file)

		#store aligned proscript data to disk
		extract_proscript_data_to_disk(aligned_proscript_spa, parallel_output_path, 'spa', cut_audio_portions = True, extract_segments_as_proscript = True, output_audio_format = 'wav', segments_subdir='segments_spa')
		extract_proscript_data_to_disk(aligned_proscript_eng, parallel_output_path, 'eng', cut_audio_portions = True, extract_segments_as_proscript = True, output_audio_format = 'wav', segments_subdir='segments_eng')
	
	#merge segments merged in the mapping. output

if __name__ == "__main__":
	usage = "usage: %prog [-s infile] [option]"
	parser = OptionParser(usage=usage)
	parser.add_option("-e", "--filelist_eng", dest="list_of_files_eng", default=None, help="list of files to process in english. Each line with id, audio, xml, lang (tab separated)", type="string")	
	parser.add_option("-s", "--filelist_spa", dest="list_of_files_spa", default=None, help="list of files to process in spanish. Each line with id, audio, xml, lang (tab separated)", type="string")	
	parser.add_option("-o", "--output-dir", dest="output_dir", default=None, help="Output directory", type="string")
	parser.add_option("-f", "--input-audio-format", dest="input_audio_format", default="mp3", help="Audio format (wav, mp3 etc.)", type="string")
	parser.add_option("-m", "--skip_mfa", dest="skip_mfa", default=False, action="store_true", help='Flag to take already made word aligned textgrid in output folder')
	parser.add_option("-p", "--annotate_prosody", dest="annotate_prosody", default=False, action="store_true", help='Flag to make prosodic annotations')

	(options, args) = parser.parse_args()
	main(options)