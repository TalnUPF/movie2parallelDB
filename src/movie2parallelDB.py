import os
import sys
import re
import csv
import copy
from optparse import OptionParser
from datetime import datetime, date, time, timedelta
from proscript.proscript import Word, Proscript, Segment
from proscript.utilities import utils
from subsegment_movie import *

MATCH_CORRELATION_THRESHOLD = 50.0
PARTIAL_MATCH_CORRELATION_THRESHOLD = 10.0

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
			if correlation > MATCH_CORRELATION_THRESHOLD:
				print("Match between spa:%i and eng:%i"%(proscript_spa.segment_list[spa_index].id, proscript_eng.segment_list[eng_index].id))
				matched.append(([spa_index], [eng_index]))
				eng_index += 1
				spa_index += 1
				break
			elif correlation > PARTIAL_MATCH_CORRELATION_THRESHOLD:
				print("Partial Correlation of %i between spa:%i and eng:%i"%(correlation, proscript_spa.segment_list[spa_index].id, proscript_eng.segment_list[eng_index].id))
				merge_fail = True
				#Try merging with the next one and see if it gets better
				if proscript_eng.segment_list[eng_index].get_duration() > proscript_spa.segment_list[spa_index].get_duration():
					merged_correlation = get_segments_correlation([proscript_spa.segment_list[spa_index], proscript_spa.segment_list[spa_index + 1]], proscript_eng.segment_list[eng_index])
					if merged_correlation > MATCH_CORRELATION_THRESHOLD:
						print("Merged match between spa:%i-%i and eng:%i"%(proscript_spa.segment_list[spa_index].id, proscript_spa.segment_list[spa_index+1].id, proscript_eng.segment_list[eng_index].id))
						matched.append(([spa_index, spa_index+1], [eng_index]))
						eng_index += 1
						spa_index += 2
						merge_fail = False
				elif proscript_spa.segment_list[spa_index].get_duration() >= proscript_eng.segment_list[eng_index].get_duration():
					merged_correlation = get_segments_correlation(proscript_spa.segment_list[spa_index], [proscript_eng.segment_list[eng_index], proscript_eng.segment_list[eng_index + 1]])
					if merged_correlation > MATCH_CORRELATION_THRESHOLD:
						print("Merged match between spa:%i and eng:%i-%i"%(proscript_spa.segment_list[spa_index].id, proscript_eng.segment_list[eng_index].id, proscript_eng.segment_list[eng_index + 1].id))
						matched.append(([spa_index], [eng_index, eng_index + 1]))
						eng_index += 2
						spa_index += 1
						merge_fail = False
						
				if merge_fail:
					#move on
					if proscript_spa.segment_list[spa_index].start_time < proscript_eng.segment_list[eng_index].start_time:
						spa_index += 1
					else:
						eng_index += 1
			else:
				#make indexes catch up
				if proscript_spa.segment_list[spa_index].start_time < proscript_eng.segment_list[eng_index].start_time:
					spa_index += 1
				else:
					eng_index += 1
		if eng_end:
			#print("at end of eng")
			break
	return matched

'''
returns the percentage of time correlation between two segment sets
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

def mapping_to_file(mapping, file_path, proscript_spa, proscript_eng):
	with open(file_path, "w") as f:
		for matching_segment_ids in mapping:
			spa_indexes = [proscript_spa.segment_list[segment_id].id for segment_id in matching_segment_ids[0]]
			eng_indexes = [proscript_eng.segment_list[segment_id].id for segment_id in matching_segment_ids[1]]
			spa_transcript = ' '.join([proscript_spa.segment_list[segment_id].transcript for segment_id in matching_segment_ids[0]])
			eng_transcript = ' '.join([proscript_eng.segment_list[segment_id].transcript for segment_id in matching_segment_ids[1]])
			f.write("%s-%s:%s|%s\n"%(spa_indexes, eng_indexes, spa_transcript, eng_transcript))

def main(options):
	process_list_eng = get_process_list_from_file(options.list_of_files_eng, options.output_dir)
	process_list_spa = get_process_list_from_file(options.list_of_files_spa, options.output_dir)

	assert len(process_list_eng) == len(process_list_spa), "Process lists are not the same length"

	for proscript_eng, proscript_spa in zip(process_files(process_list_eng, options.input_audio_format), process_files(process_list_spa, options.input_audio_format)):
		proscript_mapping = map_segments(proscript_spa, proscript_eng)
		movie_output_path = os.path.join(os.path.dirname(proscript_eng.words_csv_path), '..')
		mapping_file_path = os.path.join(movie_output_path, '%s_mapping.txt'%proscript_eng.id)
		mapping_to_file(proscript_mapping, mapping_file_path, proscript_spa, proscript_eng)
		print("Mapping extracted to %s"%mapping_file_path)

		
	# pair_index = 0
	# for file_info_eng, file_info_spa in zip(process_list_eng, process_list_spa):
	# 	proscript_eng = process_file(file_info_eng['file_id'], 
	# 								 file_info_eng['file_in_audio'], 
	# 								 file_info_eng['file_in_srt'], 
	# 								 file_info_eng['output_dir'], 
	# 								 file_info_eng['lang'], 
	# 								 options.input_audio_format)
	# 	proscript_spa = process_file(file_info_spa['file_id'], 
	# 								 file_info_spa['file_in_audio'], 
	# 								 file_info_spa['file_in_srt'], 
	# 								 file_info_spa['output_dir'], 
	# 								 file_info_spa['lang'], 
	# 								 options.input_audio_format)
		
	# 	proscript_mapping = map_segments(proscript_spa, proscript_eng)
	# 	mapping_file_path = os.path.join(file_info_eng['output_dir'], '..', '%s_mapping.txt'%file_info_eng['file_id'])
	# 	print(mapping_file_path)
	# 	with open(mapping_file_path, "w") as mapping_file:
	# 		for matching_segment_ids in proscript_mapping:
	# 			mapping_file.write("%s - %s\n"%(matching_segment_ids[0], matching_segment_ids[1]))

if __name__ == "__main__":
	usage = "usage: %prog [-s infile] [option]"
	parser = OptionParser(usage=usage)
	parser.add_option("-e", "--filelist_eng", dest="list_of_files_eng", default=None, help="list of files to process in english. Each line with id, audio, xml, lang (tab separated)", type="string")	
	parser.add_option("-s", "--filelist_spa", dest="list_of_files_spa", default=None, help="list of files to process in spanish. Each line with id, audio, xml, lang (tab separated)", type="string")	
	parser.add_option("-o", "--output-dir", dest="output_dir", default=None, help="Output directory", type="string")
	parser.add_option("-f", "--input-audio-format", dest="input_audio_format", default="mp3", help="Audio format (wav, mp3 etc.)", type="string")

	(options, args) = parser.parse_args()
	main(options)