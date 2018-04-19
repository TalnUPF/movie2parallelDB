audio_format=mp3
audio_eng=/Users/alp/Movies/heroes/s2.7/heroes.s2.7_eng.mp3
audio_spa=/Users/alp/Movies/heroes/s2.7/heroes.s2.7_spa.mp3

sub_eng=/Users/alp/Movies/heroes/s2.7/heroes.s2.7_eng_ocr.srt
sub_spa=/Users/alp/Movies/heroes/s2.7/heroes.s2.7_spa_ocr.srt

output_eng=/Users/alp/Movies/heroes/s2.7/corpus/eng
output_spa=/Users/alp/Movies/heroes/s2.7/corpus/spa

#python src/subsegment_movie.py -a $audio_eng -s $sub_eng -o $output_eng -l eng -f $audio_format
python src/subsegment_movie.py -a $audio_spa -s $sub_spa -o $output_spa -l spa -f $audio_format
