import argparse
import re
import os

ALL_PUNC = re.compile("[\\.\\,\\!\\?\\\"\\']")
DOT = re.compile("[\\.]")
STRIP_PUNC = re.compile("[\\,\\!\\?\\\"\\']")
WS = re.compile("\\s+")

# Strip punctuation
def stripPunc(text):
	text = STRIP_PUNC.sub('', text)
	return DOT.sub(' ', text)

# Normalize whitespace
def normWhitespace(text):
	return WS.sub(' ', text).strip()
	
def parseArgs():
	ap = argparse.ArgumentParser("Pre-process ASAP data.")
	ap.add_argument("input_file", help="ASAP-formatted training file")
	ap.add_argument("output_dir", help="Output directory")
	return ap.parse_args()


if __name__ == "__main__":
	
	args = parseArgs()
	
	# Read data
	print "Reading data..."
	data = {str(i):[] for i in range(1, 11)}
	with open(args.input_file) as f:
		for i, item in enumerate([l.strip().split("\t") for l in f.readlines()]):
			if i == 0:
				continue
			data[item[1]].append(
				{'id': item[0],
				 'score1': int(item[2]),
				 'score2': int(item[3]),
				 'text': item[4]})
	
	# Process text of the records
	print "Processing data..."
	for i in range(1, 11):
		for item in data[str(i)]:
			item['text'] = stripPunc(item['text'])
			item['text'] = normWhitespace(item['text'])
			item['text'] = item['text'].lower()
	
	print "Writing data..."
	if not os.path.exists(args.output_dir):
		os.makedirs(args.output_dir)
	for i in range(1, 11):
		print "\tWriting set " + str(i)
		with open(args.output_dir + '/train-' + str(i) + ".tsv", 'w') as f:
			for item in data[str(i)]:
				f.write('\t'.join([item['id'], str(item['score1']), str(item['score2']), item['text']]) + '\n')
	
	print "Done"