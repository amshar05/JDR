import spacy
import pandas as pd
nlp = spacy.load('en_core_web_sm')
# Import the PhraseMatcher library
from spacy.matcher import PhraseMatcher
matcher = PhraseMatcher(nlp.vocab)


def compare_skills_list(role, resume_file):
	manual_skill_data = pd.read_excel("skills.xlsx")
	phrase_list = []

	for i in manual_skill_data[role]:
		if str(i) != 'nan':
			phrase_list.append(i.lower())

	#this is remains of a trial
	#phrase_list = ['business analyst', 'data analytics', 'web development', 'user acceptance testing']
	#Next, convert each phrase to a Doc object:
	phrase_patterns = [nlp(text) for text in phrase_list]
	#Pass each Doc object into matcher (note the use of the asterisk!):
	matcher.add('phrase_patterns', None, *phrase_patterns)
	#Build a list of matches:
	matches = matcher(resume_file)
	#(match_id, start, end)
	#print(matches)
	matched_list = []
	for match_id, start, end in matches:
		string_id = nlp.vocab.strings[match_id]  
		span = resume_file[start:end]
		matched_list.append(span.text)
		#print(match_id, string_id, start, end, span.text)

	#compare
	#missing words
	manual_not_present_list = [string for string in phrase_list if string not in matched_list]
	manual_present_list = [string for string in phrase_list if string in matched_list]
	return manual_not_present_list,manual_present_list,phrase_list












