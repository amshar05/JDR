import docx2txt

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.metrics.pairwise import cosine_similarity

import pandas as pd
import nltk

#nltk.download()
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)
resume = "" # ADD path

jobDesc = "" # ADD path

resume_2 = docx2txt.process(resume)

jobDesc_2 = docx2txt.process(jobDesc)


####################


text = [resume_2,jobDesc_2]

#print(text)
#######sklearn

cv = CountVectorizer()
count_matrix = cv.fit_transform(text)

#print("\n Similarity Scores: ")
#print(cosine_similarity(count_matrix))


# Match percentage

matchPercentage = cosine_similarity(count_matrix)[0][1]*100
matchPercentage = round(matchPercentage,2)

print("\n Result for ~NAME~ \n")

print("\nYour Resume matches about: " + str( matchPercentage)+ "% of the job description")

X_train_counts = cv.fit_transform(text)


x = pd.DataFrame(X_train_counts.toarray(),columns=cv.get_feature_names(),index=['resume','JD'])
y = x.transpose()

print("\nWords not present in resume are: ")
z = x.columns[(x==0).iloc[0]]
z1 = z.values.tolist()

z3 = z2 = [x for x in z1 if not (x.isdigit() or x[0] == '-' and x[1:].isdigit())]


z3 = [i for i in z2 if len(i) > 2]
#print(z3)


tagged = nltk.pos_tag(z3)

#list of all Parts of speech those are being filtered
to_check = ["JJ", "JJS","NNS" "NNP","RB","VBG"]

final =[]
i =0
while i < len(tagged):
	if tagged[i][1] in to_check:
		final.append(tagged[i][0])
	else:
		pass
	i=i+1

print(final)

print("\n")
#print(y.query('resume == "0"',inplace = True))

"""
>>> for line in jd_txt.split(" "):
...     content.append(line)
content_2 = nltk.pos_tag([i for i in content if i])

"""
