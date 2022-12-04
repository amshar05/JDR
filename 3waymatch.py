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

skill = "" #Add path of the skill set file
inter_skill = []

with open(skill,'r') as f:
	for i in f:
		inter_skill.append(i.strip())

skill_2 = (" ".join(inter_skill))

resume_2 = docx2txt.process(resume)

jobDesc_2 = docx2txt.process(jobDesc)





text = [skill_2,jobDesc_2]

#print(text)
#######sklearn

cv = CountVectorizer()
count_matrix = cv.fit_transform(text)



X_train_counts = cv.fit_transform(text)


x = pd.DataFrame(X_train_counts.toarray(),columns=cv.get_feature_names(),index=['Skill','JD'])
y = x.transpose()
z_inter = list(y[(y.Skill >0) & (y.JD>0)].index)
z = x.columns[(x==0).iloc[0]]
z1 = z.values.tolist()

print("this is z1")
print(z1)

z2 = [x for x in z_inter if not (x.isdigit() or x[0] == '-' and x[1:].isdigit())]

z3 = (" ".join(z2))
#################################




text_2 = [z3,resume_2]
count_matrix_2 = cv.fit_transform(text_2)


matchPercentage_2 = cosine_similarity(count_matrix_2)[0][1]*100
matchPercentage_2 = round(matchPercentage_2,2)

print("\n Result for ~NAME~ \n")

print("\nYour Resume matches about: " + str( matchPercentage_2)+ "% of the job description")

X_train_counts_2 = cv.fit_transform(text_2)


x_2 = pd.DataFrame(X_train_counts_2.toarray(),columns=cv.get_feature_names(),index=['Common_skill','Resume'])
y_2 = x_2.transpose()
z_inter_2 = list(y_2[(y_2.Common_skill >0) & (y_2.Resume==0)].index)
print("\nWords not present in resume are: \n")
zz = x.columns[(x==0).iloc[0]]
zz1 = zz.values.tolist()

print(z_inter_2)




