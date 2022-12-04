import language_tool_python
import docx2txt
import pandas as pd

tool = language_tool_python.LanguageTool('en-US')
pd.set_option("display.max_rows", None, "display.max_columns", None)

file = "/Users/amit/Documents/grammar/Circular Economy_Ellen MacArthur.docx"
def wordtotxt(file_loc):
    text = docx2txt.process(file_loc)
    return text
 

def grammar_result(text):
    matches = tool.check(text)
    #print (matches)
    df_grammar = pd.DataFrame.from_dict(matches)
    df_grammar.columns=['Rule Id','Message','Replacements','offsetInContext', 
    'Context','Offset','Error Length','Category', 'RuleIssueType','Sentence']
    return df_grammar



if __name__ == '__main__':
     text_to_check = wordtotxt(file)
     print(grammar_result(text_to_check))


