# -*- coding: utf-8 -*-
"""
Created on Tue Sep 29 19:54:32 2015

@author: Philippe
"""
import nltk

from nltk.tag import StanfordNERTagger
  
def get_entities(content):
    st = StanfordNERTagger('C:\Users\Philippe\Downloads\stanford-ner-2015-04-20\stanford-ner-2015-04-20\classifiers\english.all.3class.distsim.crf.ser.gz')
    entity_list = st.tag(content.split())

    return entity_list
    
def get_num_subject(entity_list):
    subject_num = {'PERSON':0, 'ORGANIZATION':0, "LOCATION":0,'O':0}
    for element in entity_list:
       for subject in subject_num:
           if subject == str(element[1]):
               subject_num[subject] = subject_num[subject] +1
               break
    return subject_num
    
def main():
    get_entities('Bob')
    


if __name__ == '__main__':
    main()
