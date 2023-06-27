import csv
from collections import defaultdict
from nltk import sent_tokenize
import re
import nltk
nltk.download('punkt')


# --- Subject & object markup ---
SUB_START_CHAR = "<e1>"
SUB_END_CHAR = "<\e1> "
OBJ_START_CHAR = "<e2>"
OBJ_END_CHAR = "<\e2> "

re_id2text={'CPR:2': {'DIRECT-REGULATOR', 'INDIRECT-REGULATOR', 'REGULATOR'},
            'CPR:3': {'ACTIVATOR', 'INDIRECT-UPREGULATOR', 'UPREGULATOR'},
            'CPR:4': {'DOWNREGULATOR', 'INDIRECT-DOWNREGULATOR', 'INHIBITOR'},
            'CPR:5': {'AGONIST', 'AGONIST-ACTIVATOR', 'AGONIST-INHIBITOR'},
            'CPR:6': {'ANTAGONIST'},
            'CPR:9': {'PRODUCT-OF', 'SUBSTRATE'},
            'CPR:1': {'PART-OF'},
            'CPR:10': {'NOT'},
            'CPR:7': {'MODULATOR', 'MODULATOR-ACTIVATOR', 'MODULATOR-INHIBITOR'},
            'CPR:8': {'COFACTOR'}}

def SentToken(text):
    splited_text=sent_tokenize(text)
    #splited_text = re.split(r'(?<=[.?!])\s+', text)
    SentsSpan=[]
    #total_len=0
    sent_start=0
    sent_end=0
    sent_len=0
    for sent_id,i in enumerate(splited_text):
      if sent_id!=0:
        sent_start=sent_end+1
      sent_len=len(i)
      sent_end=sent_start+sent_len
    #total_len+=len(i)
      SentsSpan.append([sent_start,sent_end])
    return splited_text,SentsSpan


def add_marker(abstract_text, start, end, text, subject=True):
    if subject:
        new_text = str(abstract_text[:start]) + str(SUB_START_CHAR) + str(text) + str(SUB_END_CHAR) + str(abstract_text[end+1:])
    else:
        new_text = str(abstract_text[:start]) + str(OBJ_START_CHAR) + str(text) + str(OBJ_END_CHAR) + str(abstract_text[end+1:])
    return new_text


def ProcessingPositiveSent(relation_list, entity_dic, abstract_dic):
    PositiveSents = defaultdict(dict)
    re_id_dic = defaultdict(dict)

    #for each relation, get its original sentence
    for row in relation_list:
        # gold: ['23538162', 'CPR:4', 'Arg1:T5', 'Arg2:T19']
        # train: ['10047461', 'CPR:3', 'Y ', 'ACTIVATOR', 'Arg1:T13', 'Arg2:T57']
        # ['10047461', 'CPR:3', 'Arg1:T13', 'Arg2:T55']
        # test: ['10076535', 'CPR:2', 'N ', 'DIRECT-REGULATOR', 'Arg1:T23', 'Arg2:T55']
        pmid = row[0]
        re_type = row[1]
        re_positive = row[2] # if True, the relation is evaluated

        #re_type_text = re_id2text[re_type]
        re_type_text = row[3]

        arg1_id = str(row[4]).split(":")[1]
        arg2_id = str(row[5]).split(":")[1]

        #retrive the arg text
        arg1 = entity_dic[pmid][arg1_id]
        arg2 = entity_dic[pmid][arg2_id]

        #get the abstract
        abstract_text, splited_text, SentsSpan = abstract_dic[pmid]['text'], abstract_dic[pmid]['splited_text'], \
                                         abstract_dic[pmid]['SentsSpan']
        #add arg1 and arg2 markers
        if arg1['end'] > arg2['end']:
            # add_marker(abstract_text,start,end, text,subject=True)
            new_text = add_marker(abstract_text, arg1['start'], arg1['end'], arg1['text'], True)
            new_text = add_marker(new_text, arg2['start'], arg2['end'], arg2['text'], False)
        else:
            new_text = add_marker(abstract_text, arg2['start'], arg2['end'], arg2['text'], False)
            new_text = add_marker(new_text, arg1['start'], arg1['end'], arg1['text'], True)

        #Only use ". " to split the sentence
        #Other option, use the splited_text separate by sent_tokenize
        sentences = new_text.split(". ")
        if pmid not in re_id_dic.keys():
            re_id_dic[pmid] = 0
        positive_sent = 'none'
        #Find that sentence that have the arg1 and arg2
        for s in sentences:
            if SUB_START_CHAR in s and SUB_END_CHAR in s and OBJ_START_CHAR in s and OBJ_END_CHAR in s:
                positive_sent = s
                break
        #only add evaluated data
        if 'Y' in re_positive:
            PositiveSents[pmid][re_id_dic[pmid]] = {'sentence': positive_sent, 'label': re_type,
                                                    'label_text': re_type_text, 're_positive': re_positive,
                                                    'arg1': arg1['text'], 'arg2': arg2['text'],
                                                    'arg1_type': arg1['type'], 'arg2_type': arg2['type']}

        # if any(char not in string.printable for char in positive_sent):
        #    print(f"{pmid}:Non-ASCII character found: {positive_sent}")
        re_id_dic[pmid] += 1
        # print(abstract_text)
        # print(new_text)
    return PositiveSents

if __name__=='__main__':
    anstract_path='FILE_PATH'
    entity_path='FILE_PATH'
    relation_path='FILE_PATH'

    abstract_dic = defaultdict(dict)
    with open(anstract_path, 'r') as tsv_file:
        abstract_file = csv.reader(tsv_file, delimiter='\t')
        abstract_file = list(abstract_file)

        for row in abstract_file:
            pmid = row[0]
            text = str(row[1]) + " " + str(row[2])
            splited_text, SentsSpan = SentToken(text)
            abstract_dic[pmid] = {'title': row[1], 'abstract': row[2], 'text': text, 'splited_text': splited_text,
                                  'SentsSpan': SentsSpan}

    etities_dic = defaultdict(dict)
    with open(entity_path, 'r') as tsv_file:
        etities_file = csv.reader(tsv_file, delimiter='\t')
        etities_file = list(etities_file)
        for row in etities_file:
            pmid = row[0]
            entity_id = row[1]
            entity_type = row[2]
            start = int(row[3])
            end = int(row[4])
            text = row[5]
            etities_dic[pmid][entity_id] = {'type': entity_type, 'start': start, 'end': end, 'text': text}

    #load relation file
    with open(relation_path, 'r') as tsv_file:
        RE_file = csv.reader(tsv_file, delimiter='\t')
        RE_file = list(RE_file)

    PositiveSents = ProcessingPositiveSent(RE_file, etities_dic, abstract_dic)

    with open('OUT_PATH', 'w', encoding='utf-8',newline='') as f:
        writer = csv.writer(f)
        for pmid in PositiveSents.keys():
            for re_id in PositiveSents[pmid].keys():
                re_annotation = PositiveSents[pmid][re_id]
                # [s.encode('utf-8').decode('utf-8') if isinstance(s, str) else s for s in row]
                row = [pmid, re_id, re_annotation['sentence'], re_annotation['label'], re_annotation['re_positive'],
                       re_annotation['label_text'], re_annotation['arg1'], re_annotation['arg2'],
                       re_annotation['arg1_type'], re_annotation['arg2_type']]
                writer.writerow([s.encode('utf-8').decode('utf-8') if isinstance(s, str) else s for s in row])
