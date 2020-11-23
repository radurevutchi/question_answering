from nltk.tag import StanfordNERTagger
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk import StanfordPOSTagger
from nltk.parse.stanford import StanfordDependencyParser
import stanfordnlp
import re
import spacy
import nltk
import os
import sys


#print(len(sys.argv))
assert(len(sys.argv) == 4)
afile = sys.argv[1]
qfile = sys.argv[2]
numq = int(sys.argv[3])

#print(afile,qfile,numq)




nlpspacy = spacy.load("en_core_web_md")


stanforddir = 'stanford-postagger-2018-10-16/'
modelfile = stanforddir + 'models/english-bidirectional-distsim.tagger'
jarfile = stanforddir + 'stanford-postagger.jar'
postagger = StanfordPOSTagger(model_filename=modelfile,path_to_jar=jarfile)



#nltk.download("punkt")
#nltk.download("wordnet")
#nltk.download('stopwords')
#nltk.download('averaged_perceptron_tagger')
#nltk.download('wordnet')



def posToWordnet(pos):

    first = pos[0]
    if first == 'J':
        return 'a'
    elif first == 'V':
        return 'v'
    elif first == 'R':
        return 'r'
    elif first == 'N':
        return 'n'
    else:
        return 'a'

def lemmatize(sentences):


    lemmatizer = WordNetLemmatizer()

    for i in range(len(sentences)):
        if len(sentences[i]) > 80:
            sentences[i] = ['zanzibar']

    tagged_s = postagger.tag_sents(sentences)

    #print("done")
    #print(len(tagged_s),len(sentences))
    for i in range(len(sentences)):
        #print(i)
        #print(sentences[i])

        if sentences[i] != []:
            #print(i)
            #print(len(tagged_s))
            pos_tags = tagged_s[i]
            tempSent = sentences[i]
            sentences[i] += list(map(lambda x: lemmatizer.lemmatize(x[0],posToWordnet(x[1])), pos_tags))



    return sentences



def tokenizeWords(sentences):


    return list(map(lambda x: word_tokenize(x), sentences))



def removeStopwords(sentences):


    stop_words = stopwords.words('english')

    for i in range(len(sentences)):

        temp_sent = []
        for j in range(len(sentences[i])):
            if sentences[i][j].lower() not in stop_words:
                temp_sent.append(sentences[i][j].lower())
        sentences[i] = temp_sent

    return sentences



def addPairs(sentences):

    for sentence in sentences:


        for i in range(len(sentence)-1):
            w1 = sentence[i]
            w2 = sentence[i+1]
            concat = w1 + '_' + w2
            sentence.append(concat)

    return sentences


def addPairsQ(question):

    question = question[0]

    for i in range(len(question)-1):
        w1 = question[i]
        w2 = question[i+1]
        concat = w1 + '_' + w2
        question.append(concat)


    return [question]

def preprocess(sentences):

    for i in range(len(sentences)):


        sentences[i] = re.sub("-+|'+",' ',sentences[i])
	

	# Replaces invalid sentence samples with 'invalid_input_x' code
        if sentences[i] != '':
            if sentences[i][0] in [':','*']:
                sentences[i] = 'invalid_input_x'
            elif len(sentences[i]) <= 10:
                sentences[i] = 'invalid_input_x'
            elif len(sentences[i].split('\n')) > 2:
                sentences[i] = 'invalid_input_x'

    return sentences

def parseArticle(filename):
    document = open(filename).read()
    sentences = sent_tokenize(document)
    original_sentences = sent_tokenize(document)
    #print(len(original_sentences))


    sentences = preprocess(sentences)
    #print(len(sentences))
    sentences = tokenizeWords(sentences)
    #print(len(sentences))
    sentences = lemmatize(sentences)
    #print(len(sentences))

    sentences = removeStopwords(sentences)
    sentences = addPairs(sentences)

    sentences = list(map(lambda x: set(x), sentences))


    assert(len(original_sentences) == len(sentences))

    return (original_sentences, sentences)




def parseQuestion(question):

    question = [question]
    question = preprocess(question)
    question = tokenizeWords(question)
    question = lemmatize(question)
    question = [question[0][:-1]]

    question = removeStopwords(question)
    question = addPairsQ(question)


    return set(question[0])



#def removePunctuation()

def topSentenceMatchesSets(question):

    #print('\n\n\n\n\n\n\n\n\n\n')
    #print(question)
    question_text = question
    question = parseQuestion(question)
    #print(question)
    #print('**********************\n\n\n')


    common = []

    for i in range(len(sentences)):

        inter = question.intersection(sentences[i])
        length = len(inter)
        common.append((inter, length, i))







    common = sorted(common, key=lambda x: x[1])


    top3 = common[-3:]

    #for i in range(3):
        #print(original_sentences[top3[i][2]])
        #print(top3[i])
        #print()


    return top3


def questionType(question_string):

    qtype = ' '.join(question_string.split()[:2]).lower()
    if qtype == 'how many':
        return qtype
    elif qtype == 'how long':
        return qtype

    qtype = question_string.split()[0].lower()

    return qtype


def answerWhere(top3, question):



    found = False
    entity_list = ['GPE','LOC','FAC','ORG']


    for answer in list(reversed(top3)):


        final_answers = {}
        if found:
            return

        text = preprocess([original_sentences[answer[2]]])[0]
        doc = nlpspacy(text)

        for ent in doc.ents:
            if ent.label_ in final_answers:
                final_answers[ent.label_] += [ent.text]
            else:
                final_answers[ent.label_] = [ent.text]

        i = 0
        while not found and i < len(entity_list):
            if entity_list[i] in final_answers:
                for instance in final_answers[entity_list[i]]:
                    if instance.lower() not in question.lower():
                        #print(entity_list[i])
                        return(instance)
                        found = True
            i += 1

    if not found:
        return(original_sentences[top3[2][2]])


def answerWhen(top3, question):



    found = False
    entity_list = ['DATE','TIME','CARDINAL']
    for answer in list(reversed(top3)):

        final_answers = {}
        if found:
            return

        text = preprocess([original_sentences[answer[2]]])[0]
        doc = nlpspacy(text)
        for ent in doc.ents:
            #print(ent.label_,ent.text)
            if ent.label_ in final_answers:
                final_answers[ent.label_] += [ent.text]
            else:
                final_answers[ent.label_] = [ent.text]


        i = 0
        while not found and i < len(entity_list):
            #print(final_answers)
            if entity_list[i] in final_answers:

                for instance in final_answers[entity_list[i]]:

                    if instance.lower() not in question.lower():
                        #print(entity_list[i])
                        #print(original_sentences[answer[2]])
                        return(instance)
                        found = True
            i += 1

    if not found:
        return(original_sentences[top3[2][2]])





def answerWho(top3, question):

    found = False
    entity_list = ['PERSON','ORG']
    for answer in list(reversed(top3)):

        final_answers = {}
        if found:
            return

        text = preprocess([original_sentences[answer[2]]])[0]
        doc = nlpspacy(text)
        for ent in doc.ents:

            if ent.label_ in final_answers:
                final_answers[ent.label_] += [ent.text]
            else:
                final_answers[ent.label_] = [ent.text]

        #print(final_answers)
        i = 0
        while not found and i < len(entity_list):
            if entity_list[i] in final_answers:
                for instance in final_answers[entity_list[i]]:
                    #print(instance)
                    if instance.lower() not in question.lower():
                        #print(entity_list[i])
                        return(instance)
                        found = True
            i += 1

    if not found:
        return(original_sentences[top3[2][2]])



def answerHowMany(top3, question):


    entity_list = ['CARDINAL','QUANTITY','ORDINAL']
    found = False

    for answer in list(reversed(top3)):

        final_answers = {}
        if found:
            return

        text = preprocess([original_sentences[answer[2]]])[0]
        doc = nlpspacy(text)
        for ent in doc.ents:

            if ent.label_ in final_answers:
                final_answers[ent.label_] += [ent.text]
            else:
                final_answers[ent.label_] = [ent.text]


        i = 0
        while not found and i < len(entity_list):
            if entity_list[i] in final_answers:
                for instance in final_answers[entity_list[i]]:
                    if instance.lower() not in question.lower():
                        #print(entity_list[i])
                        return(instance)
                        found = True
            i += 1


    if not found:
        return(original_sentences[top3[2][2]])




def answerIs(top3, question):

    found = False
    for answer in list(reversed(top3)):

        if found:
            return('YES')


        text = preprocess([original_sentences[answer[2]]])[0]
        setA = set(lemmatize([word_tokenize(text.lower())])[0])
        setQ = set(lemmatize([word_tokenize(question.lower())])[0])



        if len(setA.intersection(setQ)) >= len(setQ)-1:
            found = True
            return('YES')

    return('NO')



def answerWhy(top3, question):


    answer = preprocess([original_sentences[top3[2][2]]])[0]
    return(answer)

    return


def answerWhat(top3, question):

    doc1 = nlpspacy(question)

    main_verb = ''
    for token in doc1:
        if token.dep_ == 'ROOT':
            main_verb = token.text


    answer = preprocess([original_sentences[top3[2][2]]])[0]
    if answer.find(main_verb) == -1:
        return(answer)
    else:
        stop_words = stopwords.words('english')
        if main_verb not in stop_words:
            short_answer = answer[answer.find(main_verb) + len(main_verb):]
            return(short_answer)
        else:
            return(answer)


    return




####
# BEGIN PROCESSING
####





(original_sentences, sentences) = parseArticle(afile)




print("\n\n")

questions = open(qfile,'r').readlines()
for q in range(len(questions)):

    '''
    q1 = " ".join(questions[q].split()[1:])

    #print(q1)
    questions[q] = q1[:1+q1.index("?")] # keeps question mark
    '''

    questions[q] = questions[q][:-1] # removes newline


    if questions[q] != '':
        mytop3 = topSentenceMatchesSets(questions[q])
        qtype = questionType(questions[q])
        questions[q] = questions[q][:-1] # removes question mark


        if qtype == 'where':
            thefinalanswer = answerWhere(mytop3,questions[q])
        elif qtype in ['how long','when']:
            thefinalanswer = answerWhen(mytop3,questions[q])
        elif qtype == 'who':
            thefinalanswer = answerWho(mytop3,questions[q])
        elif qtype == 'how many':
            thefinalanswer = answerHowMany(mytop3,questions[q])
        elif qtype in ['how','why']:
            thefinalanswer = answerWhy(mytop3,questions[q])
        elif qtype in ['is','was','are','were','has','can','have']:
            thefinalanswer = answerIs(mytop3,questions[q])
        elif qtype in ['which','what']:
            thefinalanswer = answerWhat(mytop3,questions[q])
        else:
            thefinalanswer = answerWhy(mytop3,questions[q])




        print(questions[q])
        print('Answer:')
        print(thefinalanswer)
        print(original_sentences[mytop3[2][2]])
        print('\n\n\n')
