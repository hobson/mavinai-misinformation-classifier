#All these functions are for dummy purposes

import nltk
from nltk.tokenize import sent_tokenize
from numpy.random import randint,uniform
# from sklearn.feature_extraction.text import TfidfVectorizer

# from preprocessing import *

def chunker1(paragraph):
    '''
    Creates a "lengths" list having a distribution of length factor uniformly distributed 
    from {low=0.1 to high=0.9} 
    Paragraph is tokenized into sentences, preprocessed, and concatenated to output a
    list of sub-paragraphs, which represents the dataset
    Sentences are concatenated to get multiple paragraphs of length = length_factor*total word count of paragraph
    The length_factor varies as above.

    args:
        input: raw text
    
    returns:
        output: Chunked into units  
    '''
    num_words = len(paragraph.strip().split())
    lengths = [0.1,0.3,0.5,0.7,0.9]
    list_of_sentences= sent_tokenize(paragraph)
    
    #######Uncomment after importing class from preprocessing.py - it can be modified as per use#####
#     preprocess = Preprocessing_text()
#     list_of_sentences = preprocess.run(list_of_sentences)
    ###########
    
    i=0
    j=0
    list_of_paragraphs=[]
    while(i<len(lengths) and j+1<len(list_of_sentences)):
        curr_str=""+list_of_sentences[j]
        while(len(curr_str.split())<int(lengths[i]*num_words) and j<len(list_of_sentences)):
            curr_str = curr_str + list_of_sentences[j]
            j +=1
        list_of_paragraphs.append(curr_str)
        i+=1
        j+=1

    if j<len(list_of_sentences):
        list_of_paragraphs.append(''.join(list_of_sentences[j:]))

    return list_of_paragraphs


def chunker2(list_of_paragraphs, paragraph_count=2):
    """
    This mimics the logic of the Aylien API that provides text from news articles. 
    By taken a maximum of paragraph_count paragraphs in as text.

    args:
        input: list of paragraphs either from API, or scraped via beautiful soup
    
    returns:
        output: list of chunked paragraphs,  1-3 paragraphs in length
    
    """
    output = []
    # Format the paragraphs so they look identical to how the annotators view them
    list_of_paragraphs = [paragraph + "/n/n" for paragraph in list_of_paragraphs]
    while(len(list_of_paragraphs)>0) :
        try:
            output.append("".join(list_of_paragraphs[:paragraph_count]))
            list_of_paragraphs = list_of_paragraphs[paragraph_count:]
        except:
            output.append("".join(list_of_paragraphs))
            list_of_paragraphs=[]
            
    return output


def segment1(paragraph):
    '''
    method of segmentation: tokenizes sentences and concatenates ONLY consecutive strings randomly
    to get a distribution of length and variation in single/multi sentences 
    
    use:
        for each para in list of paragraphs : 
            get output form segment1(para)

    args:
        input:preprocessed string

    returns:
        output:list of segmented strings after removing extra spaces \\n\\n

    example usage:
        list of sentences with different length distribution = segment1(paragraph text after chunking)

    '''
    list_of_sentences = PunktSentenceTokenizer(str(paragraph)).tokenize(str(paragraph))
    output = []
    i=0
    while(i+1<len(list_of_sentences)):
        if(randint(0,2)): #outputs 0 or 1 with probability = 1/5 for both
            output.append(list_of_sentences[i]+" "+ list_of_sentences[i+1])
            i+=2
        else:
            output.append(list_of_sentences[i])
            i+=1
    output.append(list_of_sentences[i])
    while(i<len(list_of_sentences)):
        output[i] = output[i].strip()
        i+=1
    return output


# def segment2(paragraph,threshold):
#   '''
#   input:preprocessed string
#   output:list of segmented strings
#   method of segmentation: tokenizes sentences and combines consecutive strings based on similarity 
#   between tfidf vectors, given the threshold
#   '''
#   list_of_sentences = sent_tokenize(paragraph)
#   output=[]
#   # TO BE COMPLETED

#   return output
