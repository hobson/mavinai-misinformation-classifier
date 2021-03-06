import nltk, re , unicodedata , string
from nltk.corpus import stopwords
nltk.download('stopwords')
from nltk.stem import WordNetLemmatizer 
nltk.download('wordnet')
from nltk.tokenize import word_tokenize 
from nltk.tokenize import sent_tokenize
from nltk.tokenize import PunktSentenceTokenizer
nltk.download('punkt')

class Preprocessing_text:
    def __init__ (self):
        print("Preprocessing object created! \n")
    
    def run(self,list_of_strings):
        """Run function to apply all required functions on a list of strings together"""
        for i,string in enumerate(list_of_strings):
            list_of_strings[i] =  to_lower(remove_non_ascii(replace_nan(string)))
        return list_of_strings
        
    def remove_URL(self,sample_str):
        """Remove URLs from a sample string"""
        return re.sub(r"http\S+", "", sample_str)

    def remove_non_ascii(self,sample_str):
        """Remove non-ASCII characters from a sample string [sample_str]"""
        words = word_tokenize(sample_str)
        new_words = []
        for word in words:
            new_word = unicodedata.normalize('NFKD', word).encode('ascii', 'ignore').decode('utf-8', 'ignore')
            new_words.append(new_word)
        return ' '.join(new_words)

    def remove_punctuation(self,sample_str):
        """Remove punctuation from a sample string"""
        words = word_tokenize(sample_str)
        new_words = []
        for word in words:
            new_word = re.sub(r'[^\w\s]', '', word)
            if new_word != '':
                new_words.append(new_word)
        return ' '.join(new_words)

    def remove_stopwords(self,sample_str,stopword_list=list(set(stopwords.words('english')))):
        """Remove stop words from a sample string"""
        words = word_tokenize(sample_str)
        new_words = []
        for word in words:
            if word not in stopword_list:
                new_words.append(word)
        return ' '.join(new_words)

    def to_lower(self,sample_str):
        """ Converting all words to lowercase in a sample string"""
        return sample_str.lower()


    def lemmatize_postags(self,sample_str):
        """Lemmatize verbs,adj and noun in a sample string"""
        tokenizer = word_tokenize()
        words = tokenizer.tokenize(words)
        lemmatizer = WordNetLemmatizer()
        lemmas = []
        for word in words:
            word = lemmatizer.lemmatize(word, pos='v')
            word = lemmatizer.lemmatize(word, pos='n')
            word = lemmatizer.lemmatize(word, pos='a')
            lemmas.append(word)
        return ' '.join(lemmas)

    def replace_nan(self,sample_str):
        """Replacing nan strings with empty strings - required for textrank"""        
        sample_str_new = re.sub('nan' , '' , str(sample_str))
        return sample_str_new
