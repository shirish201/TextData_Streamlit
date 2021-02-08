
import pandas as pd
from wordcloud import WordCloud, STOPWORDS
import numpy as npy
import matplotlib.pyplot as plt 
import pandas as pd 
from PIL import Image
import nltk
from nltk.corpus import stopwords 
import re
# for Stemming propose  
from nltk.stem.porter import PorterStemmer 
from nltk.tokenize import word_tokenize 
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyser = SentimentIntensityAnalyzer()

def word_clouds_preprocess(text_data):
    
    
    wordcloud_data_subset=text_data.copy()
    wordcloud_data_subset=wordcloud_data_subset.reset_index(drop=True)
    wordcloud_data_subset=wordcloud_data_subset.fillna(' ')
    
    for i in range(0, len(wordcloud_data_subset)):  
            # column : "Review", row ith 
        review = re.sub('[^a-zA-Z]', ' ', wordcloud_data_subset["value"][i])  
              
            # convert all cases to lower cases 
        review = review.lower()
        sentiment_score = pd.DataFrame(analyser.polarity_scores(review).items())
        sentiment_score = sentiment_score[sentiment_score[1] == sentiment_score[1][0:3].max()]
              
            # split to array(default delimiter is " ") 
            #review = review.split()  
              
            # creating PorterStemmer object to 
            # take main stem of each word 
            #ps = PorterStemmer()  
              
            # loop for stemming each word 
            # in string array at ith row  
        stop_words = set(stopwords.words('english')) 
        word_tokens = word_tokenize(review)
        filtered_sentence = [] 
        for w in word_tokens: 
            if w not in stop_words: 
                filtered_sentence.append(w) 
                
            # rejoin all string array elements 
            # to create back into a string 
        review = ' '.join(filtered_sentence)   
        wordcloud_data_subset.loc[i,'Response'] = review
        wordcloud_data_subset.loc[i,'Sentiment'] = sentiment_score.iloc[0,0] 
        wordcloud_data_subset.loc[i,'Sentiment Score'] = sentiment_score.iloc[0,1] 
        
        
    return wordcloud_data_subset
        
        
def create_wordclouds_bySentiment(wordcloud_data_subset, sentiment):
        
        wordcloud_data_subset = wordcloud_data_subset[wordcloud_data_subset["Sentiment"] == sentiment]
        comment_words = '' 
        stopwords = set(STOPWORDS) 

        for val in wordcloud_data_subset.Response:
            
        	
            	# typecaste each val to string 
            	val = str(val) 
            	# split the value 
            	tokens = val.split() 	
            	# Converts each token into lowercase 
            	for i in range(len(tokens)): 
            		tokens[i] = tokens[i].lower() 
                    
                    
            	comment_words += " ".join(tokens)+" "
            
        wordcloud = WordCloud(max_words=50,
                    background_color = 'white',
                    width = 390,
                    height = 500,
                    stopwords = stopwords,
                    include_numbers = True
                        ).generate(comment_words)
        
        wc = pd.DataFrame.from_dict(wordcloud.words_, orient="index").reset_index()
        
        for j in range(len(wc)):
            cnt_word = comment_words.count(wc["index"][j])
            wc.loc[j,"Count of Words"] = cnt_word
        
        wc = wc.rename(columns = {"index":"Word", 0:"Normalized Frequency"})
        
        wc.to_csv("extra_files\\wc_cnt_{}.csv".format(sentiment))
        wc_image = wordcloud.to_image()
        wc_image.save("extra_files\\wc_cnt_{}.png".format(sentiment))

        return wordcloud
