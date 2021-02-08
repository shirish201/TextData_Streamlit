# -*- coding: utf-8 -*-
"""
Created on Thu Feb  4 23:25:49 2021

@author: GUPTAS2Q
"""

import pandas as pd
import numpy as np
import re
import streamlit as st
from datetime import datetime
from PIL import Image
import io
import plotly.graph_objs as go
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
from wordcloud import WordCloud
import plotly
import random
import seaborn as sns
import time
from wordcloud_function import *
import seaborn as sns
st.set_option('deprecation.showfileUploaderEncoding', False)   

def Upload_Page(): 
    
    st.title("Welcome to MY F.I.R.S.T. APP")
    st.header("Text Analysis Demo")
    st.subheader("We will create a quick app to work on Text Data")
#    st.header("This is how header looks like")
#    st.subheader("And the subheader")
#    st.write("And write something")
    
#    Just for testing out
#    st.sidebar.radio('Radio Button',[1,2])
#    st.sidebar.checkbox('Checkbox',["A","B"])
#    st.sidebar.write("This is how sidebar looks like")

    try:     
       text_data = st.file_uploader("Upload your text data...", type="csv", encoding = None)
       text_data = pd.read_csv(io.TextIOWrapper(text_data), sep = ",")       
       text_data.to_csv("extra_files//text_data.csv", index=False)
       
       text_data = text_data.drop("ID", axis=1)
       st.subheader("Sample of the Uploaded File")
       st.table(text_data.head())
       
       text_data_all = text_data[['Text 1',
       'Text 2',
       'Text 3']].melt()["value"]
       text_data_all = text_data_all.dropna()

       st.subheader("Summary of Text")
       
       st.info("Total Number of Texts: {}".format(str(len(text_data))))
       
       
    except:
        pass
             
def Survey_Result_Page():
    st.title("Survey Results")
    st.subheader("Word Cloud by Sentiment Analysis")
    text_data = pd.read_csv("extra_files//text_data.csv")
    text_data_all = text_data[['Name','Text 1',
       'Text 2',
       'Text 3']].melt(id_vars = "Name")[["Name","value"]]
    text_data_all = text_data_all.dropna()
    
    wordcloud_data_subset = word_clouds_preprocess(pd.DataFrame(text_data_all))
    
    wordcloud_data_subset.to_csv("extra_files//survey_results_withSentiments.csv", index=None)
    try:
        positive_wordcloud = create_wordclouds_bySentiment(wordcloud_data_subset, "pos")
    except:
        positive_wordcloud = WordCloud(
                    background_color = 'white',
                    width = 390,
                    height = 500,
                        ).generate("None")
    try:
        neutral_wordcloud = create_wordclouds_bySentiment(wordcloud_data_subset, "neu")
    except:
        neutral_wordcloud = WordCloud(
                    background_color = 'white',
                    width = 390,
                    height = 500,
                        ).generate("None")
    
    try:
        negative_wordcloud = create_wordclouds_bySentiment(wordcloud_data_subset, "neg")
    except:
        negative_wordcloud = WordCloud(
                    background_color = 'white',
                    width = 390,
                    height = 500,
                        ).generate("None")
        
    
    if len(positive_wordcloud.words_) > 1 and len(negative_wordcloud.words_) > 1 and len(neutral_wordcloud.words_) >1:
        all_wordclouds = [positive_wordcloud.to_array(),neutral_wordcloud.to_array(),negative_wordcloud.to_array()]
        caption_text  = ["Positive","Neutral","Negative"]
        
    elif len(positive_wordcloud.words_) == 1 and len(negative_wordcloud.words_) >1 and len(neutral_wordcloud.words_) >1:
        all_wordclouds = [neutral_wordcloud.to_array(),negative_wordcloud.to_array()]
        caption_text  = ["Neutral","Negative"]
        
    elif len(positive_wordcloud.words_) == 1 and len(negative_wordcloud.words_) == 1 and len(neutral_wordcloud.words_) >1:
        all_wordclouds = [neutral_wordcloud.to_array()]
        caption_text  = ["Neutral"]
        
    elif len(positive_wordcloud.words_) == 1 and len(negative_wordcloud.words_) >1 and len(neutral_wordcloud.words_) == 1:
        all_wordclouds = [negative_wordcloud.to_array()]
        caption_text  = ["Negative"]
        
    elif len(positive_wordcloud.words_) >1 and len(negative_wordcloud.words_) == 1 and len(neutral_wordcloud.words_) ==1:
        all_wordclouds = [positive_wordcloud.to_array()]
        caption_text  = ["Positive"]
        
    elif len(positive_wordcloud.words_) >1 and len(negative_wordcloud.words_) == 1 and len(neutral_wordcloud.words_) >1:
        all_wordclouds = [positive_wordcloud.to_array(),neutral_wordcloud.to_array()]
        caption_text  = ["Positive","Neutral"]
        
    elif len(positive_wordcloud.words_) >1 and len(negative_wordcloud.words_) >1 and len(neutral_wordcloud.words_) == 1:
        all_wordclouds = [positive_wordcloud.to_array(),negative_wordcloud.to_array()]
        caption_text  = ["Positive","Negative"]
        

    
    st.image(all_wordclouds,caption = caption_text,)
    
    st.write("-----")
    st.subheader("Total Counts by Sentiments")
 
    sentiment_grouped = wordcloud_data_subset.groupby("Sentiment", as_index=False).size()
    fig = sns.barplot(x="Sentiment", y="size", data=sentiment_grouped)
    for index, row in sentiment_grouped.iterrows():
        fig.text(row.name, row[1],row[1], color='black', ha="center")
    
    st.pyplot(fig.figure)
    
    fig.figure.savefig("extra_files//SentimentGroups.png")
    st.write("----")
    
    st.subheader("Enter a key word to search in the survey results")
    
    st.markdown(
        """<style>
            .table {text-align: left !important}
        </style>
        """, unsafe_allow_html=True) 
   
    word_search = st.text_input("Enter a keyword to filter survey results")
    
    if word_search != "":   
    
        filter_data = text_data_all[text_data_all["value"].str.contains(word_search, na=False)]
        filter_data = filter_data.reset_index(drop=True)
        try:
            st.table(filter_data)
        except:
            pass
    
    return 

    
def post_login():
    pages = {
        "Upload Page": Upload_Page,
        "Second Page": Survey_Result_Page,
    }
   
    page = st.sidebar.selectbox("Select your page", tuple(pages.keys()))
        
    
    if page == "Upload Page":
        pages["Upload Page"]()
    elif page == "Second Page":
        pages["Second Page"]()
    
   
def _max_width_():
    max_width_str = f"max-width: 2000px;"
    st.markdown(
        f"""
    <style>
    .reportview-container .main .block-container{{
        {max_width_str}
    }}
    </style>    
    """,
        unsafe_allow_html=True,
    )
    
def main():
    _max_width_()
    st.image("extra_files//streamlit_logo3.png", width = 300)
    post_login()
        
       

    return

if __name__ == "__main__":
    main()
