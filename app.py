import streamlit as st
import preprocessor
import helper
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import cleantext
from textblob import TextBlob



st.sidebar.title("Whatsapp Chat Analyzer")

uploaded_file = st.sidebar.file_uploader("Choose a file")
if uploaded_file is not None:
    bytes_data = uploaded_file.getvalue()
    data = bytes_data.decode("utf-8")
    df = preprocessor1.preprocess(data)
    df = preprocessor1.add_sentiment_analysis(df)

    # fetch unique users
    user_list = df['user'].unique().tolist()
    user_list.remove('group_notification')
    user_list.sort()
    user_list.insert(0, "Overall")

    selected_user = st.sidebar.selectbox("Show analysis wrt", user_list)

    if st.sidebar.button("Show Analysis"):

        # Stats Area
        num_messages, words, num_media_messages, num_links = helper1.fetch_stats(selected_user, df)
        st.title("Top Statistics")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.header("Total Messages")
            st.title(num_messages)
        with col2:
            st.header("Total Words")
            st.title(words)
        with col3:
            st.header("Media Shared")
            st.title(num_media_messages)
        with col4:
            st.header("Links Shared")
            st.title(num_links)

        # monthly timeline
        st.title("Monthly Timeline")
        timeline = helper1.monthly_timeline(selected_user, df)
        fig, ax = plt.subplots()
        ax.plot(timeline['time'], timeline['message'], color='green')
        plt.xticks(rotation='vertical')
        st.pyplot(fig)

        # daily timeline
        st.title("Daily Timeline")
        daily_timeline = helper1.daily_timeline(selected_user, df)
        fig, ax = plt.subplots()
        ax.plot(daily_timeline['only_date'], daily_timeline['message'], color='black')
        plt.xticks(rotation='vertical')
        st.pyplot(fig)

        # activity map
        st.title('Activity Map')
        col1, col2 = st.columns(2)

        with col1:
            st.header("Most busy day")
            busy_day = helper1.week_activity_map(selected_user, df)
            fig, ax = plt.subplots()
            ax.bar(busy_day.index, busy_day.values, color='purple')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)

        with col2:
            st.header("Most busy month")
            busy_month = helper1.month_activity_map(selected_user, df)
            fig, ax = plt.subplots()
            ax.bar(busy_month.index, busy_month.values, color='orange')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)

        st.title("Weekly Activity Map")
        user_heatmap = helper1.activity_heatmap(selected_user, df)
        fig, ax = plt.subplots()
        ax = sns.heatmap(user_heatmap)
        st.pyplot(fig)

        # finding the busiest users in the group(Group level)
        if selected_user == 'Overall':
            st.title('Most Busy Users')
            x, new_df = helper1.most_busy_users(df)
            fig, ax = plt.subplots()

            col1, col2 = st.columns(2)

            with col1:
                ax.bar(x.index, x.values, color='red')
                plt.xticks(rotation='vertical')
                st.pyplot(fig)
            with col2:
                st.dataframe(new_df)

        # WordCloud
        st.title("Wordcloud")
        df_wc = helper1.create_wordcloud(selected_user, df)
        fig, ax = plt.subplots()
        ax.imshow(df_wc)
        st.pyplot(fig)

        # most common words
        most_common_df = helper1.most_common_words(selected_user, df)

        fig, ax = plt.subplots()

        ax.barh(most_common_df[0], most_common_df[1])
        plt.xticks(rotation='vertical')

        st.title('Most Commmon Words')
        st.pyplot(fig)

        # emoji analysis
        # emoji_df = helper1.emoji_helper(selected_user, df)
        # st.title("Emoji Analysis")

        # col1, col2 = st.columns(2)

        # with col1:
        #     st.dataframe(emoji_df)
        # with col2:
        #     fig, ax = plt.subplots()
        #     ax.pie(emoji_df[1].head(), labels=emoji_df[0].head(), autopct="%0.2f")
        #     st.pyplot(fig)
        

        def visualize_sentiment(df):
            plt.figure(figsize=(10, 6))
            plt.plot(df.index, df['Sentiment_Vader'], marker='o', linestyle='-')
            plt.title('Sentiment Analysis')
            plt.xlabel('Message')
            plt.ylabel('Sentiment Score')
            plt.xticks(rotation=45)
            plt.grid(True)
            plt.show()
        
        st.title('Sentiment Analysis Visualization')
        st.write("This is a visualization of sentiment analysis using Streamlit.")
        
        visualize_sentiment(df)
        
        
        # st.title("Sentiment Analysis")
        # sentimentAna = helper1.add_sentiment_analysis(df)
        # fig, ax = plt.subplots()
        # ax = sns.heatmap(sentimentAna)
        # st.pyplot(fig)
      
    # Sentiment Analysis Part
            
    # st.header('Sentiment Analysis:')
    # with st.expander('Analyze Text'):
    #     text = st.text_input('Text, Here:')
    #     if text:
    #         blob=TextBlob(text)
    #         st.write('Polarity:',round(blob.sentiment.polarity,2))
    #         st.write('Subjectivity:',round(blob.sentiment.subjectivity,2))
    #     pre = st.text_input('Clean Text:')
    #     if pre:
    #         st.write(cleantext.clean(pre,clean_all=False,extra_spaces=True,stopwords=True,lowercase=True,numbers=True,punct=True))
            
    # with st.expander('Analyze Sentiment'):
    #     def score(x):
    #         blob1=TextBlob(x)
    #         return blob1.sentiment.polarity
    #     def analyze(x):
    #         if x>=0.5:
    #             return 'Positive'
    #         elif x<=-0.5:
    #             return 'Negative'
    #         else:
    #             return 'Neutral'
    #     del df['Unnamed:0']
    #     df['score'] = df['messages']
    
