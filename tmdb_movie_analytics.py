

import requests
import time
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go


from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, silhouette_score


from textblob import TextBlob
import re


try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except:
    STREAMLIT_AVAILABLE = False



API_KEY = "196a4950d6d4fbe2aeb903d4a605a9ac"
BASE_URL = "https://api.themoviedb.org/3"



class TMDBDataCollector:
    """
    Collect movie data from TMDB API
    Uses JSON API calls to fetch structured data
    """
    
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = BASE_URL
        self.genre_map = {}
        
    def get_genres(self):
        """Fetch genre mapping from TMDB API"""
        url = f"{self.base_url}/genre/movie/list?api_key={self.api_key}&language=en-US"
        try:
            response = requests.get(url)
            data = response.json()
            self.genre_map = {g['id']: g['name'] for g in data['genres']}
            print(f"✅ Loaded {len(self.genre_map)} genres")
            return self.genre_map
        except Exception as e:
            print(f"❌ Error fetching genres: {e}")
            return {}
    
    def fetch_movies(self, num_pages=150, endpoint='top_rated'):
        """
        Fetch movies from TMDB API
        
        Parameters:
        - num_pages: Number of pages to fetch (20 movies per page)
        - endpoint: 'top_rated', 'popular', or 'now_playing'
        
        Returns: List of movie dictionaries
        """
        movies_data = []
        
        if not self.genre_map:
            self.get_genres()
        
        print(f"\n📥 DATA ACQUISITION: Fetching {num_pages * 20} movies from TMDB API")
        print(f"   Endpoint: {endpoint}")
        print(f"   Method: JSON API calls\n")
        
        for page in range(1, num_pages + 1):
            try:
                url = f"{self.base_url}/movie/{endpoint}?api_key={self.api_key}&language=en-US&page={page}"
                response = requests.get(url, timeout=10)
                
                if response.status_code != 200:
                    print(f"  ⚠️ Error on page {page}: Status {response.status_code}")
                    continue
                
                data = response.json()
                
                for movie in data.get('results', []):

                    genre_names = [self.genre_map.get(g_id, 'Unknown') 
                                 for g_id in movie.get('genre_ids', [])]
                    
                    release_date = movie.get('release_date', '')
                    year = int(release_date[:4]) if release_date and len(release_date) >= 4 else 0
                    
                    movies_data.append({
                        'title': movie.get('title', 'Unknown'),
                        'year': year,
                        'rating': movie.get('vote_average', 0),
                        'votes': movie.get('vote_count', 0),
                        'popularity': movie.get('popularity', 0),
                        'genres': ', '.join(genre_names),
                        'language': movie.get('original_language', 'unknown'),
                        'description': movie.get('overview', ''),
                        'adult': movie.get('adult', False),
                        'movie_id': movie.get('id', 0)
                    })
                
                
                if page % 10 == 0:
                    print(f"  ✅ Progress: {page}/{num_pages} pages | {len(movies_data)} movies collected")
                
                
                time.sleep(0.25)
                
            except Exception as e:
                print(f"  ❌ Error on page {page}: {e}")
                continue
        
        print(f"\n✅ Data Acquisition Complete: {len(movies_data)} movies collected\n")
        return movies_data


class DataPreprocessor:

    
    @staticmethod
    def clean_data(df):
        """Comprehensive data cleaning"""
        print("🧹 DATA CLEANING & PREPROCESSING\n")
        
        initial_count = len(df)
        print(f"  Initial records: {initial_count}")
        
        
        df = df.drop_duplicates(subset=['title', 'year'], keep='first')
        print(f"  Removed {initial_count - len(df)} duplicates")
        
        
        df = df[df['rating'] > 0]
        df = df[df['votes'] > 0]
        print(f"  Removed movies with no ratings/votes")
        
        
        df['description'].fillna('No description available', inplace=True)
        df['genres'].fillna('Unknown', inplace=True)
        
        
        vote_threshold = df['votes'].quantile(0.99)
        df = df[df['votes'] <= vote_threshold]
        
        
        df['title'] = df['title'].str.strip()
        df['description'] = df['description'].str.strip()
        
        print(f"  Final clean dataset: {len(df)} records")
        print(f"✅ Cleaning complete\n")
        
        return df.reset_index(drop=True)
    
    @staticmethod
    def engineer_features(df):
        
        print("⚙️ FEATURE ENGINEERING\n")
        
        
        df['rating_category'] = pd.cut(df['rating'], 
                                       bins=[0, 5, 6.5, 7.5, 8.5, 10],
                                       labels=['Poor', 'Average', 'Good', 'Great', 'Excellent'])
        
        
        df['popularity_score'] = df['votes'] * df['rating']
        
        
        df['decade'] = (df['year'] // 10) * 10
        df['decade'] = df['decade'].apply(lambda x: f"{x}s" if x > 0 else 'Unknown')
        
        
        df['vote_category'] = pd.cut(df['votes'],
                                     bins=[0, 100, 500, 1000, 5000, 100000],
                                     labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
        
        
        df['primary_genre'] = df['genres'].apply(lambda x: x.split(',')[0].strip() if x else 'Unknown')
        
       
        df['description_length'] = df['description'].apply(len)
        df['description_word_count'] = df['description'].apply(lambda x: len(x.split()))
        
        print("  Performing sentiment analysis on movie descriptions...")
        df['sentiment_polarity'] = df['description'].apply(
            lambda x: TextBlob(str(x)).sentiment.polarity
        )
        df['sentiment_subjectivity'] = df['description'].apply(
            lambda x: TextBlob(str(x)).sentiment.subjectivity
        )

        
        df['sentiment_category'] = pd.cut(df['sentiment_polarity'],
                                          bins=[-1, -0.1, 0.1, 1],
                                          labels=['Negative', 'Neutral', 'Positive'])
        
        
        current_year = datetime.now().year
        df['is_recent'] = df['year'] >= (current_year - 10)
        
        
        df['language_category'] = df['language'].apply(
            lambda x: 'English' if x == 'en' else 'Non-English'
        )
        
        print(f"  ✅ Created 10 new features")
        print(f"✅ Feature engineering complete\n")
        
        return df



class MovieAnalytics:
    
    
    def __init__(self, df):
        self.df = df
        self.results = {}
        
    def descriptive_statistics(self):
       
        print("📊 DESCRIPTIVE STATISTICS\n")
        
        stats = {
            'total_movies': len(self.df),
            'avg_rating': self.df['rating'].mean(),
            'median_rating': self.df['rating'].median(),
            'avg_votes': self.df['votes'].mean(),
            'total_votes': self.df['votes'].sum(),
            'year_range': f"{self.df['year'].min()} - {self.df['year'].max()}",
            'unique_genres': len(self.df['primary_genre'].unique()),
            'avg_sentiment': self.df['sentiment_polarity'].mean()
        }
        
        print(f"  Total Movies: {stats['total_movies']:,}")
        print(f"  Average Rating: {stats['avg_rating']:.2f}/10")
        print(f"  Median Rating: {stats['median_rating']:.2f}/10")
        print(f"  Average Votes: {stats['avg_votes']:,.0f}")
        print(f"  Total Votes: {stats['total_votes']:,}")
        print(f"  Year Range: {stats['year_range']}")
        print(f"  Unique Genres: {stats['unique_genres']}")
        print(f"  Average Sentiment: {stats['avg_sentiment']:.3f}")
        
        self.results['statistics'] = stats
        print(f"\n✅ Statistics calculated\n")
        return stats
    
    def time_series_analysis(self):
        
        print("📈 TIME SERIES ANALYSIS\n")
        
        
        decade_counts = self.df['decade'].value_counts().sort_index()
        print(f"  Movies by Decade:")
        for decade, count in decade_counts.head(10).items():
            print(f"    {decade}: {count} movies")
        
        
        decade_ratings = self.df.groupby('decade')['rating'].mean().sort_index()
        
        
        recent_avg = self.df[self.df['is_recent']]['rating'].mean()
        old_avg = self.df[~self.df['is_recent']]['rating'].mean()
        
        print(f"\n  Rating Trends:")
        print(f"    Recent movies (last 10 years): {recent_avg:.2f}")
        print(f"    Older movies: {old_avg:.2f}")
        
        self.results['time_series'] = {
            'decade_counts': decade_counts,
            'decade_ratings': decade_ratings
        }
        
        print(f"\n✅ Time series analysis complete\n")
        return decade_counts, decade_ratings
    
    def clustering_analysis(self, n_clusters=5):
       
        print(f"🔬 K-MEANS CLUSTERING (k={n_clusters})\n")
        
       
        features = self.df[['rating', 'votes', 'popularity', 
                           'sentiment_polarity', 'description_length']].copy()
        features = features.fillna(features.mean())
        
       
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self.df['cluster'] = kmeans.fit_predict(features_scaled)
        
       
        silhouette = silhouette_score(features_scaled, self.df['cluster'])
        
        print(f"  Silhouette Score: {silhouette:.3f}")
        print(f"\n  Cluster Distribution:")
        for cluster_id in range(n_clusters):
            count = (self.df['cluster'] == cluster_id).sum()
            avg_rating = self.df[self.df['cluster'] == cluster_id]['rating'].mean()
            print(f"    Cluster {cluster_id}: {count} movies (avg rating: {avg_rating:.2f})")
        
        self.results['kmeans'] = kmeans
        self.results['silhouette'] = silhouette
        
        print(f"\n✅ Clustering complete\n")
        return kmeans
    
    def classification_model(self):
        """Random Forest to predict rating category"""
        print("🌳 CLASSIFICATION MODEL (Random Forest)\n")
        
        
        feature_cols = ['votes', 'popularity', 'sentiment_polarity', 
                       'description_length', 'year']
        X = self.df[feature_cols].fillna(0)
        y = self.df['rating_category']
        
       
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        
        clf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
        clf.fit(X_train, y_train)
        
        
        y_pred = clf.predict(X_test)
        accuracy = (y_pred == y_test).mean()
        
        print(f"  Model: Random Forest")
        print(f"  Features: {', '.join(feature_cols)}")
        print(f"  Training samples: {len(X_train)}")
        print(f"  Test samples: {len(X_test)}")
        print(f"  Accuracy: {accuracy:.2%}")
        
        print(f"\n  Classification Report:")
        print(classification_report(y_test, y_pred, zero_division=0))
        
       
        feature_importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': clf.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"  Feature Importance:")
        for _, row in feature_importance.iterrows():
            print(f"    {row['feature']}: {row['importance']:.3f}")
        
        self.results['classifier'] = clf
        self.results['accuracy'] = accuracy
        self.results['feature_importance'] = feature_importance
        
        print(f"\n✅ Classification complete\n")
        return clf
    
    def sentiment_analysis(self):
        """Analyze sentiment patterns"""
        print("💭 SENTIMENT ANALYSIS\n")
        
        sentiment_dist = self.df['sentiment_category'].value_counts()
        print(f"  Sentiment Distribution:")
        for category, count in sentiment_dist.items():
            pct = (count / len(self.df)) * 100
            print(f"    {category}: {count} ({pct:.1f}%)")
        
        
        correlation = self.df['sentiment_polarity'].corr(self.df['rating'])
        print(f"\n  Correlation (Sentiment vs Rating): {correlation:.3f}")
        
        
        genre_sentiment = self.df.groupby('primary_genre')['sentiment_polarity'].mean().sort_values(ascending=False)
        print(f"\n  Top 5 Most Positive Genres:")
        for genre, sentiment in genre_sentiment.head(5).items():
            print(f"    {genre}: {sentiment:.3f}")
        
        self.results['sentiment'] = {
            'distribution': sentiment_dist,
            'correlation': correlation,
            'genre_sentiment': genre_sentiment
        }
        
        print(f"\n✅ Sentiment analysis complete\n")
        return sentiment_dist




class MovieVisualizer:

    
    def __init__(self, df, output_dir='outputs'):
        self.df = df
        self.output_dir = output_dir
        import os
        os.makedirs(output_dir, exist_ok=True)
        
    def create_all_visualizations(self):
        """Generate all visualizations"""
        print("📈 DATA VISUALIZATION\n")
        
        sns.set_style('whitegrid')
        plt.rcParams['figure.figsize'] = (12, 6)
        
        self.plot_rating_distribution()
        self.plot_genre_analysis()
        self.plot_time_trends()
        self.plot_sentiment_analysis()
        self.plot_clustering()
        self.plot_correlation_heatmap()
        
        print(f"✅ All visualizations saved to '{self.output_dir}/'\n")
    
    def plot_rating_distribution(self):
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        axes[0].hist(self.df['rating'], bins=30, color='skyblue', edgecolor='black')
        axes[0].set_title('Movie Rating Distribution', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Rating (0-10)')
        axes[0].set_ylabel('Frequency')
        axes[0].axvline(self.df['rating'].mean(), color='red', linestyle='--', 
                       label=f'Mean: {self.df["rating"].mean():.2f}')
        axes[0].legend()
        axes[0].grid(alpha=0.3)
        
      
        rating_counts = self.df['rating_category'].value_counts()
        axes[1].bar(rating_counts.index, rating_counts.values, color='coral')
        axes[1].set_title('Movies by Rating Category', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Category')
        axes[1].set_ylabel('Number of Movies')
        axes[1].tick_params(axis='x', rotation=45)
        axes[1].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/rating_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("  ✅ Rating distribution saved")
    
    def plot_genre_analysis(self):
        """Genre analysis"""
        fig, axes = plt.subplots(2, 1, figsize=(12, 10))
        
        
        top_genres = self.df['primary_genre'].value_counts().head(10)
        axes[0].barh(top_genres.index, top_genres.values, color='lightgreen')
        axes[0].set_title('Top 10 Genres by Movie Count', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Number of Movies')
        axes[0].grid(axis='x', alpha=0.3)
        
        
        genre_ratings = self.df.groupby('primary_genre')['rating'].mean().sort_values(ascending=False).head(10)
        axes[1].barh(genre_ratings.index, genre_ratings.values, color='steelblue')
        axes[1].set_title('Top 10 Genres by Average Rating', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Average Rating')
        axes[1].grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/genre_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("  ✅ Genre analysis saved")
    
    def plot_time_trends(self):
       
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))
        
       
        year_counts = self.df[self.df['year'] > 1950].groupby('year').size()
        axes[0].plot(year_counts.index, year_counts.values, color='purple', linewidth=2)
        axes[0].fill_between(year_counts.index, year_counts.values, alpha=0.3, color='purple')
        axes[0].set_title('Movies Released Per Year (1950+)', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Year')
        axes[0].set_ylabel('Number of Movies')
        axes[0].grid(alpha=0.3)
        
       
        decade_ratings = self.df[self.df['decade'] != 'Unknown'].groupby('decade')['rating'].mean().sort_index()
        axes[1].bar(decade_ratings.index, decade_ratings.values, color='orange')
        axes[1].set_title('Average Rating by Decade', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Decade')
        axes[1].set_ylabel('Average Rating')
        axes[1].tick_params(axis='x', rotation=45)
        axes[1].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/time_trends.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("  ✅ Time trends saved")
    
    def plot_sentiment_analysis(self):
        """Sentiment visualization"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        
        scatter = axes[0].scatter(self.df['sentiment_polarity'], self.df['rating'],
                                 c=self.df['votes'], cmap='viridis', alpha=0.5, s=30)
        axes[0].set_title('Sentiment vs Rating', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Sentiment Polarity')
        axes[0].set_ylabel('Rating')
        axes[0].grid(alpha=0.3)
        plt.colorbar(scatter, ax=axes[0], label='Votes')
        
      
        sentiment_counts = self.df['sentiment_category'].value_counts()
        colors = ['#ff6b6b', '#95a5a6', '#51cf66']
        axes[1].pie(sentiment_counts.values, labels=sentiment_counts.index, 
                   autopct='%1.1f%%', colors=colors, startangle=90)
        axes[1].set_title('Sentiment Distribution', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/sentiment_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("  ✅ Sentiment analysis saved")
    
    def plot_clustering(self):
        """Cluster visualization with PCA"""
        if 'cluster' not in self.df.columns:
            return
        
       
        features = self.df[['rating', 'votes', 'popularity', 
                           'sentiment_polarity', 'description_length']].fillna(0)
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        pca = PCA(n_components=2)
        components = pca.fit_transform(features_scaled)
        
        plt.figure(figsize=(12, 8))
        scatter = plt.scatter(components[:, 0], components[:, 1],
                            c=self.df['cluster'], cmap='Set1', alpha=0.6, s=50)
        plt.colorbar(scatter, label='Cluster')
        plt.title('Movie Clusters (PCA Visualization)', fontsize=16, fontweight='bold')
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
        plt.grid(alpha=0.3)
        plt.savefig(f'{self.output_dir}/clustering.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("  ✅ Clustering visualization saved")
    
    def plot_correlation_heatmap(self):
        """Correlation heatmap"""
        numeric_cols = ['rating', 'votes', 'popularity', 'sentiment_polarity', 
                       'description_length', 'year']
        corr_matrix = self.df[numeric_cols].corr()
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                   center=0, square=True, linewidths=1)
        plt.title('Feature Correlation Heatmap', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/correlation_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("  ✅ Correlation heatmap saved")




def main():
  
    print("="*80)
    print("🎬 TMDB MOVIE DATA ANALYTICS PIPELINE")
    print("="*80)
    print(f"Domain: Movie Industry Analytics")
    print(f"Data Source: TMDB API (JSON)")
    print(f"Target: 3000+ movie records")
    print("="*80 + "\n")
    
   
    print("="*80)
    print("STEP 1: DATA ACQUISITION (API)")
    print("="*80)
    
    collector = TMDBDataCollector(API_KEY)
    movies_list = collector.fetch_movies(num_pages=180, endpoint='top_rated')
    
    
    df = pd.DataFrame(movies_list)
    print(f"✅ Created DataFrame: {len(df)} records, {len(df.columns)} columns\n")
    
  
    df.to_csv('outputs/raw_movie_data.csv', index=False)
    print(f"💾 Raw data saved: outputs/raw_movie_data.csv\n")
    
    
    print("="*80)
    print("STEP 2: DATA CLEANING & PREPROCESSING")
    print("="*80 + "\n")
    
    preprocessor = DataPreprocessor()
    df = preprocessor.clean_data(df)
    df = preprocessor.engineer_features(df)
    
   
    df.to_csv('outputs/cleaned_movie_data.csv', index=False)
    print(f"💾 Cleaned data saved: outputs/cleaned_movie_data.csv\n")
    
    
    print("="*80)
    print("STEP 3: ANALYTICS & MACHINE LEARNING")
    print("="*80 + "\n")
    
    analytics = MovieAnalytics(df)
    analytics.descriptive_statistics()
    analytics.time_series_analysis()
    analytics.clustering_analysis(n_clusters=5)
    analytics.classification_model()
    analytics.sentiment_analysis()
    
   
    print("="*80)
    print("STEP 4: DATA VISUALIZATION")
    print("="*80 + "\n")
    
    visualizer = MovieVisualizer(df)
    visualizer.create_all_visualizations()
    
    
    print("="*80)
    print("✅ PIPELINE COMPLETE!")
    print("="*80)
    print(f"\n📊 FINAL SUMMARY:")
    print(f"  • Total movies analyzed: {len(df):,}")
    print(f"  • Average rating: {df['rating'].mean():.2f}/10")
    print(f"  • Year range: {df['year'].min()} - {df['year'].max()}")
    print(f"  • Unique genres: {df['primary_genre'].nunique()}")
    print(f"  • Average sentiment: {df['sentiment_polarity'].mean():.3f}")
    print(f"  • ML accuracy: {analytics.results.get('accuracy', 0):.1%}")
    
    print(f"\n📁 OUTPUTS:")
    print(f"  • Raw data: outputs/raw_movie_data.csv")
    print(f"  • Cleaned data: outputs/cleaned_movie_data.csv")
    print(f"  • Visualizations: outputs/*.png (6 files)")
    
    print(f"\n🎯 KEY INSIGHTS:")
    print(f"  • Most common genre: {df['primary_genre'].mode()[0]}")
    print(f"  • Highest rated decade: {df.groupby('decade')['rating'].mean().idxmax()}")
    print(f"  • Sentiment-Rating correlation: {df['sentiment_polarity'].corr(df['rating']):.3f}")
    
    print("\n" + "="*80)
    print("🎉 Analysis complete! Check the outputs/ folder for results.")
    print("="*80 + "\n")
    
    return df


if __name__ == "__main__":
    
    df_movies = main()
    
    print("\n💡 TIP: You can now explore the data further:")
    print("   - Load: df = pd.read_csv('outputs/cleaned_movie_data.csv')")
    print("   - View visualizations in outputs/ folder")
    print("   - Run custom analysis on df_movies DataFrame")

