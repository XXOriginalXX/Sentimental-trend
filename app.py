import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler

from textblob import TextBlob
import re
from collections import Counter

plt.style.use('default')
sns.set_palette("husl")

df = pd.read_csv('test.csv')

print("Dataset Shape:", df.shape)
print("\nColumn Names:", df.columns.tolist())
print("\nFirst 5 rows:")
print(df.head())
print("\nData Types:")
print(df.dtypes)
print("\nMissing Values:")
print(df.isnull().sum())

def clean_text(text):
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def get_sentiment_label(text):
    if pd.isna(text) or text == "":
        return "Neutral"
    
    cleaned_text = clean_text(text)
    if len(cleaned_text) < 3:
        return "Neutral"
    
    blob = TextBlob(cleaned_text)
    polarity = blob.sentiment.polarity
    
    if polarity > 0.1:
        return "Positive"
    elif polarity < -0.1:
        return "Negative"
    else:
        return "Neutral"

message_column = None
for col in df.columns:
    if 'message' in col.lower() or 'text' in col.lower() or 'content' in col.lower():
        message_column = col
        break

if message_column is None:
    message_column = df.select_dtypes(include=['object']).columns[0]

print(f"\nUsing column '{message_column}' for sentiment analysis")

df['sentiment'] = df[message_column].apply(get_sentiment_label)

print("\nSentiment Distribution:")
print(df['sentiment'].value_counts())

date_column = None
for col in df.columns:
    if 'date' in col.lower() or 'time' in col.lower():
        date_column = col
        break

if date_column:
    df[date_column] = pd.to_datetime(df[date_column], errors='coerce')
    df = df.dropna(subset=[date_column])
    df['year_month'] = df[date_column].dt.to_period('M')
else:
    df['year_month'] = pd.Period('2024-01', freq='M')

employee_column = None
for col in df.columns:
    if 'employee' in col.lower() or 'name' in col.lower() or 'user' in col.lower():
        employee_column = col
        break

if employee_column is None:
    df['employee'] = 'Employee_' + (df.index % 50 + 1).astype(str)
    employee_column = 'employee'

print(f"\nUsing column '{employee_column}' for employee identification")

fig, axes = plt.subplots(2, 2, figsize=(15, 12))

sentiment_counts = df['sentiment'].value_counts()
axes[0, 0].pie(sentiment_counts.values, labels=sentiment_counts.index, autopct='%1.1f%%')
axes[0, 0].set_title('Overall Sentiment Distribution')

df['message_length'] = df[message_column].astype(str).str.len()
axes[0, 1].hist(df['message_length'], bins=30, alpha=0.7)
axes[0, 1].set_title('Message Length Distribution')
axes[0, 1].set_xlabel('Message Length')
axes[0, 1].set_ylabel('Frequency')

if date_column:
    monthly_sentiment = df.groupby(['year_month', 'sentiment']).size().unstack(fill_value=0)
    monthly_sentiment.plot(kind='bar', stacked=True, ax=axes[1, 0])
    axes[1, 0].set_title('Monthly Sentiment Trends')
    axes[1, 0].set_xlabel('Month')
    axes[1, 0].set_ylabel('Count')
    axes[1, 0].tick_params(axis='x', rotation=45)

top_employees = df[employee_column].value_counts().head(10)
axes[1, 1].bar(range(len(top_employees)), top_employees.values)
axes[1, 1].set_title('Top 10 Most Active Employees')
axes[1, 1].set_xlabel('Employee Rank')
axes[1, 1].set_ylabel('Message Count')

plt.tight_layout()
plt.savefig('eda_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

def calculate_sentiment_score(sentiment):
    if sentiment == 'Positive':
        return 1
    elif sentiment == 'Negative':
        return -1
    else:
        return 0

df['sentiment_score'] = df['sentiment'].apply(calculate_sentiment_score)

monthly_scores = df.groupby([employee_column, 'year_month'])['sentiment_score'].sum().reset_index()
monthly_scores.columns = ['employee', 'month', 'monthly_score']

print("\nMonthly Sentiment Scores (Sample):")
print(monthly_scores.head(10))

def get_top_employees_by_month(monthly_scores, score_type='positive', top_n=3):
    result = {}
    
    for month in monthly_scores['month'].unique():
        month_data = monthly_scores[monthly_scores['month'] == month].copy()
        
        if score_type == 'positive':
            month_data = month_data.sort_values(['monthly_score', 'employee'], 
                                              ascending=[False, True])
        else:
            month_data = month_data.sort_values(['monthly_score', 'employee'], 
                                              ascending=[True, True])
        
        result[str(month)] = month_data.head(top_n)[['employee', 'monthly_score']].to_dict('records')
    
    return result

top_positive = get_top_employees_by_month(monthly_scores, 'positive', 3)
top_negative = get_top_employees_by_month(monthly_scores, 'negative', 3)

print("\nTop 3 Positive Employees by Month:")
for month, employees in top_positive.items():
    print(f"\n{month}:")
    for i, emp in enumerate(employees, 1):
        print(f"  {i}. {emp['employee']}: {emp['monthly_score']}")

print("\nTop 3 Negative Employees by Month:")
for month, employees in top_negative.items():
    print(f"\n{month}:")
    for i, emp in enumerate(employees, 1):
        print(f"  {i}. {emp['employee']}: {emp['monthly_score']}")

if date_column:
    df_sorted = df.sort_values([employee_column, date_column]).reset_index(drop=True)
    
    flight_risk_employees = set()
    
    for employee in df_sorted[employee_column].unique():
        emp_data = df_sorted[df_sorted[employee_column] == employee].copy()
        emp_data = emp_data.reset_index(drop=True)
        
        for i in range(len(emp_data)):
            current_date = emp_data.loc[i, date_column]
            start_date = current_date - timedelta(days=30)
            
            window_data = emp_data[
                (emp_data[date_column] >= start_date) & 
                (emp_data[date_column] <= current_date)
            ]
            
            negative_count = (window_data['sentiment'] == 'Negative').sum()
            
            if negative_count >= 4:
                flight_risk_employees.add(employee)
                break
else:
    negative_counts = df[df['sentiment'] == 'Negative'].groupby(employee_column).size()
    flight_risk_employees = set(negative_counts[negative_counts >= 4].index.tolist())

flight_risk_list = sorted(list(flight_risk_employees))

print(f"\nFlight Risk Employees ({len(flight_risk_list)} total):")
for emp in flight_risk_list:
    print(f"  - {emp}")

feature_data = df.groupby([employee_column, 'year_month']).agg({
    'sentiment_score': 'sum',
    'message_length': ['mean', 'sum', 'count'],
    message_column: lambda x: sum(len(str(msg).split()) for msg in x)
}).reset_index()

feature_data.columns = ['employee', 'month', 'monthly_score', 'avg_message_length', 
                       'total_message_length', 'message_frequency', 'total_word_count']

feature_data['avg_words_per_message'] = feature_data['total_word_count'] / feature_data['message_frequency']

features = ['avg_message_length', 'total_message_length', 'message_frequency', 
           'total_word_count', 'avg_words_per_message']
target = 'monthly_score'

X = feature_data[features].fillna(0)
y = feature_data[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LinearRegression()
model.fit(X_train_scaled, y_train)

y_pred_train = model.predict(X_train_scaled)
y_pred_test = model.predict(X_test_scaled)

train_mse = mean_squared_error(y_train, y_pred_train)
test_mse = mean_squared_error(y_test, y_pred_test)
train_r2 = r2_score(y_train, y_pred_train)
test_r2 = r2_score(y_test, y_pred_test)
train_mae = mean_absolute_error(y_train, y_pred_train)
test_mae = mean_absolute_error(y_test, y_pred_test)

print("\nLinear Regression Model Performance:")
print(f"Training R²: {train_r2:.4f}")
print(f"Testing R²: {test_r2:.4f}")
print(f"Training MSE: {train_mse:.4f}")
print(f"Testing MSE: {test_mse:.4f}")
print(f"Training MAE: {train_mae:.4f}")
print(f"Testing MAE: {test_mae:.4f}")

feature_importance = pd.DataFrame({
    'feature': features,
    'coefficient': model.coef_,
    'abs_coefficient': np.abs(model.coef_)
}).sort_values('abs_coefficient', ascending=False)

print("\nFeature Importance (Coefficients):")
print(feature_importance)

fig, axes = plt.subplots(2, 2, figsize=(15, 12))

monthly_scores.boxplot(column='monthly_score', ax=axes[0, 0])
axes[0, 0].set_title('Distribution of Monthly Sentiment Scores')
axes[0, 0].set_ylabel('Monthly Score')

feature_importance.plot(x='feature', y='coefficient', kind='bar', ax=axes[0, 1])
axes[0, 1].set_title('Feature Importance in Linear Regression')
axes[0, 1].set_ylabel('Coefficient Value')
axes[0, 1].tick_params(axis='x', rotation=45)

axes[1, 0].scatter(y_test, y_pred_test, alpha=0.6)
axes[1, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
axes[1, 0].set_xlabel('Actual Scores')
axes[1, 0].set_ylabel('Predicted Scores')
axes[1, 0].set_title(f'Actual vs Predicted (R² = {test_r2:.3f})')

if len(flight_risk_list) > 0:
    axes[1, 1].bar(['At Risk', 'Safe'], 
                   [len(flight_risk_list), len(df[employee_column].unique()) - len(flight_risk_list)])
    axes[1, 1].set_title('Flight Risk Analysis')
    axes[1, 1].set_ylabel('Number of Employees')
else:
    axes[1, 1].text(0.5, 0.5, 'No Flight Risk\nEmployees Detected', 
                    ha='center', va='center', transform=axes[1, 1].transAxes, fontsize=16)
    axes[1, 1].set_title('Flight Risk Analysis')

plt.tight_layout()
plt.savefig('analysis_results.png', dpi=300, bbox_inches='tight')
plt.show()

summary_results = {
    'dataset_info': {
        'total_records': len(df),
        'total_employees': df[employee_column].nunique(),
        'date_range': f"{df[date_column].min()} to {df[date_column].max()}" if date_column else "Date column not found"
    },
    'sentiment_distribution': df['sentiment'].value_counts().to_dict(),
    'top_positive_employees': top_positive,
    'top_negative_employees': top_negative,
    'flight_risk_employees': flight_risk_list,
    'model_performance': {
        'r2_score': test_r2,
        'mse': test_mse,
        'mae': test_mae
    },
    'feature_importance': feature_importance.to_dict('records')
}

print("\n" + "="*50)
print("FINAL SUMMARY")
print("="*50)
print(f"Total Records Analyzed: {summary_results['dataset_info']['total_records']}")
print(f"Total Employees: {summary_results['dataset_info']['total_employees']}")
print(f"Flight Risk Employees: {len(flight_risk_list)}")
print(f"Model R² Score: {test_r2:.4f}")

print("\nSentiment Distribution:")
for sentiment, count in summary_results['sentiment_distribution'].items():
    print(f"  {sentiment}: {count} ({count/len(df)*100:.1f}%)")

if flight_risk_list:
    print(f"\nFlight Risk Alert: {len(flight_risk_list)} employees need attention")
else:
    print("\nNo immediate flight risk detected")

monthly_scores.to_csv('monthly_sentiment_scores.csv', index=False)
feature_data.to_csv('feature_analysis.csv', index=False)

with open('flight_risk_employees.txt', 'w') as f:
    f.write("Flight Risk Employees:\n")
    for emp in flight_risk_list:
        f.write(f"- {emp}\n")

print(f"\nFiles saved:")
print("- monthly_sentiment_scores.csv")
print("- feature_analysis.csv") 
print("- flight_risk_employees.txt")
print("- eda_analysis.png")
print("- analysis_results.png")
