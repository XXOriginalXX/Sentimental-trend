# Employee Sentiment Analysis Project

## Project Overview

This project analyzes employee sentiment from message data to identify engagement patterns, calculate sentiment scores, rank employees, and predict flight risk. The analysis provides actionable insights for HR management and employee retention strategies.

## Dataset

- **Source**: test.csv (unlabeled employee messages)
- **Processing**: Automatic sentiment labeling using TextBlob NLP library
- **Features**: Messages, timestamps, employee identifiers

## Key Findings

### Sentiment Distribution
- **Positive Messages**: [Percentage from analysis]
- **Negative Messages**: [Percentage from analysis]  
- **Neutral Messages**: [Percentage from analysis]

### Top Performing Employees (Latest Month)

#### Most Positive Employees
1. [Employee Name] - Score: [Score]
2. [Employee Name] - Score: [Score]
3. [Employee Name] - Score: [Score]

#### Most Negative Employees
1. [Employee Name] - Score: [Score]
2. [Employee Name] - Score: [Score]
3. [Employee Name] - Score: [Score]

### Flight Risk Analysis

**Employees Flagged as Flight Risk**: [Number] employees

Flight risk criteria: 4+ negative messages within any 30-day rolling period

**Flight Risk Employees**:
- [List of employee names]

### Predictive Model Performance

- **Model Type**: Linear Regression
- **R² Score**: [Score from analysis]
- **Mean Absolute Error**: [MAE value]
- **Key Predictors**: Message frequency, message length, word count

## Methodology

### 1. Sentiment Labeling
- **Technique**: TextBlob polarity analysis
- **Categories**: Positive (>0.1), Negative (<-0.1), Neutral (-0.1 to 0.1)
- **Preprocessing**: Text cleaning, lowercase conversion, special character removal

### 2. Scoring System
- **Positive Message**: +1 point
- **Negative Message**: -1 point
- **Neutral Message**: 0 points
- **Aggregation**: Monthly totals per employee

### 3. Flight Risk Detection
- **Method**: Rolling 30-day window analysis
- **Threshold**: 4+ negative messages in any 30-day period
- **Purpose**: Early identification of disengaged employees

### 4. Predictive Modeling
- **Features**: Average message length, total message length, message frequency, total word count, average words per message
- **Target**: Monthly sentiment score
- **Validation**: Train/test split with standardized features

## Technical Implementation

### Libraries Used
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computations
- **matplotlib/seaborn**: Data visualization
- **scikit-learn**: Machine learning models and metrics
- **textblob**: Natural language processing for sentiment analysis
- **re**: Text preprocessing with regular expressions

### Code Structure
```
employee_sentiment_analysis.py
├── Data Loading & Exploration
├── Sentiment Analysis
├── Exploratory Data Analysis
├── Monthly Score Calculation
├── Employee Ranking
├── Flight Risk Identification
├── Predictive Modeling
└── Results Visualization
```

## Output Files

### Generated Files
- `monthly_sentiment_scores.csv` - Employee scores by month
- `feature_analysis.csv` - Features used for predictive modeling
- `flight_risk_employees.txt` - List of at-risk employees
- `eda_analysis.png` - Exploratory data analysis visualizations
- `analysis_results.png` - Model performance and ranking visualizations

### Visualizations
1. **Sentiment Distribution Pie Chart** - Overall message sentiment breakdown
2. **Message Length Histogram** - Distribution of message lengths
3. **Monthly Sentiment Trends** - Time series of sentiment patterns
4. **Top Active Employees** - Most frequent message senders
5. **Monthly Score Distribution** - Box plot of sentiment scores
6. **Feature Importance** - Linear regression coefficients
7. **Model Performance** - Actual vs predicted scatter plot
8. **Flight Risk Summary** - At-risk vs safe employee counts

## Key Insights

### Employee Engagement Patterns
- [Insight about overall sentiment trends]
- [Insight about seasonal patterns if any]
- [Insight about message frequency patterns]

### Risk Factors
- [Key predictive factors for negative sentiment]
- [Patterns observed in flight risk employees]
- [Correlation between message characteristics and sentiment]

### Recommendations

#### Immediate Actions
1. **Review Flight Risk Employees**: Schedule one-on-one meetings with flagged employees
2. **Monitor Negative Trend Employees**: Provide additional support to employees with declining scores
3. **Recognize Top Performers**: Acknowledge employees with consistently positive sentiment

#### Long-term Strategies
1. **Implement Early Warning System**: Use the predictive model for proactive intervention
2. **Improve Communication Channels**: Address factors contributing to negative sentiment
3. **Regular Sentiment Monitoring**: Establish monthly sentiment tracking processes

## Model Limitations

- **Sentiment Analysis**: TextBlob may not capture context-specific sentiment nuances
- **Data Scope**: Analysis limited to available message data
- **Temporal Coverage**: Results depend on data collection period
- **Feature Engineering**: Additional features could improve predictive accuracy

## Usage Instructions

### Prerequisites
```bash
pip install pandas numpy matplotlib seaborn scikit-learn textblob
```

### Running the Analysis
1. Place `test.csv` in the project directory
2. Run the Python script
3. Review generated outputs and visualizations
4. Examine CSV files for detailed results

### Customization
- Adjust sentiment thresholds in `get_sentiment_label()` function
- Modify flight risk criteria (currently 4 negative messages in 30 days)
- Add additional features for predictive modeling
- Change visualization styles and colors


---

**Note**: This analysis is based on automated sentiment detection and should be supplemented with human judgment for critical decisions regarding employee management.
