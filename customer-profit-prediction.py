# Calyx Customer Profit Prediction System
# Advanced ML-powered customer lifetime value prediction with Monte Carlo simulation

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import scipy.stats as stats

warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

print("📊 Calyx Customer Profit Prediction System")
print("=" * 50)

# =============================================================================
# 1. DATA GENERATION
# =============================================================================

def generate_customers(n_customers=20):
    """Generate realistic customer data"""
    
    company_names = [
        'TechFlow Solutions', 'DataSync Corp', 'CloudPrime Inc', 'InnovateSys',
        'DigitalEdge Ltd', 'SmartCore Technologies', 'NextGen Analytics',
        'CyberVault Solutions', 'StreamLine Enterprises', 'FlexiCloud Systems',
        'ProActive IT', 'Quantum Networks', 'SynergyTech', 'OptimalData Corp',
        'FusionSoft Solutions', 'EliteCloud Services', 'RapidScale Inc',
        'DynamicSys Solutions', 'TechnovaPrime', 'CloudMaster Technologies'
    ]
    
    industries = ['Healthcare', 'Finance', 'Retail', 'Manufacturing', 'Education', 'Government']
    sizes = ['Small', 'Medium', 'Large']
    
    customers_data = []
    
    for i in range(n_customers):
        size = np.random.choice(sizes)
        size_multiplier = {'Small': 1, 'Medium': 2, 'Large': 3}[size]
        
        # Generate correlated customer characteristics
        credit_score = np.random.normal(550, 150)
        credit_score = max(300, min(850, credit_score))  # Bound between 300-850
        
        # Higher credit scores tend to have longer relationships
        relationship_length = max(1, int(np.random.exponential(12) + (credit_score - 400) / 50))
        
        # Satisfaction correlated with relationship length and credit score
        base_satisfaction = 2.5 + (credit_score - 400) / 200 + relationship_length / 60
        satisfaction_score = max(1, min(5, np.random.normal(base_satisfaction, 0.5)))
        
        # Monthly spend correlated with size and satisfaction
        base_spend = 2000 * size_multiplier * (1 + satisfaction_score / 10)
        avg_monthly_spend = max(500, np.random.normal(base_spend, base_spend * 0.3))
        
        # Support tickets inversely correlated with satisfaction
        support_tickets = max(0, int(np.random.poisson(max(1, 40 - satisfaction_score * 8))))
        
        customers_data.append({
            'customer_id': i + 1,
            'company_name': company_names[i] if i < len(company_names) else f'Company_{i+1}',
            'industry': np.random.choice(industries),
            'size': size,
            'credit_score': credit_score,
            'relationship_length_months': relationship_length,
            'avg_monthly_spend': avg_monthly_spend,
            'support_tickets': support_tickets,
            'satisfaction_score': satisfaction_score
        })
    
    return pd.DataFrame(customers_data)

def generate_contracts(customers_df):
    """Generate historical contract data"""
    
    contract_types = ['Cloud Infrastructure', 'Data Analytics', 'Cybersecurity', 'IT Support', 'Software Development']
    statuses = ['New', 'Open', 'Closed']
    
    contracts_data = []
    contract_id = 1
    
    for _, customer in customers_df.iterrows():
        # Number of contracts based on company size and relationship length
        size_factor = {'Small': 1, 'Medium': 2, 'Large': 3}[customer['size']]
        num_contracts = max(1, np.random.poisson(1.5 * size_factor))
        
        for _ in range(num_contracts):
            # Contract timing spread over relationship period
            start_offset_days = np.random.randint(0, max(30, customer['relationship_length_months'] * 20))
            start_date = datetime.now() - timedelta(days=start_offset_days)
            
            # Contract duration (6-36 months, influenced by customer size)
            duration_months = np.random.choice([6, 12, 18, 24, 36], 
                                             p=[0.1, 0.3, 0.3, 0.2, 0.1])
            if customer['size'] == 'Large':
                duration_months = max(12, duration_months)
            
            end_date = start_date + timedelta(days=duration_months * 30)
            
            # Monthly value based on customer characteristics
            base_value = customer['avg_monthly_spend'] * np.random.uniform(0.3, 1.2)
            monthly_value = max(500, base_value * (1 + customer['satisfaction_score'] / 10))
            
            total_value = monthly_value * duration_months
            
            # Profit margin varies by contract type and customer satisfaction
            contract_type = np.random.choice(contract_types)
            type_margins = {
                'Cloud Infrastructure': 0.25,
                'Data Analytics': 0.35,
                'Cybersecurity': 0.40,
                'IT Support': 0.20,
                'Software Development': 0.45
            }
            
            base_margin = type_margins[contract_type]
            satisfaction_bonus = (customer['satisfaction_score'] - 2.5) * 0.05
            profit_margin = base_margin + satisfaction_bonus + np.random.normal(0, 0.05)
            profit_margin = max(0.05, min(0.60, profit_margin))
            
            total_profit = total_value * profit_margin
            
            # Status based on end date
            if datetime.now() > end_date:
                status = 'Closed'
            elif np.random.random() > 0.7:
                status = 'New'
            else:
                status = 'Open'
            
            contracts_data.append({
                'contract_id': contract_id,
                'customer_id': customer['customer_id'],
                'contract_type': contract_type,
                'status': status,
                'start_date': start_date,
                'end_date': end_date,
                'duration_months': duration_months,
                'monthly_value': monthly_value,
                'total_value': total_value,
                'total_profit': total_profit,
                'profit_margin': profit_margin
            })
            
            contract_id += 1
    
    return pd.DataFrame(contracts_data)

def generate_opportunities(customers_df):
    """Generate current sales opportunities"""
    
    contract_types = ['Cloud Infrastructure', 'Data Analytics', 'Cybersecurity', 'IT Support', 'Software Development']
    stages = ['Discovery', 'Proposal', 'Negotiation', 'Final Review']
    
    opportunities_data = []
    
    for _, customer in customers_df.iterrows():
        # Probability of having opportunities based on satisfaction and relationship
        opp_probability = 0.3 + (customer['satisfaction_score'] - 2.5) * 0.2
        opp_probability += min(0.3, customer['relationship_length_months'] / 60)
        
        if np.random.random() < opp_probability:
            num_opportunities = np.random.poisson(1.5) + 1
            
            for i in range(num_opportunities):
                # Estimated value based on customer profile
                size_multiplier = {'Small': 1, 'Medium': 2, 'Large': 3}[customer['size']]
                base_value = customer['avg_monthly_spend'] * np.random.uniform(8, 24)  # 8-24 months of spending
                estimated_value = base_value * size_multiplier * np.random.uniform(0.7, 1.5)
                
                # Conversion probability based on customer characteristics
                base_probability = 0.4
                if customer['satisfaction_score'] > 4:
                    base_probability += 0.2
                if customer['relationship_length_months'] > 24:
                    base_probability += 0.15
                if customer['credit_score'] > 650:
                    base_probability += 0.1
                
                # Stage influences probability
                stage = np.random.choice(stages)
                stage_multipliers = {
                    'Discovery': 0.8,
                    'Proposal': 0.9,
                    'Negotiation': 1.1,
                    'Final Review': 1.2
                }
                
                conversion_probability = min(0.9, base_probability * stage_multipliers[stage] * np.random.uniform(0.8, 1.2))
                
                opportunities_data.append({
                    'opportunity_id': f'OPP-{customer["customer_id"]}-{i+1}',
                    'customer_id': customer['customer_id'],
                    'contract_type': np.random.choice(contract_types),
                    'stage': stage,
                    'estimated_value': estimated_value,
                    'conversion_probability': conversion_probability
                })
    
    return pd.DataFrame(opportunities_data)

# Generate the datasets
print("🔄 Generating customer data...")
customers_df = generate_customers(25)  # Increased to 25 customers

print("🔄 Generating contract data...")
contracts_df = generate_contracts(customers_df)

print("🔄 Generating opportunities data...")
opportunities_df = generate_opportunities(customers_df)

print(f"✅ Data generation complete!")
print(f"   📊 Customers: {len(customers_df)}")
print(f"   📋 Contracts: {len(contracts_df)}")
print(f"   🎯 Opportunities: {len(opportunities_df)}")

# =============================================================================
# 2. EXPLORATORY DATA ANALYSIS
# =============================================================================

def analyze_customer_data():
    """Comprehensive analysis of customer data"""
    
    print("\n📈 CUSTOMER DATA ANALYSIS")
    print("=" * 30)
    
    # Basic statistics
    print("\n🔍 Customer Overview:")
    print(f"Total customers: {len(customers_df)}")
    print(f"Industries: {customers_df['industry'].nunique()}")
    print(f"Size distribution: {customers_df['size'].value_counts().to_dict()}")
    
    # Customer metrics summary
    print(f"\n📊 Customer Metrics:")
    metrics = ['credit_score', 'relationship_length_months', 'avg_monthly_spend', 'satisfaction_score']
    print(customers_df[metrics].describe().round(2))
    
    # Contract analysis
    print(f"\n📋 Contract Analysis:")
    print(f"Total contracts: {len(contracts_df)}")
    print(f"Status distribution: {contracts_df['status'].value_counts().to_dict()}")
    print(f"Total revenue: ${contracts_df['total_value'].sum():,.0f}")
    print(f"Total profit: ${contracts_df['total_profit'].sum():,.0f}")
    print(f"Average profit margin: {contracts_df['profit_margin'].mean():.1%}")
    
    # Opportunities analysis
    print(f"\n🎯 Opportunities Analysis:")
    print(f"Total opportunities: {len(opportunities_df)}")
    print(f"Total pipeline value: ${opportunities_df['estimated_value'].sum():,.0f}")
    print(f"Average conversion probability: {opportunities_df['conversion_probability'].mean():.1%}")

def create_visualizations():
    """Create comprehensive visualizations"""
    
    # Set up the plotting style
    plt.style.use('seaborn-v0_8')
    fig = plt.figure(figsize=(20, 16))
    
    # 1. Customer size and industry distribution
    plt.subplot(3, 4, 1)
    customers_df['size'].value_counts().plot(kind='pie', autopct='%1.1f%%')
    plt.title('Customer Size Distribution')
    plt.ylabel('')
    
    plt.subplot(3, 4, 2)
    customers_df['industry'].value_counts().plot(kind='bar')
    plt.title('Customers by Industry')
    plt.xticks(rotation=45)
    
    # 2. Customer satisfaction vs spending
    plt.subplot(3, 4, 3)
    plt.scatter(customers_df['satisfaction_score'], customers_df['avg_monthly_spend'], 
                c=customers_df['credit_score'], cmap='viridis', alpha=0.7)
    plt.xlabel('Satisfaction Score')
    plt.ylabel('Avg Monthly Spend ($)')
    plt.title('Satisfaction vs Spending (Color: Credit Score)')
    plt.colorbar(label='Credit Score')
    
    # 3. Contract value distribution
    plt.subplot(3, 4, 4)
    contracts_df['total_value'].hist(bins=20, alpha=0.7)
    plt.xlabel('Contract Value ($)')
    plt.ylabel('Frequency')
    plt.title('Contract Value Distribution')
    
    # 4. Profit margin by contract type
    plt.subplot(3, 4, 5)
    contract_profit = contracts_df.groupby('contract_type')['profit_margin'].mean().sort_values(ascending=False)
    contract_profit.plot(kind='bar')
    plt.title('Average Profit Margin by Contract Type')
    plt.ylabel('Profit Margin')
    plt.xticks(rotation=45)
    
    # 5. Relationship length vs total customer value
    plt.subplot(3, 4, 6)
    customer_totals = contracts_df.groupby('customer_id')['total_value'].sum()
    customer_data = customers_df.set_index('customer_id')
    plt.scatter(customer_data['relationship_length_months'], customer_totals, alpha=0.7)
    plt.xlabel('Relationship Length (months)')
    plt.ylabel('Total Contract Value ($)')
    plt.title('Relationship Length vs Customer Value')
    
    # 6. Opportunities by stage and conversion probability
    plt.subplot(3, 4, 7)
    stage_conversion = opportunities_df.groupby('stage')['conversion_probability'].mean()
    stage_conversion.plot(kind='bar')
    plt.title('Average Conversion Rate by Stage')
    plt.ylabel('Conversion Probability')
    plt.xticks(rotation=45)
    
    # 7. Monthly contract values over time
    plt.subplot(3, 4, 8)
    contracts_df['start_month'] = contracts_df['start_date'].dt.to_period('M')
    monthly_values = contracts_df.groupby('start_month')['total_value'].sum()
    monthly_values.plot()
    plt.title('Contract Values Over Time')
    plt.ylabel('Total Contract Value ($)')
    plt.xticks(rotation=45)
    
    # 8. Customer profitability heatmap
    plt.subplot(3, 4, 9)
    profit_by_size_industry = contracts_df.merge(customers_df, on='customer_id').groupby(['size', 'industry'])['total_profit'].sum().unstack(fill_value=0)
    sns.heatmap(profit_by_size_industry, annot=True, fmt='.0f', cmap='YlOrRd')
    plt.title('Profit by Size and Industry')
    
    # 9. Support tickets vs satisfaction
    plt.subplot(3, 4, 10)
    plt.scatter(customers_df['support_tickets'], customers_df['satisfaction_score'], alpha=0.7)
    plt.xlabel('Support Tickets')
    plt.ylabel('Satisfaction Score')
    plt.title('Support Load vs Satisfaction')
    
    # 10. Opportunity value distribution
    plt.subplot(3, 4, 11)
    opportunities_df['estimated_value'].hist(bins=15, alpha=0.7)
    plt.xlabel('Opportunity Value ($)')
    plt.ylabel('Frequency')
    plt.title('Opportunity Value Distribution')
    
    # 11. Credit score distribution
    plt.subplot(3, 4, 12)
    customers_df['credit_score'].hist(bins=20, alpha=0.7)
    plt.xlabel('Credit Score')
    plt.ylabel('Frequency')
    plt.title('Credit Score Distribution')
    
    plt.tight_layout()
    plt.show()

# Run analysis and create visualizations
analyze_customer_data()
create_visualizations()

# =============================================================================
# 3. MONTE CARLO SIMULATION
# =============================================================================

def monte_carlo_simulation(opportunities_df, n_simulations=10000):
    """
    Run Monte Carlo simulation to estimate opportunity outcomes
    """
    print("\n🎲 MONTE CARLO SIMULATION")
    print("=" * 30)
    
    results = []
    
    for _, opportunity in opportunities_df.iterrows():
        # Run simulations for each opportunity
        simulation_results = []
        
        for _ in range(n_simulations):
            # Market volatility factor (90%-110% of expected value)
            market_factor = np.random.uniform(0.9, 1.1)
            
            # Adjusted probability with market conditions
            adjusted_probability = opportunity['conversion_probability'] * market_factor
            adjusted_probability = min(0.95, adjusted_probability)  # Cap at 95%
            
            # Determine if opportunity converts
            converted = np.random.random() < adjusted_probability
            
            # Calculate value if converted
            if converted:
                # Add some randomness to the final contract value
                value_variance = np.random.normal(1.0, 0.15)  # ±15% variance
                final_value = opportunity['estimated_value'] * market_factor * value_variance
                final_value = max(0, final_value)  # Ensure non-negative
            else:
                final_value = 0
            
            simulation_results.append(final_value)
        
        # Calculate statistics
        simulation_results = np.array(simulation_results)
        conversion_rate = np.sum(simulation_results > 0) / n_simulations
        expected_value = np.mean(simulation_results)
        
        # Confidence intervals
        ci_lower = np.percentile(simulation_results, 2.5)
        ci_upper = np.percentile(simulation_results, 97.5)
        
        # Value at Risk (10% worst case)
        var_10 = np.percentile(simulation_results, 10)
        
        results.append({
            'opportunity_id': opportunity['opportunity_id'],
            'customer_id': opportunity['customer_id'],
            'original_probability': opportunity['conversion_probability'],
            'original_value': opportunity['estimated_value'],
            'simulated_conversion_rate': conversion_rate,
            'expected_value': expected_value,
            'confidence_interval_lower': ci_lower,
            'confidence_interval_upper': ci_upper,
            'value_at_risk_10': var_10,
            'volatility': np.std(simulation_results)
        })
    
    simulation_df = pd.DataFrame(results)
    
    # Summary statistics
    total_original_pipeline = opportunities_df['estimated_value'].sum()
    total_expected_value = simulation_df['expected_value'].sum()
    weighted_conversion_rate = simulation_df['expected_value'].sum() / opportunities_df['estimated_value'].sum()
    
    print(f"📊 Simulation Results:")
    print(f"   Total original pipeline: ${total_original_pipeline:,.0f}")
    print(f"   Total expected value: ${total_expected_value:,.0f}")
    print(f"   Expected conversion rate: {weighted_conversion_rate:.1%}")
    print(f"   Pipeline risk adjustment: {(total_expected_value/total_original_pipeline - 1):.1%}")
    
    return simulation_df

def plot_simulation_results(simulation_df):
    """Visualize Monte Carlo simulation results"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. Original vs Expected Value
    axes[0, 0].scatter(simulation_df['original_value'], simulation_df['expected_value'], alpha=0.7)
    axes[0, 0].plot([0, simulation_df['original_value'].max()], [0, simulation_df['original_value'].max()], 'r--', alpha=0.5)
    axes[0, 0].set_xlabel('Original Estimated Value ($)')
    axes[0, 0].set_ylabel('Expected Value ($)')
    axes[0, 0].set_title('Original vs Simulated Expected Value')
    
    # 2. Conversion Rate Distribution
    axes[0, 1].hist(simulation_df['simulated_conversion_rate'], bins=20, alpha=0.7)
    axes[0, 1].axvline(simulation_df['simulated_conversion_rate'].mean(), color='red', linestyle='--', label='Mean')
    axes[0, 1].set_xlabel('Simulated Conversion Rate')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Distribution of Conversion Rates')
    axes[0, 1].legend()
    
    # 3. Value at Risk Analysis
    axes[1, 0].scatter(simulation_df['expected_value'], simulation_df['value_at_risk_10'], alpha=0.7)
    axes[1, 0].set_xlabel('Expected Value ($)')
    axes[1, 0].set_ylabel('Value at Risk (10th percentile) ($)')
    axes[1, 0].set_title('Expected Value vs Value at Risk')
    
    # 4. Volatility vs Expected Value
    axes[1, 1].scatter(simulation_df['expected_value'], simulation_df['volatility'], alpha=0.7)
    axes[1, 1].set_xlabel('Expected Value ($)')
    axes[1, 1].set_ylabel('Volatility (Std Dev) ($)')
    axes[1, 1].set_title('Expected Value vs Volatility')
    
    plt.tight_layout()
    plt.show()

# Run Monte Carlo simulation
simulation_results = monte_carlo_simulation(opportunities_df)
plot_simulation_results(simulation_results)

# =============================================================================
# 4. FEATURE ENGINEERING
# =============================================================================

def create_ml_features(customers_df, contracts_df, opportunities_df, simulation_results):
    """Create comprehensive features for machine learning"""
    
    print("\n🔧 FEATURE ENGINEERING")
    print("=" * 25)
    
    # Customer-level aggregations
    customer_features = customers_df.copy()
    
    # Contract history features
    contract_agg = contracts_df.groupby('customer_id').agg({
        'total_value': ['sum', 'mean', 'count'],
        'total_profit': ['sum', 'mean'],
        'profit_margin': ['mean', 'std'],
        'duration_months': ['mean', 'max'],
        'monthly_value': ['mean', 'max']
    }).round(2)
    
    # Flatten column names
    contract_agg.columns = ['_'.join(col).strip() for col in contract_agg.columns]
    contract_agg = contract_agg.reset_index()
    
    # Contract diversity (number of different contract types)
    contract_diversity = contracts_df.groupby('customer_id')['contract_type'].nunique().reset_index()
    contract_diversity.columns = ['customer_id', 'contract_type_diversity']
    
    # Recent activity (contracts in last 12 months)
    recent_date = datetime.now() - timedelta(days=365)
    recent_contracts = contracts_df[contracts_df['start_date'] >= recent_date]
    recent_activity = recent_contracts.groupby('customer_id').size().reset_index()
    recent_activity.columns = ['customer_id', 'recent_contracts_12m']
    
    # Active contracts
    active_contracts = contracts_df[contracts_df['status'].isin(['Open', 'New'])]
    active_count = active_contracts.groupby('customer_id').size().reset_index()
    active_count.columns = ['customer_id', 'active_contracts']
    
    # Opportunity features
    if len(opportunities_df) > 0:
        opp_agg = opportunities_df.groupby('customer_id').agg({
            'estimated_value': ['sum', 'count', 'mean'],
            'conversion_probability': ['mean', 'max']
        }).round(3)
        opp_agg.columns = ['_'.join(col).strip() for col in opp_agg.columns]
        opp_agg = opp_agg.reset_index()
        
        # Simulation results aggregation
        sim_agg = simulation_results.groupby('customer_id').agg({
            'expected_value': 'sum',
            'simulated_conversion_rate': 'mean',
            'volatility': 'mean'
        }).round(2)
        sim_agg.columns = ['sim_' + col for col in sim_agg.columns]
        sim_agg = sim_agg.reset_index()
    else:
        opp_agg = pd.DataFrame({'customer_id': customer_features['customer_id']})
        sim_agg = pd.DataFrame({'customer_id': customer_features['customer_id']})
    
    # Merge all features
    features_df = customer_features.copy()
    features_df = features_df.merge(contract_agg, on='customer_id', how='left')
    features_df = features_df.merge(contract_diversity, on='customer_id', how='left')
    features_df = features_df.merge(recent_activity, on='customer_id', how='left')
    features_df = features_df.merge(active_count, on='customer_id', how='left')
    features_df = features_df.merge(opp_agg, on='customer_id', how='left')
    features_df = features_df.merge(sim_agg, on='customer_id', how='left')
    
    # Fill missing values
    features_df = features_df.fillna(0)
    
    # Create derived features
    # Customer value density (total value per month of relationship)
    features_df['value_per_relationship_month'] = features_df['total_value_sum'] / features_df['relationship_length_months']
    
    # Support efficiency (spending per support ticket)
    features_df['spend_per_support_ticket'] = features_df['avg_monthly_spend'] / (features_df['support_tickets'] + 1)
    
    # Customer stickiness (contract diversity * relationship length)
    features_df['customer_stickiness'] = features_df['contract_type_diversity'] * features_df['relationship_length_months']
    
    # Risk score (composite)
    features_df['risk_score'] = (
        (features_df['credit_score'] < 500).astype(int) * 0.3 +
        (features_df['support_tickets'] > 30).astype(int) * 0.2 +
        (features_df['satisfaction_score'] < 3).astype(int) * 0.3 +
        (features_df['relationship_length_months'] < 6).astype(int) * 0.2
    )
    
    # Size encoding
    size_encoding = {'Small': 1, 'Medium': 2, 'Large': 3}
    features_df['size_encoded'] = features_df['size'].map(size_encoding)
    
    # Industry encoding (high-value industries)
    high_value_industries = ['Finance', 'Healthcare', 'Government']
    features_df['high_value_industry'] = features_df['industry'].isin(high_value_industries).astype(int)
    
    print(f"✅ Feature engineering complete!")
    print(f"   📊 Total features: {len(features_df.columns)}")
    print(f"   🎯 Customers with features: {len(features_df)}")
    
    return features_df

# Create features for ML
ml_features = create_ml_features(customers_df, contracts_df, opportunities_df, simulation_results)

# Display feature summary
print("\n📋 Feature Summary:")
print(ml_features.describe().round(2))

# =============================================================================
# 5. MACHINE LEARNING MODELS
# =============================================================================

def prepare_training_data(contracts_df, ml_features):
    """Prepare training dataset for profit prediction"""
    
    # Create training examples at contract level
    training_data = contracts_df.merge(ml_features, on='customer_id', how='left')
    
    # Select features for training
    feature_columns = [
        'credit_score', 'relationship_length_months', 'avg_monthly_spend',
        'support_tickets', 'satisfaction_score', 'size_encoded', 'high_value_industry',
        'contract_type_diversity', 'recent_contracts_12m', 'active_contracts',
        'value_per_relationship_month', 'spend_per_support_ticket', 
        'customer_stickiness', 'risk_score', 'duration_months', 'monthly_value'
    ]
    
    # Ensure all feature columns exist
    for col in feature_columns:
        if col not in training_data.columns:
            training_data[col] = 0
    
    X = training_data[feature_columns].fillna(0)
    y = training_data['total_profit']
    
    return X, y, feature_columns

def train_multiple_models(X, y, feature_columns):
    """Train and compare multiple ML models"""
    
    print("\n🧠 MACHINE LEARNING MODELS")
    print("=" * 30)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Define models
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(alpha=1.0),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\n🔄 Training {name}...")
        
        # Train model
        if name in ['Linear Regression', 'Ridge Regression']:
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            
            # Cross-validation on scaled data
            cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='r2')
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            # Cross-validation on original data
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        results[name] = {
            'model': model,
            'scaler': scaler if name in ['Linear Regression', 'Ridge Regression'] else None,
            'mse': mse,
            'rmse': np.sqrt(mse),
            'mae': mae,
            'r2': r2,
            'cv_r2_mean': cv_scores.mean(),
            'cv_r2_std': cv_scores.std(),
            'predictions': y_pred,
            'actual': y_test
        }
        
        print(f"   📊 R² Score: {r2:.3f}")
        print(f"   📊 RMSE: ${np.sqrt(mse):,.0f}")
        print(f"   📊 CV R² (mean±std): {cv_scores.mean():.3f}±{cv_scores.std():.3f}")
    
    # Select best model based on cross-validation R²
    best_model_name = max(results.keys(), key=lambda x: results[x]['cv_r2_mean'])
    best_model_info = results[best_model_name]
    
    print(f"\n🏆 Best Model: {best_model_name}")
    print(f"   📊 CV R² Score: {best_model_info['cv_r2_mean']:.3f}")
    
    return results, best_model_name, feature_columns

def analyze_feature_importance(results, feature_columns):
    """Analyze feature importance for tree-based models"""
    
    # Feature importance from Random Forest
    if 'Random Forest' in results:
        rf_model = results['Random Forest']['model']
        feature_importance = pd.DataFrame({
            'feature': feature_columns,
            'importance': rf_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"\n🔍 FEATURE IMPORTANCE (Random Forest)")
        print("=" * 40)
        for _, row in feature_importance.head(10).iterrows():
            print(f"   {row['feature']}: {row['importance']:.3f}")
        
        # Plot feature importance
        plt.figure(figsize=(12, 8))
        sns.barplot(data=feature_importance.head(15), x='importance', y='feature')
        plt.title('Top 15 Feature Importance (Random Forest)')
        plt.xlabel('Importance Score')
        plt.tight_layout()
        plt.show()
        
        return feature_importance
    
    return None

def plot_model_comparison(results):
    """Compare model performance visually"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. R² scores comparison
    r2_scores = [results[name]['r2'] for name in results.keys()]
    cv_r2_scores = [results[name]['cv_r2_mean'] for name in results.keys()]
    
    x = np.arange(len(results))
    width = 0.35
    
    axes[0, 0].bar(x - width/2, r2_scores, width, label='Test R²', alpha=0.8)
    axes[0, 0].bar(x + width/2, cv_r2_scores, width, label='CV R²', alpha=0.8)
    axes[0, 0].set_xlabel('Models')
    axes[0, 0].set_ylabel('R² Score')
    axes[0, 0].set_title('Model Performance Comparison')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(results.keys(), rotation=45)
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. RMSE comparison
    rmse_scores = [results[name]['rmse'] for name in results.keys()]
    axes[0, 1].bar(results.keys(), rmse_scores, alpha=0.8)
    axes[0, 1].set_xlabel('Models')
    axes[0, 1].set_ylabel('RMSE ($)')
    axes[0, 1].set_title('RMSE Comparison')
    axes[0, 1].tick_params(axis='x', rotation=45)
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Actual vs Predicted for best model
    best_model_name = max(results.keys(), key=lambda x: results[x]['cv_r2_mean'])
    best_result = results[best_model_name]
    
    axes[1, 0].scatter(best_result['actual'], best_result['predictions'], alpha=0.7)
    min_val = min(best_result['actual'].min(), best_result['predictions'].min())
    max_val = max(best_result['actual'].max(), best_result['predictions'].max())
    axes[1, 0].plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
    axes[1, 0].set_xlabel('Actual Profit ($)')
    axes[1, 0].set_ylabel('Predicted Profit ($)')
    axes[1, 0].set_title(f'Actual vs Predicted ({best_model_name})')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Residuals for best model
    residuals = best_result['actual'] - best_result['predictions']
    axes[1, 1].scatter(best_result['predictions'], residuals, alpha=0.7)
    axes[1, 1].axhline(y=0, color='r', linestyle='--', alpha=0.8)
    axes[1, 1].set_xlabel('Predicted Profit ($)')
    axes[1, 1].set_ylabel('Residuals ($)')
    axes[1, 1].set_title(f'Residual Plot ({best_model_name})')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# Prepare training data and train models
X, y, feature_columns = prepare_training_data(contracts_df, ml_features)
model_results, best_model_name, feature_columns = train_multiple_models(X, y, feature_columns)

# Analyze feature importance
feature_importance = analyze_feature_importance(model_results, feature_columns)

# Plot model comparisons
plot_model_comparison(model_results)

# =============================================================================
# 6. PROFIT PREDICTIONS
# =============================================================================

def predict_customer_profits(customers_df, ml_features, simulation_results, 
                           model_results, best_model_name, feature_columns, 
                           months_to_predict=36):
    """Generate comprehensive profit predictions for all customers"""
    
    print(f"\n📈 PROFIT PREDICTIONS ({months_to_predict} months)")
    print("=" * 40)
    
    best_model_info = model_results[best_model_name]
    model = best_model_info['model']
    scaler = best_model_info['scaler']
    
    predictions = []
    
    for _, customer in customers_df.iterrows():
        customer_id = customer['customer_id']
        
        # Get customer features
        customer_features = ml_features[ml_features['customer_id'] == customer_id].iloc[0]
        
        # Existing contracts profit (remaining value from active contracts)
        active_contracts = contracts_df[
            (contracts_df['customer_id'] == customer_id) & 
            (contracts_df['status'].isin(['Open', 'New']))
        ]
        
        existing_profit = 0
        for _, contract in active_contracts.iterrows():
            # Calculate remaining months
            remaining_months = min(months_to_predict, contract['duration_months'])
            monthly_profit = contract['total_profit'] / contract['duration_months']
            existing_profit += monthly_profit * remaining_months
        
        # Opportunity-based profit (from simulation results)
        customer_sim_results = simulation_results[simulation_results['customer_id'] == customer_id]
        opportunity_profit = customer_sim_results['expected_value'].sum() * 0.25  # Assume 25% profit margin
        
        # ML-based new business prediction
        # Prepare features for prediction
        ml_features_for_pred = [
            customer_features['credit_score'],
            customer_features['relationship_length_months'],
            customer_features['avg_monthly_spend'],
            customer_features['support_tickets'],
            customer_features['satisfaction_score'],
            customer_features['size_encoded'],
            customer_features['high_value_industry'],
            customer_features.get('contract_type_diversity', 0),
            customer_features.get('recent_contracts_12m', 0),
            customer_features.get('active_contracts', 0),
            customer_features.get('value_per_relationship_month', 0),
            customer_features.get('spend_per_support_ticket', 0),
            customer_features.get('customer_stickiness', 0),
            customer_features.get('risk_score', 0),
            months_to_predict / 3,  # Average contract duration assumption
            customer_features['avg_monthly_spend']  # Monthly value assumption
        ]
        
        # Make prediction
        if scaler:
            ml_features_scaled = scaler.transform([ml_features_for_pred])
            ml_prediction = model.predict(ml_features_scaled)[0]
        else:
            ml_prediction = model.predict([ml_features_for_pred])[0]
        
        # Adjust ML prediction for time period
        ml_prediction = max(0, ml_prediction * (months_to_predict / 12))
        
        # Calculate total predicted profit
        total_profit = existing_profit + opportunity_profit + ml_prediction
        
        # Calculate risk and confidence scores
        risk_score = calculate_risk_score(customer_features)
        confidence_score = calculate_confidence_score(customer_features, active_contracts)
        
        predictions.append({
            'customer_id': customer_id,
            'company_name': customer['company_name'],
            'industry': customer['industry'],
            'size': customer['size'],
            'existing_contracts_profit': existing_profit,
            'opportunities_profit': opportunity_profit,
            'ml_predicted_new_business': ml_prediction,
            'total_predicted_profit': total_profit,
            'risk_score': risk_score,
            'confidence_score': confidence_score,
            'months_predicted': months_to_predict
        })
    
    predictions_df = pd.DataFrame(predictions)
    predictions_df = predictions_df.sort_values('total_predicted_profit', ascending=False)
    
    # Summary statistics
    total_predicted = predictions_df['total_predicted_profit'].sum()
    avg_risk = predictions_df['risk_score'].mean()
    high_value_customers = len(predictions_df[predictions_df['total_predicted_profit'] > 100000])
    
    print(f"📊 Prediction Summary:")
    print(f"   Total predicted profit: ${total_predicted:,.0f}")
    print(f"   Average risk score: {avg_risk:.2f}")
    print(f"   High-value customers (>$100K): {high_value_customers}")
    print(f"   Top customer profit: ${predictions_df.iloc[0]['total_predicted_profit']:,.0f}")
    
    return predictions_df

def calculate_risk_score(customer_features):
    """Calculate comprehensive risk score"""
    risk = 0
    
    # Credit risk
    if customer_features['credit_score'] < 400:
        risk += 0.25
    elif customer_features['credit_score'] < 600:
        risk += 0.15
    
    # Support burden
    if customer_features['support_tickets'] > 40:
        risk += 0.2
    elif customer_features['support_tickets'] > 25:
        risk += 0.1
    
    # Satisfaction risk
    if customer_features['satisfaction_score'] < 2.5:
        risk += 0.25
    elif customer_features['satisfaction_score'] < 3.5:
        risk += 0.15
    
    # Relationship maturity
    if customer_features['relationship_length_months'] < 6:
        risk += 0.15
    elif customer_features['relationship_length_months'] < 12:
        risk += 0.05
    
    return min(1, risk)

def calculate_confidence_score(customer_features, active_contracts):
    """Calculate prediction confidence score"""
    confidence = 0.5  # Base confidence
    
    # Positive factors
    if customer_features['satisfaction_score'] > 4:
        confidence += 0.2
    if customer_features['relationship_length_months'] > 24:
        confidence += 0.15
    if customer_features['credit_score'] > 700:
        confidence += 0.1
    if len(active_contracts) > 2:
        confidence += 0.05
    
    # Industry stability
    if customer_features['high_value_industry']:
        confidence += 0.1
    
    return min(1, confidence)

def create_prediction_visualizations(predictions_df):
    """Create comprehensive prediction visualizations"""
    
    fig, axes = plt.subplots(3, 2, figsize=(16, 18))
    
    # 1. Top 10 customers by predicted profit
    top_10 = predictions_df.head(10)
    bars = axes[0, 0].barh(range(len(top_10)), top_10['total_predicted_profit'])
    axes[0, 0].set_yticks(range(len(top_10)))
    axes[0, 0].set_yticklabels(top_10['company_name'])
    axes[0, 0].set_xlabel('Predicted Profit ($)')
    axes[0, 0].set_title('Top 10 Customers by Predicted Profit')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, (bar, value) in enumerate(zip(bars, top_10['total_predicted_profit'])):
        axes[0, 0].text(value + max(top_10['total_predicted_profit']) * 0.01, 
                       bar.get_y() + bar.get_height()/2, 
                       f'${value:,.0f}', 
                       va='center', fontsize=8)
    
    # 2. Profit breakdown for top 10
    profit_components = top_10[['existing_contracts_profit', 'opportunities_profit', 'ml_predicted_new_business']].values
    bottom1 = top_10['existing_contracts_profit']
    bottom2 = bottom1 + top_10['opportunities_profit']
    
    axes[0, 1].barh(range(len(top_10)), top_10['existing_contracts_profit'], 
                    label='Existing Contracts', alpha=0.8)
    axes[0, 1].barh(range(len(top_10)), top_10['opportunities_profit'], 
                    left=bottom1, label='Opportunities', alpha=0.8)
    axes[0, 1].barh(range(len(top_10)), top_10['ml_predicted_new_business'], 
                    left=bottom2, label='ML Predicted New', alpha=0.8)
    
    axes[0, 1].set_yticks(range(len(top_10)))
    axes[0, 1].set_yticklabels(top_10['company_name'])
    axes[0, 1].set_xlabel('Profit Components ($)')
    axes[0, 1].set_title('Profit Breakdown - Top 10 Customers')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Risk vs Predicted Profit
    scatter = axes[1, 0].scatter(predictions_df['risk_score'], 
                                predictions_df['total_predicted_profit'],
                                c=predictions_df['confidence_score'], 
                                cmap='viridis', alpha=0.7, s=60)
    axes[1, 0].set_xlabel('Risk Score')
    axes[1, 0].set_ylabel('Predicted Profit ($)')
    axes[1, 0].set_title('Risk vs Predicted Profit (Color: Confidence)')
    plt.colorbar(scatter, ax=axes[1, 0], label='Confidence Score')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Industry analysis
    industry_profits = predictions_df.groupby('industry')['total_predicted_profit'].sum().sort_values(ascending=False)
    axes[1, 1].bar(range(len(industry_profits)), industry_profits.values)
    axes[1, 1].set_xticks(range(len(industry_profits)))
    axes[1, 1].set_xticklabels(industry_profits.index, rotation=45)
    axes[1, 1].set_ylabel('Total Predicted Profit ($)')
    axes[1, 1].set_title('Predicted Profit by Industry')
    axes[1, 1].grid(True, alpha=0.3)
    
    # 5. Size analysis
    size_analysis = predictions_df.groupby('size').agg({
        'total_predicted_profit': ['sum', 'mean', 'count']
    }).round(0)
    size_analysis.columns = ['Total Profit', 'Avg Profit', 'Count']
    
    x_pos = np.arange(len(size_analysis))
    axes[2, 0].bar(x_pos - 0.2, size_analysis['Total Profit'], 0.4, label='Total', alpha=0.8)
    axes[2, 0].bar(x_pos + 0.2, size_analysis['Avg Profit'], 0.4, label='Average', alpha=0.8)
    axes[2, 0].set_xticks(x_pos)
    axes[2, 0].set_xticklabels(size_analysis.index)
    axes[2, 0].set_ylabel('Profit ($)')
    axes[2, 0].set_title('Profit Analysis by Customer Size')
    axes[2, 0].legend()
    axes[2, 0].grid(True, alpha=0.3)
    
    # 6. Confidence and Risk Distribution
    axes[2, 1].hist(predictions_df['confidence_score'], alpha=0.7, bins=15, label='Confidence Score')
    axes[2, 1].hist(predictions_df['risk_score'], alpha=0.7, bins=15, label='Risk Score')
    axes[2, 1].set_xlabel('Score')
    axes[2, 1].set_ylabel('Frequency')
    axes[2, 1].set_title('Distribution of Confidence and Risk Scores')
    axes[2, 1].legend()
    axes[2, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# Generate predictions
customer_predictions = predict_customer_profits(
    customers_df, ml_features, simulation_results, 
    model_results, best_model_name, feature_columns, 
    months_to_predict=36
)

# Create prediction visualizations
create_prediction_visualizations(customer_predictions)

# =============================================================================
# 7. DETAILED CUSTOMER ANALYSIS
# =============================================================================

def detailed_customer_analysis(customer_predictions, customers_df, contracts_df, opportunities_df):
    """Provide detailed analysis for top customers"""
    
    print(f"\n🔍 DETAILED CUSTOMER ANALYSIS")
    print("=" * 35)
    
    top_5_customers = customer_predictions.head(5)
    
    for _, customer_pred in top_5_customers.iterrows():
        customer_id = customer_pred['customer_id']
        customer_info = customers_df[customers_df['customer_id'] == customer_id].iloc[0]
        customer_contracts = contracts_df[contracts_df['customer_id'] == customer_id]
        customer_opportunities = opportunities_df[opportunities_df['customer_id'] == customer_id]
        
        print(f"\n🏢 {customer_pred['company_name']}")
        print("-" * (len(customer_pred['company_name']) + 5))
        
        print(f"📊 Customer Profile:")
        print(f"   Industry: {customer_info['industry']}")
        print(f"   Size: {customer_info['size']}")
        print(f"   Credit Score: {customer_info['credit_score']:.0f}")
        print(f"   Relationship: {customer_info['relationship_length_months']:.0f} months")
        print(f"   Satisfaction: {customer_info['satisfaction_score']:.1f}/5.0")
        print(f"   Avg Monthly Spend: ${customer_info['avg_monthly_spend']:,.0f}")
        print(f"   Support Tickets: {customer_info['support_tickets']:.0f}")
        
        print(f"\n📋 Contract Portfolio:")
        print(f"   Total Contracts: {len(customer_contracts)}")
        print(f"   Active Contracts: {len(customer_contracts[customer_contracts['status'] != 'Closed'])}")
        print(f"   Contract Types: {customer_contracts['contract_type'].nunique()}")
        print(f"   Total Historical Value: ${customer_contracts['total_value'].sum():,.0f}")
        print(f"   Total Historical Profit: ${customer_contracts['total_profit'].sum():,.0f}")
        print(f"   Avg Profit Margin: {customer_contracts['profit_margin'].mean():.1%}")
        
        if len(customer_opportunities) > 0:
            print(f"\n🎯 Current Opportunities:")
            print(f"   Active Opportunities: {len(customer_opportunities)}")
            print(f"   Total Pipeline Value: ${customer_opportunities['estimated_value'].sum():,.0f}")
            print(f"   Avg Conversion Probability: {customer_opportunities['conversion_probability'].mean():.1%}")
        
        print(f"\n📈 36-Month Prediction:")
        print(f"   Existing Contracts: ${customer_pred['existing_contracts_profit']:,.0f}")
        print(f"   Expected Opportunities: ${customer_pred['opportunities_profit']:,.0f}")
        print(f"   ML Predicted New Business: ${customer_pred['ml_predicted_new_business']:,.0f}")
        print(f"   📊 TOTAL PREDICTED PROFIT: ${customer_pred['total_predicted_profit']:,.0f}")
        
        print(f"\n⚠️ Risk Assessment:")
        risk_level = "Low" if customer_pred['risk_score'] < 0.3 else "Medium" if customer_pred['risk_score'] < 0.6 else "High"
        print(f"   Risk Score: {customer_pred['risk_score']:.2f} ({risk_level} Risk)")
        print(f"   Confidence Score: {customer_pred['confidence_score']:.2f}")
        
        # Recommendations
        print(f"\n💡 Recommendations:")
        if customer_pred['risk_score'] > 0.5:
            print("   ⚠️ High risk customer - consider additional credit checks and closer monitoring")
        if customer_info['satisfaction_score'] < 3.5:
            print("   📞 Low satisfaction - prioritize customer success initiatives")
        if len(customer_opportunities) > 0:
            print("   🎯 Active opportunities - focus sales efforts for pipeline conversion")
        if customer_info['support_tickets'] > 30:
            print("   🔧 High support load - investigate service quality issues")

# Run detailed analysis
detailed_customer_analysis(customer_predictions, customers_df, contracts_df, opportunities_df)

# =============================================================================
# 8. BUSINESS INSIGHTS AND SUMMARY
# =============================================================================

def generate_business_insights(customer_predictions, customers_df, contracts_df, opportunities_df, simulation_results):
    """Generate comprehensive business insights"""
    
    print(f"\n💼 BUSINESS INSIGHTS & RECOMMENDATIONS")
    print("=" * 45)
    
    # Portfolio analysis
    total_predicted_profit = customer_predictions['total_predicted_profit'].sum()
    total_historical_profit = contracts_df['total_profit'].sum()
    growth_rate = (total_predicted_profit / (total_historical_profit * 3)) - 1  # Annualized growth
    
    print(f"\n📈 Portfolio Performance:")
    print(f"   Historical profit: ${total_historical_profit:,.0f}")
    print(f"   36-month predicted profit: ${total_predicted_profit:,.0f}")
    print(f"   Implied annual growth rate: {growth_rate:.1%}")
    
    # Customer segmentation insights
    high_value = customer_predictions[customer_predictions['total_predicted_profit'] > 100000]
    medium_value = customer_predictions[
        (customer_predictions['total_predicted_profit'] > 50000) & 
        (customer_predictions['total_predicted_profit'] <= 100000)
    ]
    low_value = customer_predictions[customer_predictions['total_predicted_profit'] <= 50000]
    
    print(f"\n🎯 Customer Segmentation:")
    print(f"   High Value (>$100K): {len(high_value)} customers, ${high_value['total_predicted_profit'].sum():,.0f} profit")
    print(f"   Medium Value ($50K-$100K): {len(medium_value)} customers, ${medium_value['total_predicted_profit'].sum():,.0f} profit")
    print(f"   Low Value (<$50K): {len(low_value)} customers, ${low_value['total_predicted_profit'].sum():,.0f} profit")
    
    # Risk analysis
    high_risk = customer_predictions[customer_predictions['risk_score'] > 0.5]
    total_high_risk_value = high_risk['total_predicted_profit'].sum()
    
    print(f"\n⚠️ Risk Analysis:")
    print(f"   High-risk customers: {len(high_risk)} ({len(high_risk)/len(customer_predictions):.1%} of portfolio)")
    print(f"   Profit at risk: ${total_high_risk_value:,.0f} ({total_high_risk_value/total_predicted_profit:.1%} of total)")
    
    # Industry insights
    industry_analysis = customer_predictions.groupby('industry').agg({
        'total_predicted_profit': ['sum', 'mean', 'count'],
        'risk_score': 'mean'
    }).round(2)
    industry_analysis.columns = ['Total Profit', 'Avg Profit', 'Count', 'Avg Risk']
    industry_analysis = industry_analysis.sort_values('Total Profit', ascending=False)
    
    print(f"\n🏭 Industry Analysis:")
    for industry, row in industry_analysis.iterrows():
        print(f"   {industry}: ${row['Total Profit']:,.0f} total, {row['Count']:.0f} customers, {row['Avg Risk']:.2f} avg risk")
    
    # Opportunity pipeline
    total_pipeline = opportunities_df['estimated_value'].sum()
    expected_pipeline = simulation_results['expected_value'].sum()
    pipeline_conversion = expected_pipeline / total_pipeline if total_pipeline > 0 else 0
    
    print(f"\n🎲 Opportunity Pipeline:")
    print(f"   Total pipeline value: ${total_pipeline:,.0f}")
    print(f"   Expected converted value: ${expected_pipeline:,.0f}")
    print(f"   Expected conversion rate: {pipeline_conversion:.1%}")
    
    # Strategic recommendations
    print(f"\n🚀 Strategic Recommendations:")
    
    # Customer focus
    print(f"\n   👥 Customer Management:")
    if len(high_value) > 0:
        print(f"   • Focus on {len(high_value)} high-value customers generating {high_value['total_predicted_profit'].sum()/total_predicted_profit:.1%} of profit")
    
    if len(high_risk) > 0:
        print(f"   • Implement risk mitigation for {len(high_risk)} high-risk customers")
    
    # Sales focus
    print(f"\n   💰 Sales Strategy:")
    top_industry = industry_analysis.index[0]
    print(f"   • Prioritize {top_industry} industry (highest profit potential)")
    
    if len(opportunities_df) > 0:
        high_prob_opps = opportunities_df[opportunities_df['conversion_probability'] > 0.7]
        print(f"   • Focus on {len(high_prob_opps)} high-probability opportunities")
    
    # Service optimization
    low_satisfaction = customers_df[customers_df['satisfaction_score'] < 3.5]
    high_support = customers_df[customers_df['support_tickets'] > 30]
    
    print(f"\n   🔧 Service Optimization:")
    if len(low_satisfaction) > 0:
        print(f"   • Improve satisfaction for {len(low_satisfaction)} underperforming customers")
    if len(high_support) > 0:
        print(f"   • Reduce support burden for {len(high_support)} high-maintenance customers")

# Generate comprehensive business insights
generate_business_insights(customer_predictions, customers_df, contracts_df, opportunities_df, simulation_results)

# =============================================================================
# 9. EXPORT RESULTS
# =============================================================================

def export_results_to_csv():
    """Export all results to CSV files for further analysis"""
    
    print(f"\n💾 EXPORTING RESULTS")
    print("=" * 20)
    
    # Export customer predictions
    customer_predictions.to_csv('customer_profit_predictions.csv', index=False)
    print("✅ Customer predictions exported to 'customer_profit_predictions.csv'")
    
    # Export simulation results
    simulation_results.to_csv('monte_carlo_simulation_results.csv', index=False)
    print("✅ Simulation results exported to 'monte_carlo_simulation_results.csv'")
    
    # Export feature importance
    if feature_importance is not None:
        feature_importance.to_csv('feature_importance_analysis.csv', index=False)
        print("✅ Feature importance exported to 'feature_importance_analysis.csv'")
    
    # Export raw data
    customers_df.to_csv('customers_data.csv', index=False)
    contracts_df.to_csv('contracts_data.csv', index=False)
    opportunities_df.to_csv('opportunities_data.csv', index=False)
    print("✅ Raw datasets exported to CSV files")
    
    # Create summary report
    summary_data = {
        'Total Customers': [len(customers_df)],
        'Total Contracts': [len(contracts_df)],
        'Total Opportunities': [len(opportunities_df)],
        'Total Historical Profit': [contracts_df['total_profit'].sum()],
        'Total Predicted Profit (36m)': [customer_predictions['total_predicted_profit'].sum()],
        'Best Model': [best_model_name],
        'Best Model R2': [model_results[best_model_name]['cv_r2_mean']],
        'High Risk Customers': [len(customer_predictions[customer_predictions['risk_score'] > 0.5])],
        'High Value Customers': [len(customer_predictions[customer_predictions['total_predicted_profit'] > 100000])]
    }
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv('executive_summary.csv', index=False)
    print("✅ Executive summary exported to 'executive_summary.csv'")

# Export all results
export_results_to_csv()

print(f"\n🎉 ANALYSIS COMPLETE!")
print("=" * 25)
print("All data, models, and predictions have been generated and exported.")
print("Check the CSV files for detailed results and further analysis.")
print("\nKey deliverables:")
print("• Customer profit predictions for next 36 months")
print("• Monte Carlo simulation of opportunity conversions")
print("• Machine learning model for profit prediction")
print("• Comprehensive risk and confidence scoring")
print("• Business insights and strategic recommendations")
