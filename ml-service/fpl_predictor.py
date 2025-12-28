import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import psycopg2
from datetime import datetime
import json

class FPLPointsPredictor:
    def __init__(self, db_config):
        """
        Initialize the FPL Points Predictor
        
        Args:
            db_config (dict): PostgreSQL database configuration
        """
        self.db_config = db_config
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = []
        
    def fetch_data_from_db(self):
        """Fetch player data from PostgreSQL database"""
        try:
            conn = psycopg2.connect(**self.db_config)
            query = """
                SELECT 
                    player_id,
                    player_name,
                    team,
                    position,
                    total_points,
                    goals_scored,
                    assists,
                    clean_sheets,
                    minutes_played,
                    bonus_points,
                    bps,
                    influence,
                    creativity,
                    threat,
                    ict_index,
                    selected_by_percent,
                    price,
                    price_change,
                    form,
                    points_per_game,
                    expected_goals,
                    expected_assists,
                    expected_goal_involvements,
                    expected_goals_conceded,
                    difficulty_next_5,
                    fixture_count_next_5
                FROM player_stats
                WHERE minutes_played > 0
                ORDER BY gameweek DESC
            """
            df = pd.read_sql(query, conn)
            conn.close()
            return df
        except Exception as e:
            print(f"Error fetching data: {e}")
            return None
    
    def create_features(self, df):
        """Create features for the ML model"""
        # Position encoding
        position_map = {'GKP': 1, 'DEF': 2, 'MID': 3, 'FWD': 4}
        df['position_encoded'] = df['position'].map(position_map)
        
        # Rolling averages (last 5 games)
        df['goals_avg_5'] = df.groupby('player_id')['goals_scored'].transform(
            lambda x: x.rolling(5, min_periods=1).mean()
        )
        df['assists_avg_5'] = df.groupby('player_id')['assists'].transform(
            lambda x: x.rolling(5, min_periods=1).mean()
        )
        df['minutes_avg_5'] = df.groupby('player_id')['minutes_played'].transform(
            lambda x: x.rolling(5, min_periods=1).mean()
        )
        df['points_avg_5'] = df.groupby('player_id')['total_points'].transform(
            lambda x: x.rolling(5, min_periods=1).mean()
        )
        
        # Form indicators
        df['form_numeric'] = pd.to_numeric(df['form'], errors='coerce').fillna(0)
        df['points_per_game_numeric'] = pd.to_numeric(df['points_per_game'], errors='coerce').fillna(0)
        
        # Expected stats
        df['xg_numeric'] = pd.to_numeric(df['expected_goals'], errors='coerce').fillna(0)
        df['xa_numeric'] = pd.to_numeric(df['expected_assists'], errors='coerce').fillna(0)
        df['xgi_numeric'] = pd.to_numeric(df['expected_goal_involvements'], errors='coerce').fillna(0)
        df['xgc_numeric'] = pd.to_numeric(df['expected_goals_conceded'], errors='coerce').fillna(0)
        
        # Momentum features
        df['goals_momentum'] = df.groupby('player_id')['goals_scored'].transform(
            lambda x: x.diff().fillna(0)
        )
        df['assists_momentum'] = df.groupby('player_id')['assists'].transform(
            lambda x: x.diff().fillna(0)
        )
        
        # Fixture difficulty
        df['difficulty_numeric'] = pd.to_numeric(df['difficulty_next_5'], errors='coerce').fillna(3)
        df['fixture_count_numeric'] = pd.to_numeric(df['fixture_count_next_5'], errors='coerce').fillna(1)
        
        return df
    
    def prepare_training_data(self, df):
        """Prepare data for training"""
        df = self.create_features(df)
        
        # Select features for training
        self.feature_columns = [
            'position_encoded', 'minutes_played', 'form_numeric',
            'points_per_game_numeric', 'goals_avg_5', 'assists_avg_5',
            'minutes_avg_5', 'points_avg_5', 'xg_numeric', 'xa_numeric',
            'xgi_numeric', 'xgc_numeric', 'goals_momentum', 'assists_momentum',
            'bps', 'influence', 'creativity', 'threat', 'ict_index',
            'selected_by_percent', 'price', 'difficulty_numeric', 
            'fixture_count_numeric', 'clean_sheets', 'bonus_points'
        ]
        
        # Handle missing values
        for col in self.feature_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        X = df[self.feature_columns]
        y = df['total_points']
        
        return X, y, df
    
    def train_model(self, use_gradient_boost=False):
        """Train the ML model"""
        print("Fetching data from database...")
        df = self.fetch_data_from_db()
        
        if df is None or len(df) == 0:
            print("No data available for training")
            return None
        
        print(f"Preparing {len(df)} records for training...")
        X, y, processed_df = self.prepare_training_data(df)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        print("Training model...")
        if use_gradient_boost:
            self.model = GradientBoostingRegressor(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=5,
                random_state=42
            )
        else:
            self.model = RandomForestRegressor(
                n_estimators=200,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
        
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate
        train_pred = self.model.predict(X_train_scaled)
        test_pred = self.model.predict(X_test_scaled)
        
        print("\n=== Model Performance ===")
        print(f"Train MAE: {mean_absolute_error(y_train, train_pred):.2f}")
        print(f"Test MAE: {mean_absolute_error(y_test, test_pred):.2f}")
        print(f"Train RMSE: {np.sqrt(mean_squared_error(y_train, train_pred)):.2f}")
        print(f"Test RMSE: {np.sqrt(mean_squared_error(y_test, test_pred)):.2f}")
        print(f"Train R²: {r2_score(y_train, train_pred):.3f}")
        print(f"Test R²: {r2_score(y_test, test_pred):.3f}")
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\n=== Top 10 Important Features ===")
        print(feature_importance.head(10))
        
        return {
            'train_mae': mean_absolute_error(y_train, train_pred),
            'test_mae': mean_absolute_error(y_test, test_pred),
            'train_r2': r2_score(y_train, train_pred),
            'test_r2': r2_score(y_test, test_pred),
            'feature_importance': feature_importance.to_dict('records')
        }
    
    def predict_points(self, player_data):
        """
        Predict points for given player data
        
        Args:
            player_data (dict or DataFrame): Player statistics
            
        Returns:
            float or array: Predicted points
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train_model() first.")
        
        if isinstance(player_data, dict):
            player_data = pd.DataFrame([player_data])
        
        # Create features
        player_data = self.create_features(player_data)
        
        # Ensure all required columns exist
        for col in self.feature_columns:
            if col not in player_data.columns:
                player_data[col] = 0
        
        X = player_data[self.feature_columns]
        X_scaled = self.scaler.transform(X)
        
        predictions = self.model.predict(X_scaled)
        return predictions
    
    def predict_all_players(self):
        """Predict points for all players in database"""
        df = self.fetch_data_from_db()
        if df is None:
            return None
        
        df = self.create_features(df)
        
        # Get latest data for each player
        latest_df = df.groupby('player_id').last().reset_index()
        
        X = latest_df[self.feature_columns]
        X_scaled = self.scaler.transform(X)
        
        predictions = self.model.predict(X_scaled)
        
        result_df = latest_df[['player_id', 'player_name', 'team', 'position', 'price']].copy()
        result_df['predicted_points'] = predictions
        result_df['value_rating'] = result_df['predicted_points'] / result_df['price']
        
        return result_df.sort_values('predicted_points', ascending=False)
    
    def save_model(self, filepath='fpl_model.pkl'):
        """Save trained model to file"""
        if self.model is None:
            raise ValueError("No model to save. Train the model first.")
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns
        }
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath='fpl_model.pkl'):
        """Load trained model from file"""
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_columns = model_data['feature_columns']
        print(f"Model loaded from {filepath}")


# Example usage
if __name__ == "__main__":
    # Database configuration
    db_config = {
        'host': 'localhost',
        'database': 'fpl_db',
        'user': 'your_username',
        'password': 'your_password',
        'port': 5432
    }
    
    # Initialize predictor
    predictor = FPLPointsPredictor(db_config)
    
    # Train model
    metrics = predictor.train_model(use_gradient_boost=False)
    
    # Save model
    predictor.save_model('fpl_model.pkl')
    
    # Predict for all players
    predictions = predictor.predict_all_players()
    print("\n=== Top 20 Predicted Players ===")
    print(predictions.head(20))
    
    # Save predictions to CSV
    predictions.to_csv('predictions.csv', index=False)
    print("\nPredictions saved to predictions.csv")
