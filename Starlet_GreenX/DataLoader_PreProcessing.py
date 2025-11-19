import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib
import os

class CO2Dataset(Dataset):
    def __init__(self, csv_path, scaler_path=None, mode='train'):
        self.mode = mode
        if mode == 'train':
            print(f"Loading data from {csv_path}...")
            self.data = pd.read_csv(csv_path)    
    
            required_cols = ['x', 'y', 'z', 'u', 'CO2']
            for col in required_cols:
                if col not in self.data.columns:
                    raise ValueError(f"File CSV thiếu cột bắt buộc: {col}")

            if self.data.isnull().values.any():
                self.data = self.data.dropna()

            if 'Ps' not in self.data.columns:
                 self.data['Ps'] = self.data['Vs']
            
            # Branch Input u (Context): [Vs, Ps, Source, Q]
            self.u_raw = self.data[['Vs', 'Ps', 'CO2_source', 'Q_supply']].values.astype(np.float32)
            
            # Trunk Input y (Space): [x, y, z]
            self.y_raw = self.data[['x', 'y', 'z']].values.astype(np.float32)
            
            # 3. Target (Output) = CO2
            self.target_raw = self.data[['CO2']].values.astype(np.float32)
            
        
        # Normalization
        self.scaler_u = MinMaxScaler()
        self.scaler_y = MinMaxScaler(feature_range=(-1, 1)) # Trunk về [-1, 1]
        self.scaler_target = MinMaxScaler()

        if mode == 'train':
            self.u = self.scaler_u.fit_transform(self.u_raw)
            self.y = self.scaler_y.fit_transform(self.y_raw)
            self.target = self.scaler_target.fit_transform(self.target_raw)
            
            if scaler_path:
                os.makedirs(scaler_path, exist_ok=True)
                joblib.dump(self.scaler_u, os.path.join(scaler_path, 'scaler_u.pkl'))
                joblib.dump(self.scaler_y, os.path.join(scaler_path, 'scaler_y.pkl'))
                joblib.dump(self.scaler_target, os.path.join(scaler_path, 'scaler_target.pkl'))
                print(f"-> Scalers saved to {scaler_path}")
                
        elif mode == 'inference':
            if not scaler_path or not os.path.exists(os.path.join(scaler_path, 'scaler_u.pkl')):
                raise FileNotFoundError("Scalers not found. Run training first.")
            
            self.scaler_u = joblib.load(os.path.join(scaler_path, 'scaler_u.pkl'))
            self.scaler_y = joblib.load(os.path.join(scaler_path, 'scaler_y.pkl'))
            self.scaler_target = joblib.load(os.path.join(scaler_path, 'scaler_target.pkl'))

    def transform_input(self, u_raw, y_raw):
        u_scaled = self.scaler_u.transform(u_raw)
        y_scaled = self.scaler_y.transform(y_raw)
        return u_scaled, y_scaled

    def inverse_transform_output(self, target_scaled):
        return self.scaler_target.inverse_transform(target_scaled)

    def __len__(self):
        return len(self.data) if self.mode == 'train' else 0

    def __getitem__(self, idx):
        return (torch.tensor(self.u[idx]), 
                torch.tensor(self.y[idx]), 
                torch.tensor(self.target[idx]))