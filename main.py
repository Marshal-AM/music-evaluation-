from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import librosa
import soundfile as sf
import warnings
import shutil
import os
from typing import List
import uvicorn
from eth_utils import is_address
from datetime import datetime
import json
from pathlib import Path
from web3 import Web3
from eth_account import Account
from dotenv import load_dotenv
import gdown
import tempfile
os.environ['PRIVATE_KEY'] = '0xc2a12ea9d8e4dc226270d2d7aee56c4292f9a50ca3a794698fdc5e0853c3b7f4'
# Load environment variables
load_dotenv()

warnings.filterwarnings('ignore')

# Create necessary directories
Path("temp_uploads").mkdir(exist_ok=True)
Path("evaluation_results").mkdir(exist_ok=True)

# Web3 setup
BSC_TESTNET_RPC = "https://data-seed-prebsc-1-s1.binance.org:8545/"
w3 = Web3(Web3.HTTPProvider(BSC_TESTNET_RPC))
if not w3.is_connected():
    raise Exception("Failed to connect to BSC testnet")

# Contract setup
CONTRACT_ADDRESS = "0xA546819d48330FB2E02D3424676d13D7B8af3bB2"
CONTRACT_ABI = json.loads('[{"inputs":[],"stateMutability":"nonpayable","type":"constructor"},{"anonymous":false,"inputs":[{"indexed":true,"internalType":"address","name":"recipient","type":"address"},{"indexed":false,"internalType":"uint256","name":"amount","type":"uint256"}],"name":"FundsSent","type":"event"},{"anonymous":false,"inputs":[{"indexed":true,"internalType":"address","name":"user","type":"address"},{"indexed":false,"internalType":"uint256","name":"amount","type":"uint256"}],"name":"Staked","type":"event"},{"inputs":[],"name":"STAKE_AMOUNT","outputs":[{"internalType":"uint256","name":"","type":"uint256"}],"stateMutability":"view","type":"function"},{"inputs":[],"name":"owner","outputs":[{"internalType":"address","name":"","type":"address"}],"stateMutability":"view","type":"function"},{"inputs":[{"internalType":"address payable","name":"recipient","type":"address"}],"name":"sendFundsTo","outputs":[],"stateMutability":"nonpayable","type":"function"},{"inputs":[],"name":"stake","outputs":[],"stateMutability":"payable","type":"function"},{"inputs":[],"name":"withdraw","outputs":[],"stateMutability":"nonpayable","type":"function"}]')

# Initialize contract
contract = w3.eth.contract(address=CONTRACT_ADDRESS, abi=CONTRACT_ABI)

# Load private key from environment variable
PRIVATE_KEY = os.getenv('PRIVATE_KEY')
if not PRIVATE_KEY:
    raise ValueError("Private key not found in environment variables")

account = Account.from_key(PRIVATE_KEY)
account = w3.eth.account.from_key(PRIVATE_KEY)
app = FastAPI(title="Music Evaluator API",
             description="Evaluate music tracks and associate them with wallet addresses")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TrackEvaluation(BaseModel):
    wallet_address: str
    file_name: str
    quality_score: float
    features: dict

class WinnerAnnouncement(BaseModel):
    winner_wallet: str
    winning_track: str
    score: float
    timestamp: str
    all_rankings: List[TrackEvaluation]
    score_differences: List[float]
    transaction_hash: str = None

class PopMusicEvaluator:
    def __init__(self):
        self.scaler = MinMaxScaler()
        self.model = RandomForestRegressor(
            n_estimators=100,
            random_state=42
        )
        self.feature_columns = [
            'acousticness', 'danceability', 'energy',
            'instrumentalness', 'key', 'liveness',
            'loudness'
        ]
        # Initialize the model on startup
        self.download_and_train_model()

    def download_dataset(self):
        """Download dataset from Google Drive"""
        # Convert the sharing URL to direct download URL
        file_id = "1sROpIEX4itKXYM7YhHSFQnXsEfwM5UTT"
        
        # Create a temporary directory for the dataset
        temp_dir = tempfile.mkdtemp()
        output_path = os.path.join(temp_dir, "song_data.csv")
        
        try:
            # Download the file
            url = f"https://drive.google.com/uc?id={file_id}"
            gdown.download(url, output_path, quiet=False)
            
            if not os.path.exists(output_path):
                raise Exception("Failed to download the dataset")
                
            return output_path
        except Exception as e:
            raise Exception(f"Error downloading dataset: {str(e)}")

    def download_and_train_model(self):
        """Download the dataset and train the model"""
        try:
            dataset_path = self.download_dataset()
            self.train_model(dataset_path)
            # Clean up the temporary file
            os.remove(dataset_path)
        except Exception as e:
            raise Exception(f"Error in model initialization: {str(e)}")

    def prepare_dataset(self, dataset_path):
        """Load dataset from CSV and prepare it"""
        df = pd.read_csv(dataset_path)
        numeric_columns = [
            'song_popularity', 'song_duration_ms', 'acousticness',
            'danceability', 'energy', 'instrumentalness', 'key',
            'liveness', 'loudness'
        ]
        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        return df

    def train_model(self, dataset_path):
        """Train the model using the provided dataset"""
        df = self.prepare_dataset(dataset_path)
        quality_score = df['song_popularity'].values.reshape(-1, 1)
        quality_score_normalized = self.scaler.fit_transform(quality_score)
        X = df[self.feature_columns]
        X_train, X_test, y_train, y_test = train_test_split(
            X, quality_score_normalized,
            test_size=0.2,
            random_state=42
        )
        self.model.fit(X_train, y_train.ravel())
        self.feature_ranges = {
            column: (df[column].min(), df[column].max())
            for column in self.feature_columns
        }
        return self.model.score(X_test, y_test)

    # [Rest of the PopMusicEvaluator class methods remain the same]
    def extract_features(self, audio_path):
        """Extract audio features from a track"""
        y, sr = librosa.load(audio_path, duration=5)
        features = {}
        features['acousticness'] = float(np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr)))
        features['energy'] = float(np.mean(librosa.feature.rms(y=y)))
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        features['danceability'] = float(tempo / 200.0)
        harmonic, percussive = librosa.effects.hpss(y)
        features['instrumentalness'] = float(np.mean(harmonic) / (np.mean(harmonic) + np.mean(percussive)))
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        features['key'] = int(np.argmax(np.mean(chroma, axis=1)))
        features['liveness'] = float(np.mean(librosa.feature.zero_crossing_rate(y)))
        features['loudness'] = float(librosa.amplitude_to_db(np.mean(np.abs(y))))
        return features

    def evaluate_track(self, audio_path):
        """Evaluate a single audio track"""
        features = self.extract_features(audio_path)
        X = pd.DataFrame([features])[self.feature_columns]
        quality_score = self.model.predict(X)[0]
        quality_score = float(self.scaler.inverse_transform([[quality_score]])[0][0])
        return {
            'quality_score': quality_score,
            'features': features
        }

# Initialize the evaluator
evaluator = PopMusicEvaluator()

# [Rest of the code remains the same]
def validate_wallet_address(address: str) -> bool:
    """Validate Ethereum wallet address"""
    return is_address(address)

def save_evaluation_result(result: dict, timestamp: str):
    """Save evaluation result to JSON file"""
    filename = f"evaluation_results/evaluation_{timestamp}.json"
    with open(filename, 'w') as f:
        json.dump(result, f, indent=4)
    return filename

async def send_funds_to_winner(winner_address: str) -> str:
    """Send funds to the winner through the smart contract"""
    try:
        nonce = w3.eth.get_transaction_count(account.address)
        transaction = contract.functions.sendFundsTo(w3.to_checksum_address(winner_address)).build_transaction({
            'chainId': 97,
            'gas': 200000,
            'gasPrice': w3.eth.gas_price,
            'nonce': nonce,
            'from': account.address
        })
        signed_txn = w3.eth.account.sign_transaction(transaction, private_key=PRIVATE_KEY)
        tx_hash = w3.eth.send_raw_transaction(signed_txn.raw_transaction)
        tx_receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
        return '0x' + tx_hash.hex()
    except Exception as e:
        print(f"Error details: {str(e)}")
        print(f"Account address: {account.address}")
        print(f"Transaction details: {transaction}")
        raise HTTPException(status_code=500, detail=f"Failed to send funds: {str(e)}")

@app.post("/evaluate-tracks/")
async def evaluate_tracks(
    wallet_addresses: List[str] = Form(...),
    files: List[UploadFile] = File(...),
):
    """
    Evaluate multiple music tracks and associate them with wallet addresses.
    Requires three files and three wallet addresses.
    """
    if len(files) != 3 or len(wallet_addresses) != 3:
        raise HTTPException(status_code=400, detail="Exactly 3 files and 3 wallet addresses required")

    for address in wallet_addresses:
        if not validate_wallet_address(address):
            raise HTTPException(status_code=400, detail=f"Invalid wallet address: {address}")

    evaluations = []
    temp_files = []

    try:
        for file, address in zip(files, wallet_addresses):
            if not file.filename.lower().endswith(('.mp3', '.wav', '.m4a')):
                raise HTTPException(status_code=400, detail="Unsupported file format")

            temp_path = f"temp_uploads/{file.filename}"
            temp_files.append(temp_path)
            
            with open(temp_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)

            evaluation = evaluator.evaluate_track(temp_path)
            
            evaluations.append(TrackEvaluation(
                wallet_address=address,
                file_name=file.filename,
                quality_score=evaluation['quality_score'],
                features=evaluation['features']
            ))

        evaluations.sort(key=lambda x: x.quality_score, reverse=True)

        differences = []
        for i in range(len(evaluations) - 1):
            score_diff = evaluations[i].quality_score - evaluations[i + 1].quality_score
            differences.append(score_diff)

        winner = evaluations[0]
        tx_hash = await send_funds_to_winner(winner.wallet_address)

        result = WinnerAnnouncement(
            winner_wallet=winner.wallet_address,
            winning_track=winner.file_name,
            score=winner.quality_score,
            timestamp=datetime.now().isoformat(),
            all_rankings=evaluations,
            score_differences=differences,
            transaction_hash=tx_hash
        )

        save_evaluation_result(
            result.dict(),
            datetime.now().strftime("%Y%m%d_%H%M%S")
        )

        return result

    finally:
        for temp_file in temp_files:
            if os.path.exists(temp_file):
                os.remove(temp_file)

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
