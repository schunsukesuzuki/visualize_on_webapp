import React, { useState, useEffect } from 'react';
import './App.css';

function App() {
  const [visitors, setVisitors] = useState('');
  const [prediction, setPrediction] = useState(null);
  const [modelInfo, setModelInfo] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const API_URL = 'http://localhost:8000';

  // モデル情報を取得
  useEffect(() => {
    fetchModelInfo();
  }, []);

  const fetchModelInfo = async () => {
    try {
      const response = await fetch(`${API_URL}/model/info`);
      if (!response.ok) throw new Error('モデル情報の取得に失敗しました');
      const data = await response.json();
      setModelInfo(data);
    } catch (err) {
      console.error('モデル情報取得エラー:', err);
    }
  };

  const handlePredict = async (e) => {
    e.preventDefault();
    setError(null);
    
    const visitorCount = parseFloat(visitors);
    
    if (isNaN(visitorCount) || visitorCount <= 0) {
      setError('正の数値を入力してください');
      return;
    }

    setLoading(true);

    try {
      const response = await fetch(`${API_URL}/predict`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ visitors: visitorCount }),
      });

      if (!response.ok) {
        throw new Error('予測に失敗しました');
      }

      const data = await response.json();
      setPrediction(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const formatNumber = (num) => {
    return new Intl.NumberFormat('ja-JP').format(Math.round(num));
  };

  return (
    <div className="App">
      <div className="container">
        <header className="header">
          <h1>📊 売上高予測システム</h1>
          <p className="subtitle">来客者数から売上高を予測します</p>
        </header>

        {modelInfo && (
          <div className="model-info">
            <h3>📈 回帰モデル情報</h3>
            <p className="equation">{modelInfo.equation}</p>
            <div className="stats">
              <div className="stat-item">
                <span className="stat-label">傾き:</span>
                <span className="stat-value">{modelInfo.slope.toFixed(2)}</span>
              </div>
              <div className="stat-item">
                <span className="stat-label">切片:</span>
                <span className="stat-value">{formatNumber(modelInfo.intercept)}</span>
              </div>
            </div>
          </div>
        )}

        <form onSubmit={handlePredict} className="prediction-form">
          <div className="form-group">
            <label htmlFor="visitors">来客者数（人）</label>
            <input
              type="number"
              id="visitors"
              value={visitors}
              onChange={(e) => setVisitors(e.target.value)}
              placeholder="例: 100"
              min="1"
              step="1"
              required
            />
          </div>

          <button type="submit" disabled={loading} className="predict-button">
            {loading ? '予測中...' : '売上高を予測'}
          </button>
        </form>

        {error && (
          <div className="error-message">
            ⚠️ {error}
          </div>
        )}

        {prediction && (
          <div className="prediction-result">
            <h2>予測結果</h2>
            <div className="result-card">
              <div className="result-item">
                <span className="result-label">来客者数</span>
                <span className="result-value">{formatNumber(prediction.visitors)} 人</span>
              </div>
              <div className="arrow">→</div>
              <div className="result-item highlight">
                <span className="result-label">予測売上高</span>
                <span className="result-value sales">¥{formatNumber(prediction.predicted_sales)}</span>
              </div>
            </div>
            
            <div className="model-performance">
              <p>モデル精度 (R²): <strong>{(prediction.r2_score * 100).toFixed(2)}%</strong></p>
            </div>
          </div>
        )}

        <footer className="footer">
          <p>🔧 Backend: FastAPI + JAX | Frontend: React</p>
        </footer>
      </div>
    </div>
  );
}

export default App;
