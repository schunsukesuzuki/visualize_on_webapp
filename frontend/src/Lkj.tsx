import React, { useState } from 'react';
import './Lkj.css';

interface VarianceStats {
  mean: number;
  median: number;
  min: number;
  max: number;
  std: number;
}

interface CovarianceStats {
  mean: number;
  median: number;
  min: number;
  max: number;
  std: number;
}

interface CovarianceSummary {
  matrix_shape: number[];
  variance_statistics: VarianceStats;
  covariance_statistics: CovarianceStats;
  variances: number[];
  mean_covariance_matrix: number[][];
}

interface MCMCResult {
  status: string;
  message: string;
  elapsed_time: number;
  elapsed_time_minutes: number;
}

const API_BASE_URL = 'http://localhost:8000';

const Lkj: React.FC = () => {
  const [isRunningMCMC, setIsRunningMCMC] = useState(false);
  const [mcmcResult, setMcmcResult] = useState<MCMCResult | null>(null);
  const [summary, setSummary] = useState<CovarianceSummary | null>(null);
  const [heatmapImage, setHeatmapImage] = useState<string | null>(null);
  const [isLoadingSummary, setIsLoadingSummary] = useState(false);
  const [isLoadingHeatmap, setIsLoadingHeatmap] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const runMCMC = async () => {
    setIsRunningMCMC(true);
    setError(null);
    setMcmcResult(null);
    setSummary(null);
    setHeatmapImage(null);

    try {
      const response = await fetch(`${API_BASE_URL}/api/lkj/run-mcmc`, {
        method: 'POST',
      });

      if (!response.ok) {
        throw new Error('failled MCMC sampling');
      }

      const data: MCMCResult = await response.json();
      setMcmcResult(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'error occurred');
    } finally {
      setIsRunningMCMC(false);
    }
  };

  const loadSummary = async () => {
    setIsLoadingSummary(true);
    setError(null);

    try {
      const response = await fetch(`${API_BASE_URL}/api/lkj/covariance-summary`);

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'failed to get summary');
      }

      const data: CovarianceSummary = await response.json();
      setSummary(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'error occurred');
    } finally {
      setIsLoadingSummary(false);
    }
  };

  const loadHeatmap = async () => {
    setIsLoadingHeatmap(true);
    setError(null);

    try {
      const response = await fetch(`${API_BASE_URL}/api/lkj/covariance-heatmap-base64`);

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'failed to get heatmap');
      }

      const data = await response.json();
      setHeatmapImage(data.image);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'error occurred');
    } finally {
      setIsLoadingHeatmap(false);
    }
  };

  const clearCache = async () => {
    try {
      await fetch(`${API_BASE_URL}/api/lkj/clear-cache`, {
        method: 'DELETE',
      });
      setMcmcResult(null);
      setSummary(null);
      setHeatmapImage(null);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'failed to clear cashes');
    }
  };

  return (
    <div className="lkj-container">
      <header className="lkj-header">
        <h1>LKJ corr matrix estimation - cov matrix analysis</h1>
        <p className="subtitle">posterior from cov matrix by beyesian modeling</p>
      </header>

      {error && (
        <div className="error-message">
          <strong>error:</strong> {error}
        </div>
      )}

      <div className="control-panel">
        <button
          className="btn btn-primary"
          onClick={runMCMC}
          disabled={isRunningMCMC}
        >
          {isRunningMCMC ? 'now sampling with MCMC...' : 'run MCMC'}
        </button>

        <button
          className="btn btn-secondary"
          onClick={loadSummary}
          disabled={!mcmcResult || isLoadingSummary}
        >
          {isLoadingSummary ? 'getting summary...' : 'getting summary of cov matrix'}
        </button>

        <button
          className="btn btn-secondary"
          onClick={loadHeatmap}
          disabled={!summary || isLoadingHeatmap}
        >
          {isLoadingHeatmap ? 'getting heatmap...' : 'showing heatmap'}
        </button>

        <button
          className="btn btn-danger"
          onClick={clearCache}
        >
          clear cashes
        </button>
      </div>

      {mcmcResult && (
        <div className="result-section">
          <h2>results of MCMC sampling</h2>
          <div className="result-card">
            <p><strong>status:</strong> {mcmcResult.status}</p>
            <p><strong>message:</strong> {mcmcResult.message}</p>
            <p><strong>running time:</strong> {mcmcResult.elapsed_time.toFixed(2)}秒 ({mcmcResult.elapsed_time_minutes.toFixed(2)}分)</p>
          </div>
        </div>
      )}

      {summary && (
        <div className="result-section">
          <h2>summary of stats for cov matrix</h2>
          
          <div className="result-card">
            <h3>matrix info</h3>
            <p><strong>size:</strong> {summary.matrix_shape[0]} × {summary.matrix_shape[1]}</p>
          </div>

          <div className="stats-grid">
            <div className="result-card">
              <h3>statsfor cov</h3>
              <table className="stats-table">
                <tbody>
                  <tr>
                    <td>mean:</td>
                    <td>{summary.variance_statistics.mean.toFixed(4)}</td>
                  </tr>
                  <tr>
                    <td>median:</td>
                    <td>{summary.variance_statistics.median.toFixed(4)}</td>
                  </tr>
                  <tr>
                    <td>min:</td>
                    <td>{summary.variance_statistics.min.toFixed(4)}</td>
                  </tr>
                  <tr>
                    <td>max:</td>
                    <td>{summary.variance_statistics.max.toFixed(4)}</td>
                  </tr>
                  <tr>
                    <td>std:</td>
                    <td>{summary.variance_statistics.std.toFixed(4)}</td>
                  </tr>
                </tbody>
              </table>
            </div>

            <div className="result-card">
              <h3>共分散の統計量 (非対角要素)</h3>
              <table className="stats-table">
                <tbody>
                  <tr>
                    <td>平均:</td>
                    <td>{summary.covariance_statistics.mean.toFixed(4)}</td>
                  </tr>
                  <tr>
                    <td>中央値:</td>
                    <td>{summary.covariance_statistics.median.toFixed(4)}</td>
                  </tr>
                  <tr>
                    <td>最小値:</td>
                    <td>{summary.covariance_statistics.min.toFixed(4)}</td>
                  </tr>
                  <tr>
                    <td>最大値:</td>
                    <td>{summary.covariance_statistics.max.toFixed(4)}</td>
                  </tr>
                  <tr>
                    <td>標準偏差:</td>
                    <td>{summary.covariance_statistics.std.toFixed(4)}</td>
                  </tr>
                </tbody>
              </table>
            </div>
          </div>

          <div className="result-card">
            <h3>各特徴量の分散 (対角要素)</h3>
            <div className="variance-list">
              {summary.variances.map((variance, index) => (
                <div key={index} className="variance-item">
                  <span className="variance-label">Feature {index}:</span>
                  <span className="variance-value">{variance.toFixed(4)}</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      {heatmapImage && (
        <div className="result-section">
          <h2>分散共分散行列のヒートマップ</h2>
          <div className="heatmap-container">
            <img src={heatmapImage} alt="Covariance Heatmap" className="heatmap-image" />
          </div>
        </div>
      )}

      {!mcmcResult && !summary && !heatmapImage && (
        <div className="placeholder">
          <p>👆 MCMCサンプリングを実行してください</p>
        </div>
      )}
    </div>
  );
};

export default Lkj;
