import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';

// 型定義
interface GraphicalLassoResponse {
  covariance_matrix: number[][];
  precision_matrix: number[][];
  feature_names: string[];
  partial_correlations: number[][];
  config: {
    alpha: number;
    max_iter: number;
    tol: number;
    rho: number;
  };
  convergence_info?: {
    total_iterations: number;
    final_primal_residual: number;
    final_dual_residual: number;
    converged: boolean;
  };
}

interface GraphicalLassoRequest {
  alpha?: number;
  max_iter?: number;
  tol?: number;
  rho?: number;
}

type MatrixType = 'covariance' | 'precision' | 'partial_correlation';

const Graphicallasso: React.FC = () => {
// const GraphicalLassoVisualization: React.FC = () => {
  const [data, setData] = useState<GraphicalLassoResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [selectedMatrix, setSelectedMatrix] = useState<MatrixType>('partial_correlation');
  const [hoveredCell, setHoveredCell] = useState<{ i: number; j: number } | null>(null);
  
  // パラメータ設定
  const [params, setParams] = useState<GraphicalLassoRequest>({
    alpha: 0.01,
    max_iter: 100,
    tol: 0.0001,
    rho: 1.0,
  });

  const fetchGraphicalLasso = async () => {
    setLoading(true);
    setError(null);
    
    try {
      const response = await fetch('http://localhost:8000/api/graphical-lasso', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(params),
      });
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      
      const result = await response.json();
      setData(result);
    } catch (err) {
      setError(err instanceof Error ? err.message : '不明なエラーが発生しました');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchGraphicalLasso();
  }, []);

  const getMatrix = (): number[][] => {
    if (!data) return [];
    switch (selectedMatrix) {
      case 'covariance':
        return data.covariance_matrix;
      case 'precision':
        return data.precision_matrix;
      case 'partial_correlation':
        return data.partial_correlations;
      default:
        return data.partial_correlations;
    }
  };

  const getColorFromValue = (value: number, type: MatrixType): string => {
    const absValue = Math.abs(value);
    
    if (type === 'partial_correlation') {
      // 偏相関用: -1から1の範囲
      if (value > 0) {
        const intensity = Math.min(value * 255, 255);
        return `rgba(220, 38, 38, ${absValue})`;
      } else {
        const intensity = Math.min(absValue * 255, 255);
        return `rgba(37, 99, 235, ${absValue})`;
      }
    } else {
      // 共分散・精度行列用: 正の値のみ
      const normalizedValue = Math.min(absValue / 2, 1);
      return `rgba(168, 85, 247, ${normalizedValue})`;
    }
  };

  const matrixOptions: { value: MatrixType; label: string; description: string }[] = [
    { 
      value: 'partial_correlation', 
      label: '偏相関係数', 
      description: '他の変数の影響を除いた直接的な関係性'
    },
    { 
      value: 'covariance', 
      label: '共分散行列', 
      description: '変数間の線形関係の強さ'
    },
    { 
      value: 'precision', 
      label: '精度行列', 
      description: '条件付き独立性を表現'
    },
  ];

  return (
    <div style={{
      minHeight: '100vh',
      background: 'linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%)',
      padding: '3rem 2rem',
      fontFamily: '"Söhne", system-ui, -apple-system, sans-serif',
      color: '#e2e8f0',
    }}>
      <div style={{ maxWidth: '1400px', margin: '0 auto' }}>
        {/* ヘッダー */}
        <motion.div
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
          style={{ marginBottom: '3rem' }}
        >
          <h1 style={{
            fontSize: '3.5rem',
            fontWeight: 700,
            marginBottom: '0.5rem',
            background: 'linear-gradient(135deg, #a78bfa 0%, #ec4899 100%)',
            WebkitBackgroundClip: 'text',
            WebkitTextFillColor: 'transparent',
            letterSpacing: '-0.02em',
          }}>
            Graphical Lasso
          </h1>
          <p style={{
            fontSize: '1.125rem',
            color: '#94a3b8',
            marginBottom: '2rem',
          }}>
            高次元データにおけるスパースな精度行列推定
          </p>
        </motion.div>

        {/* パラメータ設定 */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.1 }}
          style={{
            background: 'rgba(255, 255, 255, 0.05)',
            backdropFilter: 'blur(10px)',
            borderRadius: '16px',
            padding: '2rem',
            marginBottom: '2rem',
            border: '1px solid rgba(255, 255, 255, 0.1)',
          }}
        >
          <h2 style={{ fontSize: '1.5rem', marginBottom: '1.5rem', fontWeight: 600 }}>
            パラメータ設定
          </h2>
          
          <div style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))',
            gap: '1.5rem',
            marginBottom: '1.5rem',
          }}>
            {[
              { key: 'alpha', label: 'Alpha (α)', step: 0.001, description: 'L1正則化パラメータ' },
              { key: 'rho', label: 'Rho (ρ)', step: 0.1, description: 'ADMM ペナルティ係数' },
              { key: 'max_iter', label: '最大反復数', step: 10, description: '収束までの最大ステップ' },
              { key: 'tol', label: '許容誤差', step: 0.00001, description: '収束判定閾値' },
            ].map(({ key, label, step, description }) => (
              <div key={key}>
                <label style={{
                  display: 'block',
                  fontSize: '0.875rem',
                  fontWeight: 500,
                  marginBottom: '0.5rem',
                  color: '#cbd5e1',
                }}>
                  {label}
                </label>
                <input
                  type="number"
                  value={params[key as keyof GraphicalLassoRequest]}
                  onChange={(e) => setParams({ ...params, [key]: parseFloat(e.target.value) })}
                  step={step}
                  style={{
                    width: '100%',
                    padding: '0.75rem',
                    background: 'rgba(255, 255, 255, 0.05)',
                    border: '1px solid rgba(255, 255, 255, 0.1)',
                    borderRadius: '8px',
                    color: '#e2e8f0',
                    fontSize: '0.875rem',
                  }}
                />
                <p style={{ fontSize: '0.75rem', color: '#64748b', marginTop: '0.25rem' }}>
                  {description}
                </p>
              </div>
            ))}
          </div>

          <button
            onClick={fetchGraphicalLasso}
            disabled={loading}
            style={{
              padding: '0.875rem 2rem',
              background: loading 
                ? 'rgba(168, 85, 247, 0.5)' 
                : 'linear-gradient(135deg, #a78bfa 0%, #ec4899 100%)',
              color: '#fff',
              border: 'none',
              borderRadius: '8px',
              fontSize: '1rem',
              fontWeight: 600,
              cursor: loading ? 'not-allowed' : 'pointer',
              transition: 'all 0.2s',
              opacity: loading ? 0.6 : 1,
            }}
          >
            {loading ? '計算中...' : '実行'}
          </button>
        </motion.div>

        {/* エラー表示 */}
        {error && (
          <motion.div
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            style={{
              background: 'rgba(220, 38, 38, 0.1)',
              border: '1px solid rgba(220, 38, 38, 0.3)',
              borderRadius: '12px',
              padding: '1.5rem',
              marginBottom: '2rem',
              color: '#fca5a5',
            }}
          >
            <strong>エラー:</strong> {error}
          </motion.div>
        )}

        {/* 結果表示 */}
        {data && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.2 }}
          >
            {/* 収束情報 */}
            {data.convergence_info && (
              <div style={{
                background: 'rgba(34, 197, 94, 0.1)',
                border: '1px solid rgba(34, 197, 94, 0.3)',
                borderRadius: '12px',
                padding: '1.5rem',
                marginBottom: '2rem',
              }}>
                <h3 style={{ fontSize: '1.25rem', marginBottom: '1rem', fontWeight: 600 }}>
                  収束情報
                </h3>
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '1rem' }}>
                  <div>
                    <p style={{ fontSize: '0.875rem', color: '#94a3b8', marginBottom: '0.25rem' }}>反復回数</p>
                    <p style={{ fontSize: '1.5rem', fontWeight: 700, color: '#22c55e' }}>
                      {data.convergence_info.total_iterations}
                    </p>
                  </div>
                  <div>
                    <p style={{ fontSize: '0.875rem', color: '#94a3b8', marginBottom: '0.25rem' }}>主残差</p>
                    <p style={{ fontSize: '1.5rem', fontWeight: 700, color: '#22c55e' }}>
                      {data.convergence_info.final_primal_residual.toExponential(2)}
                    </p>
                  </div>
                  <div>
                    <p style={{ fontSize: '0.875rem', color: '#94a3b8', marginBottom: '0.25rem' }}>双対残差</p>
                    <p style={{ fontSize: '1.5rem', fontWeight: 700, color: '#22c55e' }}>
                      {data.convergence_info.final_dual_residual.toExponential(2)}
                    </p>
                  </div>
                  <div>
                    <p style={{ fontSize: '0.875rem', color: '#94a3b8', marginBottom: '0.25rem' }}>収束状態</p>
                    <p style={{ fontSize: '1.5rem', fontWeight: 700, color: data.convergence_info.converged ? '#22c55e' : '#f59e0b' }}>
                      {data.convergence_info.converged ? '✓ 収束' : '未収束'}
                    </p>
                  </div>
                </div>
              </div>
            )}

            {/* 行列選択 */}
            <div style={{ marginBottom: '2rem' }}>
              <div style={{ display: 'flex', gap: '1rem', flexWrap: 'wrap' }}>
                {matrixOptions.map(({ value, label, description }) => (
                  <motion.button
                    key={value}
                    onClick={() => setSelectedMatrix(value)}
                    whileHover={{ scale: 1.02 }}
                    whileTap={{ scale: 0.98 }}
                    style={{
                      flex: '1 1 250px',
                      padding: '1.25rem',
                      background: selectedMatrix === value 
                        ? 'linear-gradient(135deg, #a78bfa 0%, #ec4899 100%)'
                        : 'rgba(255, 255, 255, 0.05)',
                      border: selectedMatrix === value 
                        ? 'none'
                        : '1px solid rgba(255, 255, 255, 0.1)',
                      borderRadius: '12px',
                      color: '#fff',
                      cursor: 'pointer',
                      textAlign: 'left',
                      transition: 'all 0.2s',
                    }}
                  >
                    <div style={{ fontSize: '1.125rem', fontWeight: 600, marginBottom: '0.5rem' }}>
                      {label}
                    </div>
                    <div style={{ fontSize: '0.875rem', opacity: 0.8 }}>
                      {description}
                    </div>
                  </motion.button>
                ))}
              </div>
            </div>

            {/* 行列ヒートマップ */}
            <div style={{
              background: 'rgba(255, 255, 255, 0.05)',
              backdropFilter: 'blur(10px)',
              borderRadius: '16px',
              padding: '2rem',
              border: '1px solid rgba(255, 255, 255, 0.1)',
              overflowX: 'auto',
            }}>
              <h3 style={{ fontSize: '1.5rem', marginBottom: '1.5rem', fontWeight: 600 }}>
                {matrixOptions.find(opt => opt.value === selectedMatrix)?.label}
              </h3>
              
              <div style={{ position: 'relative', display: 'inline-block' }}>
                <table style={{
                  borderCollapse: 'collapse',
                  fontSize: '0.75rem',
                }}>
                  <thead>
                    <tr>
                      <th style={{ padding: '0.5rem', minWidth: '100px' }}></th>
                      {data.feature_names.map((name, i) => (
                        <th
                          key={i}
                          style={{
                            padding: '0.5rem',
                            writingMode: 'vertical-rl',
                            textAlign: 'left',
                            color: '#94a3b8',
                            fontWeight: 500,
                          }}
                        >
                          {name}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {getMatrix().map((row, i) => (
                      <tr key={i}>
                        <td style={{
                          padding: '0.5rem',
                          color: '#94a3b8',
                          fontWeight: 500,
                          textAlign: 'right',
                        }}>
                          {data.feature_names[i]}
                        </td>
                        {row.map((value, j) => (
                          <motion.td
                            key={j}
                            onMouseEnter={() => setHoveredCell({ i, j })}
                            onMouseLeave={() => setHoveredCell(null)}
                            whileHover={{ scale: 1.1, zIndex: 10 }}
                            style={{
                              padding: '0',
                              width: '24px',
                              height: '24px',
                              position: 'relative',
                            }}
                          >
                            <div style={{
                              width: '100%',
                              height: '100%',
                              background: getColorFromValue(value, selectedMatrix),
                              border: '1px solid rgba(255, 255, 255, 0.1)',
                              cursor: 'pointer',
                              display: 'flex',
                              alignItems: 'center',
                              justifyContent: 'center',
                            }}>
                              {hoveredCell?.i === i && hoveredCell?.j === j && (
                                <motion.div
                                  initial={{ opacity: 0, scale: 0.8 }}
                                  animate={{ opacity: 1, scale: 1 }}
                                  style={{
                                    position: 'absolute',
                                    top: '50%',
                                    left: '50%',
                                    transform: 'translate(-50%, -50%)',
                                    background: 'rgba(0, 0, 0, 0.9)',
                                    padding: '0.5rem 0.75rem',
                                    borderRadius: '6px',
                                    whiteSpace: 'nowrap',
                                    fontSize: '0.75rem',
                                    zIndex: 1000,
                                    pointerEvents: 'none',
                                    boxShadow: '0 4px 6px rgba(0, 0, 0, 0.3)',
                                  }}
                                >
                                  {value.toFixed(4)}
                                </motion.div>
                              )}
                            </div>
                          </motion.td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>

                {/* 凡例 */}
                <div style={{
                  marginTop: '1.5rem',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '1rem',
                }}>
                  <span style={{ fontSize: '0.875rem', color: '#94a3b8' }}>
                    {selectedMatrix === 'partial_correlation' ? '負の相関' : '弱い'}
                  </span>
                  <div style={{
                    width: '200px',
                    height: '20px',
                    background: selectedMatrix === 'partial_correlation'
                      ? 'linear-gradient(to right, rgba(37, 99, 235, 1), rgba(255, 255, 255, 0), rgba(220, 38, 38, 1))'
                      : 'linear-gradient(to right, rgba(168, 85, 247, 0), rgba(168, 85, 247, 1))',
                    borderRadius: '4px',
                  }} />
                  <span style={{ fontSize: '0.875rem', color: '#94a3b8' }}>
                    {selectedMatrix === 'partial_correlation' ? '正の相関' : '強い'}
                  </span>
                </div>
              </div>
            </div>
          </motion.div>
        )}
      </div>
    </div>
  );
};

export default Graphicallasso;
// export default GraphicalLassoVisualization;
