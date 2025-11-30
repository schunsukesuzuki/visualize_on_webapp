import { useState, useEffect } from 'react';

function Preprocess() {
  const [data, setData] = useState<Record<string, any>[]>([]);
  const [columns, setColumns] = useState<string[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [heatmapUrl, setHeatmapUrl] = useState<string>('');
  const [heatmapLoading, setHeatmapLoading] = useState(false);

  useEffect(() => {
    setLoading(true);
    fetch('http://localhost:8000/api/test_dataframe')
      .then(res => {
        if (!res.ok) {
          throw new Error(`HTTP error! status: ${res.status}`);
        }
        return res.json();
      })
      .then((data: Record<string, any>[]) => {
        setData(data);
        if (data.length > 0) {
          setColumns(Object.keys(data[0]));
        }
        setLoading(false);
      })
      .catch(err => {
        setError(err.message);
        setLoading(false);
      });
  }, []);

  const loadHeatmap = () => {
    setHeatmapLoading(true);
    // 画像URLを設定（タイムスタンプでキャッシュ回避）
    setHeatmapUrl(`http://localhost:8000/api/test_heatmap?t=${Date.now()}`);
    setHeatmapLoading(false);
  };

  if (loading) {
    return <div>読み込み中...</div>;
  }

  if (error) {
    return <div>エラー: {error}</div>;
  }

  return (
    <div style={{ padding: '20px' }}>
      <h1>エラーマトリックス</h1>
      <p>データ件数: index先頭から{data.length}件</p>
      
      {/* ヒートマップセクション */}
      <div style={{ marginBottom: '20px' }}>
        <button 
          onClick={loadHeatmap}
          style={{ 
            padding: '10px 20px', 
            fontSize: '16px',
            cursor: 'pointer',
            marginBottom: '10px'
          }}
        >
          ヒートマップを生成
        </button>
        
        {heatmapLoading && <p>ヒートマップを生成中...</p>}
        
        {heatmapUrl && (
          <div>
            <h2>共分散行列ヒートマップ</h2>
            <img 
              src={heatmapUrl} 
              alt="Covariance Heatmap" 
              style={{ maxWidth: '100%', height: 'auto' }}
              onLoad={() => setHeatmapLoading(false)}
            />
          </div>
        )}
      </div>

      {/* データテーブル */}
      <div style={{ overflowX: 'auto' }}>
        <table border={1} style={{ borderCollapse: 'collapse' }}>
          <thead>
            <tr>
              {columns.map((col) => (
                <th key={col} style={{ padding: '8px', backgroundColor: '#f0f0f0' }}>
                  {col}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {data.slice(0, 100).map((row, index) => (
              <tr key={index}>
                {columns.map((col) => (
                  <td key={col} style={{ padding: '8px' }}>
                    {row[col]}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {data.length > 100 && (
        <p>※ 最初の100件のみ表示しています</p>
      )}
    </div>
  );
}

export default Preprocess;
