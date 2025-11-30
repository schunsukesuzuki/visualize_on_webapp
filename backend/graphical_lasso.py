import jax
import jax.numpy as jnp
from jax import grad, jit
import polars as pl
import numpy as np
from typing import Tuple, Optional, List
from dataclasses import dataclass
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Hiragino Sans'
import seaborn as sns


@dataclass
class GraphicalLassoConfig:
    """Graphical Lassoの設定


    L_ρ(Θ, Z, U) = -log det(Θ) + trace(SΘ) + α||Z||₁ + (ρ/2)||Θ - Z + U||²_F
                                                              ↑
                                                          この係数がrho
    alpha: 制御 parameter
    1. alpha が小さい（0.01）
    lambda_val = 0.01 / 1.0 = 0.01
    → 多くの要素が残る（密なグラフ）
    → 次元削減が少ない

    2. alpha が大きい（0.5）
    lambda_val = 0.5 / 1.0 = 0.5
    → 多くの要素がゼロになる（疎なグラフ）
    → 次元削減が大きい


    rho：拡張ラグランジュ法におけるこのペナルティの強さ

    1. rhoが小さい場合（例：rho = 0.1）

    制約違反に対するペナルティが弱い
    Θ と Z が離れていても許容される
    収束が遅くなる
    より多くの反復が必要

    2. rhoが大きい場合（例：rho = 100）

    制約違反に対するペナルティが強い
    Θ と Z を強制的に近づける
    収束は早いが数値的に不安定になる可能性
    条件数（condition number）が悪化

    なぜ rho = 1.0 がデフォルトなのか
    理論的理由

    1. スケール中立性
      1-1. 問題が適切に正規化されていれば、rho=1.0 は自然なスケール
      1-2. 標本共分散行列 S の要素が O(1) の場合、rho=1.0 が適切

    2. 収束保証
      2-1. ADMMの収束理論では、rho > 0 であれば任意の正の値で収束
      2-2. rho=1.0 は中庸で安全な選択

    3. 実装の簡便性
      3-1. 係数が1なので計算が簡潔
      3-2. デバッグしやすい
    """
    alpha: float = 0.01
    max_iter: int = 100
    tol: float = 1e-4
    rho: float = 1.0

class GraphicalLassoModel:
    """Graphical Lassoモデル（改良版）"""

    def __init__(self, config: GraphicalLassoConfig):
        self.config = config
        self.precision_matrix = None
        self.covariance_matrix = None
        self.convergence_history = []

    @staticmethod
    @jit
    def soft_threshold(x: jnp.ndarray, lambda_val: float) -> jnp.ndarray:
        """ソフト閾値処理

        これは**L1正則化による要素の選択（スパース化）**を実現します
                  ⎧ x - λ    if x > λ
        S_λ(x) = ⎨ 0        if |x| ≤ λ
                  ⎩ x + λ    if x < -λ
        """
        return jnp.sign(x) * jnp.maximum(jnp.abs(x) - lambda_val, 0.0)

    '''
    以下では入力がjnp.ndarrayになってますが、
    これは訓練データであるdataframeがあったとして(行がサンプル数、列が変数)、
    そのまま2次元配列に変換したもので良い
    '''
    @staticmethod
    @jit
    def compute_sample_covariance(X: jnp.ndarray) -> jnp.ndarray:
        """標本共分散行列の計算"""
        n_samples = X.shape[0]
        X_centered = X - jnp.mean(X, axis=0, keepdims=True)
        return (X_centered.T @ X_centered) / n_samples

    def fit(self, X: jnp.ndarray, verbose: bool = True) -> 'GraphicalLassoModel':
        """Graphical Lassoの学習

        minimize -log det(Θ) + trace(SΘ) + λ||Θ||₁
        subject to Θ ≻ 0

        Θ は精度行列（precision matrix、分散共分散行列の逆行列）
        S は標本分散共分散行列
        ||Θ||₁ は非対角要素の絶対値の和

        ADMMを適用できるのは、以下の性質があるためです：

        1. 凸最適化問題：目的関数が凸関数
        2. 分離可能な構造：問題を複数の部分問題に分解できる
          2-1. 滑らかな項（対数行列式 + トレース）
          2-2. 非滑らかな項（L1ペナルティ）
        3. 制約条件の扱い：正定値対称行列という制約をADMMの枠組みで扱える
          3-1. 推定したい精度行列Θは正定値対称行列（positive definite symmetric matrix）
          3-2. 正定値なので当然正則（逆行列が存在）
          3-3. この性質により、ADMMの各ステップで必要な行列演算（固有値分解など）が安定に計算できる

        なお近年では、ヒルベルト空間やより一般的な距離空間への拡張も研究されていますが、
        標準的なADMM/拡張ラグランジュ法はユークリッド空間での理論と実装が基本です。

        Parameters:
        -----------
        X : jnp.ndarray, shape (n_samples, n_features)
            訓練データ

        Returns:
        --------
        self : GraphicalLassoModel
        """


        S = self.compute_sample_covariance(X)
        n_features = S.shape[0]

        # 初期化
        Theta = jnp.eye(n_features)
        Z = jnp.eye(n_features)
        Z_prev = jnp.eye(n_features)
        U = jnp.zeros((n_features, n_features))

        alpha = self.config.alpha
        rho = self.config.rho

        self.convergence_history = []

        for iteration in range(self.config.max_iter):
            # Theta の更新
            # v5
            # ===============================
            # Theta 更新（正しい方法）
            # ===============================
            # minimize -log det(Θ) + (1/2) * trace(SΘ) + (rho/2)||Θ - Z + U||²_F

            # A = Z - U を対称化
            A = (Z - U + (Z - U).T) / 2

            # (rho * A - S) の固有値分解
            # 注意: 元のコードでは S と A の関係が逆だった！
            Q, Lambda = jnp.linalg.eigh(rho * A - S)

            # Theta の固有値を計算（解析解）
            # λ_Θ = (Q + sqrt(Q² + 4*rho)) / (2*rho)
            Theta_eigvals = (Q + jnp.sqrt(Q**2 + 4*rho)) / (2*rho)

            # Theta を再構成（固有ベクトルは A-S と同じ）
            eigvecs = jnp.linalg.eigh(rho * A - S)[1]
            Theta = eigvecs @ jnp.diag(Theta_eigvals) @ eigvecs.T

            # 対称化
            Theta = (Theta + Theta.T) / 2

            # ===============================
            # Z 更新
            # ===============================
            Z_old = Z

            Theta_hat = Theta + U

            # ソフト閾値処理（非対角要素のみ）
            Z = jnp.sign(Theta_hat) * jnp.maximum(jnp.abs(Theta_hat) - alpha/rho, 0)

            # 対角要素は閾値処理しない
            Z = Z.at[jnp.diag_indices(n_features)].set(jnp.diag(Theta_hat))

            # 対称化
            Z = (Z + Z.T) / 2

            # ===============================
            # U 更新
            # ===============================
            U = U + Theta - Z


            # 収束判定
            primal_residual = float(jnp.linalg.norm(Theta - Z))
            dual_residual = float(rho * jnp.linalg.norm(Z - Z_prev))

            self.convergence_history.append({
                'iteration': iteration,
                'primal_residual': primal_residual,
                'dual_residual': dual_residual
            })

            if primal_residual < self.config.tol and iteration > 5:
                if verbose:
                    print(f"収束しました（反復回数: {iteration + 1}）")
                break

        self.precision_matrix = (Z + Z.T) / 2

        eigvals = jnp.linalg.eigvalsh(self.precision_matrix)
        if jnp.any(eigvals <= 0):
            print(f"警告: 精度行列が正定値でありません。最小固有値: {jnp.min(eigvals)}")
            # 正則化を追加
            epsilon = 1e-6 - jnp.min(eigvals) if jnp.min(eigvals) < 1e-6 else 0
            self.precision_matrix = self.precision_matrix + epsilon * jnp.eye(n_features)

        try:
            self.covariance_matrix = jnp.linalg.inv(self.precision_matrix)
        except:
            self.covariance_matrix = jnp.linalg.pinv(self.precision_matrix)

        return self

    def score(self, X: jnp.ndarray) -> float:
        """対数尤度の計算"""
        if self.precision_matrix is None:
            raise ValueError("モデルが学習されていません")

        S_test = self.compute_sample_covariance(X)
        sign, logdet = jnp.linalg.slogdet(self.precision_matrix)
        log_likelihood = logdet - jnp.trace(S_test @ self.precision_matrix)

        return float(log_likelihood)

    def get_partial_correlations(self) -> jnp.ndarray:
        """偏相関係数行列の計算"""
        if self.precision_matrix is None:
            raise ValueError("モデルが学習されていません")

        # 偏相関 = -Theta_ij / sqrt(Theta_ii * Theta_jj)
        diag = jnp.diag(self.precision_matrix)
        D = jnp.sqrt(jnp.outer(diag, diag))
        partial_corr = -self.precision_matrix / D
        partial_corr = partial_corr.at[jnp.diag_indices_from(partial_corr)].set(1.0)

        return partial_corr
    

def prepare_data(
    df: pl.DataFrame,
    error_code_columns: list,
    test_size: float = 0.2,
    random_seed: int = 42
) -> Tuple[jnp.ndarray, jnp.ndarray, pl.DataFrame, pl.DataFrame]:
    """prepare and split data"""
    np.random.seed(random_seed)
    n_samples = df.height
    indices = np.random.permutation(n_samples)

    n_train = int(n_samples * (1 - test_size))
    train_indices = indices[:n_train]
    test_indices = indices[n_train:]

    df_train = df[train_indices]
    df_test = df[test_indices]

    X_train = jnp.array(df_train.select(error_code_columns).to_numpy(), dtype=jnp.float32)
    X_test = jnp.array(df_test.select(error_code_columns).to_numpy(), dtype=jnp.float32)

    print(f"train data: {X_train.shape}")
    print(f"test data: {X_test.shape}")

    return X_train, X_test, df_train, df_test


def get_results_from_graphical_lasso():
    """
    getting results info after training graphical lasso
    """
    from preprocess import process_complete_pipeline

    # from fastapi.responses import StreamingResponse
    from io import BytesIO
    import matplotlib
    matplotlib.use('Agg')  # for environment without GUI
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # getting dataset via somewhat preprocessing
    results = process_complete_pipeline(
        rawdata1_path="../data/rawdata1.csv",
        rawdata2_path="../data/rawdata2.csv",
        combined_path="../data/results.csv",
        display_results=False
    )
    
    df_error_matrix = results['error_matrix']
    
    # DataFrameをnumpy配列に変換
    data_array = df_error_matrix.to_numpy()
    
    # 共分散行列を計算
    cov_matrix = jnp.cov(data_array.T)
    
    # エラーコード名を取得
    error_code_names = df_error_matrix.columns
    

    # error_code_columns = df_error_counts_2_2.columns
    X_train, X_test, df_train, df_test = prepare_data(df_error_matrix, error_code_names)

    graphicalLassoConfig= GraphicalLassoConfig() 
    graphicalLassoModel = GraphicalLassoModel(graphicalLassoConfig)
    graphicalLassoModel.fit(X_train)

    graphical_lasso_results_matrix_train = graphicalLassoModel.covariance_matrix

    print(graphical_lasso_results_matrix_train)

    return graphical_lasso_results_matrix_train