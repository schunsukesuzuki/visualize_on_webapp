import os
os.environ['XLA_FLAGS'] = '--xla_gpu_cuda_data_dir=/usr/local/cuda'
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['XLA_CLIENT_MEM_FRACTION'] = '0.8'  # 修正: XLA_CLIENT_MEM_FRACTION

import subprocess

# find path for CUDA
try:
    nvcc_path = subprocess.check_output(['which', 'nvcc']).decode().strip()
    cuda_path = os.path.dirname(os.path.dirname(nvcc_path))
    print(f"CUDAパス: {cuda_path}")
except:
    cuda_path = '/usr/local/cuda'  # default
    print(f"nvcc is not found, use CUDA: {cuda_path}")


from jax import grad, jit
import jax.numpy as jnp
from jax import random

# ========================================
# 1. Sharding configuration
# ========================================
import jax
print(f'Backend: {jax.default_backend()}')
print(f'Devices: {jax.devices()}')
devices = jax.devices()
from jax.sharding import PositionalSharding
sharding = PositionalSharding(devices) 

import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS

import numpy as np

import polars as pl

from tqdm import tqdm

from typing import Tuple, Optional, List
from dataclasses import dataclass

import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Noto Sans CJK JP'
import seaborn as sns

import matplotlib.pyplot as plt
import matplotlib as mpl

# confirm fonts
from matplotlib.font_manager import FontManager
fm = FontManager()
fonts = [f.name for f in fm.ttflist]
print([f for f in fonts if 'Noto' in f or 'Sans' in f][:10])

# uses Noto Sans CJK JP
plt.rcParams['font.family'] = 'Noto Sans CJK JP'


import time
from functools import wraps

import os
# GPU optimization configuration（running first）
os.environ['XLA_FLAGS'] = '--xla_gpu_cuda_data_dir=/usr/local/cuda'
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['XLA_CLIENT_MEM_FRACTION'] = '0.8'




# ========================================
# 2. preprocessing
# ========================================
@jax.jit
def preprocess(data):
    """log transformation + standardization"""
    data_transformed = jnp.log1p(data)
    mean = data_transformed.mean(axis=0, keepdims=True)
    std = data_transformed.std(axis=0, keepdims=True)
    std = jnp.where(std < 1e-6, 1.0, std)
    return (data_transformed - mean) / std



def make_jax_error_matrix_gpu():

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
    
    # transform DataFrame to jax.numpy
    jax_error_matrix = jnp.array(df_error_matrix.to_numpy())

    jax_error_matrix_float32 = jax_error_matrix.astype(jnp.float32)
    jax_error_matrix_float32_standardized = preprocess(jax_error_matrix_float32)
    jax_error_matrix_gpu = jax.device_put(jax_error_matrix_float32_standardized, sharding.replicate())

    print(f"shape: {jax_error_matrix_gpu.shape}")
    print(f"type: {jax_error_matrix_gpu.dtype}")
    print(f"device: {jax_error_matrix_gpu.devices()}")

    return jax_error_matrix_gpu


# jax_error_matrix_gpu = make_jax_error_matrix_gpu()



# ========================================
# 3. (optimized) model building
# ========================================
def lkj_correlation_model(data, eta=1.0):
    """
    LKJ相関行列推定モデル（最適化版）
    
    Parameters:
    -----------
    data : array (n_samples, n_features)
        標準化済みデータ
    eta : float
        LKJ濃度パラメータ（1=一様, >1=単位行列寄り）
    """
    n_samples, n_features = data.shape
    
    # Cholesky decomposition (for corr matrix)
    L_corr = numpyro.sample(
        "L_corr", 
        dist.LKJCholesky(n_features, concentration=eta)
    )
    
    # std（Exponential prior）
    sigma = numpyro.sample(
        "sigma", 
        dist.Exponential(1.0).expand([n_features])
    )
    
    # corr matrix
    corr_matrix = jnp.matmul(L_corr, L_corr.T)
    
    # cov matrix
    scale_matrix = jnp.diag(sigma)
    cov_matrix = scale_matrix @ corr_matrix @ scale_matrix
    cov_matrix = cov_matrix + jnp.eye(n_features) * 1e-3  # digit stabilization
    
    # average
    mu = numpyro.sample(
        "mu", 
        dist.Normal(0, 1).expand([n_features])
    )
    
    # likelihood
    with numpyro.plate("data", n_samples):
        numpyro.sample(
            "obs", 
            dist.MultivariateNormal(loc=mu, covariance_matrix=cov_matrix), 
            obs=data
        )


# ========================================
# 4. running MCMC
# ========================================
def mcmc_sampling():
    print("="*80)
    print("MCMC sampling")
    print("="*80)

    rng_key = random.PRNGKey(42)

    # NUTS configuration
    nuts_kernel = NUTS(
        lkj_correlation_model,
        target_accept_prob=0.9,
        max_tree_depth=10,
        init_strategy=numpyro.infer.init_to_median,
        regularize_mass_matrix=True
    )

    # MCMC configuration
    mcmc = MCMC(
        nuts_kernel,
        num_warmup=300,
        num_samples=700,
        num_chains=1,
        progress_bar=True
    )

    # running
    import time
    start_time = time.time()

    jax_error_matrix_gpu = make_jax_error_matrix_gpu()

    with jax.default_device(devices[0]):
        mcmc.run(rng_key, jax_error_matrix_gpu, eta=2.0)

    elapsed_time = time.time() - start_time

    print(f"\nrunning time: {elapsed_time:.2f}secs ({elapsed_time/60:.2f}mins)")
    print()

    return mcmc, elapsed_time


# ========================================
# 5. getting results and analysis
# ========================================
print("="*80)
print("results")
print("="*80)

mcmc, elapsed_time = mcmc_sampling()

samples = mcmc.get_samples()
L_corr_samples = samples["L_corr"]

# calcuratition for corr matrix
corr_matrices = jnp.matmul(
    L_corr_samples, 
    jnp.swapaxes(L_corr_samples, -2, -1)
)

# posterior stats
mean_corr = corr_matrices.mean(axis=0)
std_corr = corr_matrices.std(axis=0)
lower_ci = jnp.percentile(corr_matrices, 5, axis=0)
upper_ci = jnp.percentile(corr_matrices, 95, axis=0)

print(f"estimated corr matrix（posterior average）:")
print(mean_corr)
print()


# ========================================
# 6. diagnosis stats
# ========================================
print("="*80)
print("diagnosis stats")
print("="*80)
mcmc.print_summary(prob=0.9)
print()


# ========================================
# 7. 相関の要約統計
# ========================================
print("="*80)
print("相関係数の要約統計（対角要素除く）")
print("="*80)

off_diag_indices = jnp.triu_indices(mean_corr.shape[0], k=1)
off_diag_corrs = mean_corr[off_diag_indices]

print(f"平均: {off_diag_corrs.mean():.4f}")
print(f"中央値: {jnp.median(off_diag_corrs):.4f}")
print(f"最小: {off_diag_corrs.min():.4f}")
print(f"最大: {off_diag_corrs.max():.4f}")
print(f"標準偏差: {off_diag_corrs.std():.4f}")
print()

# 強い相関の数
strong_positive = (off_diag_corrs > 0.5).sum()
strong_negative = (off_diag_corrs < -0.5).sum()
total_pairs = len(off_diag_corrs)

print(f"強い正の相関 (r > 0.5): {strong_positive} / {total_pairs} ({strong_positive/total_pairs*100:.1f}%)")
print(f"強い負の相関 (r < -0.5): {strong_negative} / {total_pairs} ({strong_negative/total_pairs*100:.1f}%)")
print()


# ========================================
# 8. 結果の保存
# ========================================
results = {
    'mean_correlation': mean_corr,
    'std_correlation': std_corr,
    'lower_ci': lower_ci,
    'upper_ci': upper_ci,
    'samples': samples,
    'correlation_samples': corr_matrices,
    'execution_time': elapsed_time
}

print("="*80)
print("結果オブジェクト")
print("="*80)
print("results 辞書に以下が保存されています:")
print("  - mean_correlation: 事後平均相関行列")
print("  - std_correlation: 事後標準偏差")
print("  - lower_ci: 90%信用区間下限")
print("  - upper_ci: 90%信用区間上限")
print("  - samples: 全パラメータのMCMCサンプル")
print("  - correlation_samples: 相関行列のサンプル")
print()


'''
追加；学習後のMCMCサンプルからの分散共分散行列の計算
'''

# 次の関数の内部で使う関数
def compute_covariance_matrix(sigma, L_corr):
    """
    標準偏差とCholesky分解された相関行列から分散共分散行列を計算
    
    Parameters:
    -----------
    sigma : array (n_features,)
        標準偏差
    L_corr : array (n_features, n_features)
        相関行列のCholesky分解
    
    Returns:
    --------
    cov_matrix : array (n_features, n_features)
        分散共分散行列
    """
    # 相関行列を復元: R = L * L^T
    corr_matrix = jnp.matmul(L_corr, L_corr.T)
    
    # スケール行列(標準偏差の対角行列)
    scale_matrix = jnp.diag(sigma)
    
    # 分散共分散行列 = D * R * D 
    # (D: 標準偏差の対角行列, R: 相関行列)
    cov_matrix = scale_matrix @ corr_matrix @ scale_matrix
    
    return cov_matrix


def extract_covariance_from_mcmc_samples(samples):
    """
    MCMCサンプルから分散共分散行列の事後分布を計算
    
    Parameters:
    -----------
    samples : dict
        mcmc.get_samples()の結果
        必要なキー: 'sigma', 'L_corr'
    
    Returns:
    --------
    results : dict
        以下のキーを含む辞書:
        - mean_covariance: 事後平均分散共分散行列
        - std_covariance: 分散共分散行列の事後標準偏差
        - lower_ci_cov: 90%信用区間下限
        - upper_ci_cov: 90%信用区間上限
        - covariance_samples: 全サンプルの分散共分散行列
        - variances: 各特徴量の分散(対角要素)
    """
    
    # サンプルを取得
    sigma_samples = samples["sigma"]      # shape: (n_samples, n_features)
    L_corr_samples = samples["L_corr"]    # shape: (n_samples, n_features, n_features)
    
    print(f"サンプル数: {sigma_samples.shape[0]}")
    print(f"特徴量数: {sigma_samples.shape[1]}")
    
    # ベクトル化して全サンプルの分散共分散行列を計算
    print("\n分散共分散行列を計算中...")
    cov_matrices = jax.vmap(compute_covariance_matrix)(sigma_samples, L_corr_samples)
    print(f"計算完了: shape = {cov_matrices.shape}")
    
    # 事後統計量を計算
    mean_cov = cov_matrices.mean(axis=0)
    std_cov = cov_matrices.std(axis=0)
    lower_ci_cov = jnp.percentile(cov_matrices, 5, axis=0)
    upper_ci_cov = jnp.percentile(cov_matrices, 95, axis=0)
    
    # 分散(対角要素)を抽出
    variances = jnp.diagonal(mean_cov)
    
    # 結果を辞書にまとめる
    results = {
        'mean_covariance': mean_cov,
        'std_covariance': std_cov,
        'lower_ci_cov': lower_ci_cov,
        'upper_ci_cov': upper_ci_cov,
        'covariance_samples': cov_matrices,
        'variances': variances
    }
    
    return results


# visualize
def print_covariance_summary(results, show_heatmap=True, figsize=(16, 7), save_path=None):
# def print_covariance_summary(results):
    """
    分散共分散行列の結果を表示
    
    Parameters:
    -----------
    results : dict
        extract_covariance_from_mcmc_samples()の結果
    show_heatmap : bool
        ヒートマップを表示するかどうか (デフォルト: True)
    figsize : tuple
        図のサイズ (デフォルト: (16, 7))
    save_path : str or None
        ヒートマップの保存先パス (Noneの場合は保存しない)
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    print("\n" + "="*80)
    print("分散共分散行列の結果")
    print("="*80)
    
    mean_cov = results['mean_covariance']
    variances = results['variances']
    
    print(f"\n推定分散共分散行列(事後平均):")
    print(mean_cov)
    
    print(f"\n各特徴量の分散(対角要素):")
    print(variances)
    
    print(f"\n分散の要約統計量:")
    print(f"  平均: {variances.mean():.4f}")
    print(f"  中央値: {jnp.median(variances):.4f}")
    print(f"  最小: {variances.min():.4f}")
    print(f"  最大: {variances.max():.4f}")
    print(f"  標準偏差: {variances.std():.4f}")
    
    # 共分散(非対角要素)の統計
    n = mean_cov.shape[0]
    off_diag_indices = jnp.triu_indices(n, k=1)
    off_diag_covs = mean_cov[off_diag_indices]
    
    print(f"\n共分散(非対角要素)の要約統計量:")
    print(f"  平均: {off_diag_covs.mean():.4f}")
    print(f"  中央値: {jnp.median(off_diag_covs):.4f}")
    print(f"  最小: {off_diag_covs.min():.4f}")
    print(f"  最大: {off_diag_covs.max():.4f}")
    print(f"  標準偏差: {off_diag_covs.std():.4f}")
    
    # ヒートマップの表示
    if show_heatmap:
        print("\n" + "="*80)
        print("分散共分散行列のヒートマップ")
        print("="*80)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # 1. 分散共分散行列(事後平均)
        sns.heatmap(mean_cov, 
                    annot=False, 
                    cmap='RdBu_r', 
                    center=0, 
                    square=True, 
                    ax=ax1,
                    cbar_kws={'label': 'Covariance'})
        ax1.set_title('Covariance Matrix (Posterior Mean)', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Feature Index', fontsize=12)
        ax1.set_ylabel('Feature Index', fontsize=12)
        
        # 2. 分散共分散行列(事後標準偏差) - 不確実性
        if 'std_covariance' in results:
            std_cov = results['std_covariance']
            sns.heatmap(std_cov, 
                        annot=False, 
                        cmap='viridis', 
                        square=True, 
                        ax=ax2,
                        cbar_kws={'label': 'Standard Deviation'})
            ax2.set_title('Covariance Uncertainty (Posterior Std)', fontsize=14, fontweight='bold')
            ax2.set_xlabel('Feature Index', fontsize=12)
            ax2.set_ylabel('Feature Index', fontsize=12)
        else:
            # std_covarianceがない場合は分散のバープロットを表示
            ax2.bar(range(len(variances)), variances, edgecolor='black', alpha=0.7)
            ax2.set_xlabel('Feature Index', fontsize=12)
            ax2.set_ylabel('Variance', fontsize=12)
            ax2.set_title('Variance per Feature', fontsize=14, fontweight='bold')
            ax2.grid(alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"\nヒートマップを保存しました: {save_path}")
        
        plt.show()
        print("\nヒートマップを表示しました。")
    


# ========================================
# 使用例
# ========================================
if __name__ == "__main__":

    # 元のlkj.pyでMCMCサンプリングを実行した後、
    # 以下のようにして分散共分散行列を取得できます:
    
    # MCMCサンプリング後
    samples = mcmc.get_samples()
    
    # 分散共分散行列を計算
    # from covariance_calculation import extract_covariance_from_mcmc_samples, print_covariance_summary
    
    cov_results = extract_covariance_from_mcmc_samples(samples)
    print_covariance_summary(cov_results)
    
    # 結果を元のresults辞書に追加
    results.update(cov_results)
    
    # 使用方法:
    # - cov_results['mean_covariance']  # 事後平均分散共分散行列
    # - cov_results['variances']        # 各特徴量の分散
    # - cov_results['covariance_samples']  # 全サンプル(ベイズ推定の不確実性評価に使用)
    
    print(__doc__)




# ========================================
# 9. 可視化コード
# ========================================
# print("="*80)
# print("可視化（オプション）")
# print("="*80)
# print("""
# import matplotlib.pyplot as plt
# import seaborn as sns

# # 相関行列のヒートマップ
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

# # 事後平均
# sns.heatmap(results['mean_correlation'], annot=False, cmap='coolwarm', 
#             center=0, vmin=-1, vmax=1, square=True, ax=ax1,
#             cbar_kws={'label': 'Correlation'})
# ax1.set_title('Estimated Correlation Matrix (Posterior Mean)', fontsize=14)
# ax1.set_xlabel('Error Type', fontsize=12)
# ax1.set_ylabel('Error Type', fontsize=12)

# # 事後標準偏差
# sns.heatmap(results['std_correlation'], annot=False, cmap='viridis', 
#             square=True, ax=ax2, cbar_kws={'label': 'Standard Deviation'})
# ax2.set_title('Posterior Standard Deviation', fontsize=14)
# ax2.set_xlabel('Error Type', fontsize=12)
# ax2.set_ylabel('Error Type', fontsize=12)

# plt.tight_layout()
# plt.savefig('correlation_matrix.png', dpi=300, bbox_inches='tight')
# plt.show()

# # 相関係数の分布
# fig, ax = plt.subplots(figsize=(10, 6))
# ax.hist(off_diag_corrs, bins=50, edgecolor='black', alpha=0.7)
# ax.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero correlation')
# ax.set_xlabel('Correlation Coefficient', fontsize=12)
# ax.set_ylabel('Frequency', fontsize=12)
# ax.set_title('Distribution of Pairwise Correlations', fontsize=14)
# ax.legend()
# ax.grid(alpha=0.3)
# plt.tight_layout()
# plt.savefig('correlation_distribution.png', dpi=300, bbox_inches='tight')
# plt.show()
# """)

# print("\n処理完了！ 🎉")

# ```

# **パフォーマンス比較:**
# ```
# 実装                          実行時間        速度比
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PyMC (CPU)                    ~600秒 (10分)   1.0x
# JAX/NumPyro (CPU)             ~150秒 (2.5分)  4.0x
# JAX/NumPyro (GPU, 最適化前)   ~660秒 (11分)   0.9x ❌
# JAX/NumPyro (GPU, 最適化後)    21秒 (0.35分)  28.6x ✅