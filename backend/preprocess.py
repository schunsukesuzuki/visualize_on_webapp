"""
preprocessing pipeline
for generating matrix as dataset
"""

import polars as pl
import time
from functools import wraps, partial
from typing import Optional, List, Dict

import polars as pl
import jax
import jax.numpy as jnp
from jax import random
from jax.experimental import mesh_utils
from jax.sharding import PositionalSharding, NamedSharding, Mesh, PartitionSpec as P

from tqdm import tqdm

from pathlib import Path

def timer(func):
    """making decorator to estimate time"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        print(f"{func.__name__}: {end - start:.4f}秒")
        return result
    return wrapper


@timer
def process_complete_pipeline(
    rawdata1_path: str = "../data/rawdata1.csv",
    rawdata2_path: str = "../data/rawdata2.csv",
    combined_path: str = "../data/df_combined_1.csv",
    drop_error_columns: Optional[List[str]] = None,
    display_results: bool = True
) -> Dict[str, pl.DataFrame]:
    """
    Parameters
    ----------
    routeruns_path : str, optional
        (default: "../data/rawdata1.csv")
    navstack_path : str, optional
        (default: "../data/rawdata2.csv")
    drop_error_columns : list of str, optional
        (default: None)
    display_results : bool, optional
        (default: True)
    
    Returns
    -------
    dict
        以下のキーを持つ辞書:
        - 'rawdata1': DataFrame
        - 'rawdata2': DataFrame
        - 'error_matrix_with_id': matrix including id
        - 'error_matrix': matrix
    
    Examples
    --------
    >>> results = process_complete_pipeline()
    >>> df_rawdata1 = results['rawdata1']
    >>> df_rawdata2 = results['rawdata2']
    >>> df_error_matrix = results['error_matrix']
    """
    if drop_error_columns is None:
        drop_error_columns = []
    
    # rawdata2
    df_rawdata2 = (
        pl.read_csv(navstack_path)
        .rename({
            "event_time": "event",
            "data.code": "code",
            "reason": "error_message"
        })
        .with_columns(
            pl.col('event').str.strptime(pl.Datetime(time_zone='UTC'), '%+')
        )
        .sort(['id', 'event'])
    )
    
    if display_results:
        print(f"len(df_rawdata2): {len(df_rawdata2)}")
        print('*' * 80)
        print(df_navstack.head(15))
    
    # rawdata1
    df_rawdata1 = (
        pl.read_csv(rawdata1_path)
        .rename({
            "starttime": "start",
            "endtime": "end"
        })
        .with_columns([
            pl.col('start').str.strptime(pl.Datetime(time_zone='UTC'), '%+'),
            pl.col('end').str.strptime(pl.Datetime(time_zone='UTC'), '%+')
        ])
        .sort(['id', 'start'])
    )
    
    if display_results:
        print(f"len(df_rawdata1): {len(df_rawdata1)}")
        print('*' * 80)
        print(df_rawdata1.head(15))
    
    # function to check time stamp range
    @partial(jax.jit, static_argnums=(3,))
    def compute_matches_sharded(starts, ends, events, use_sharding=True):
        """
        calcuration of JIT compiler and sharding for time range matching
        (using rawdata1 and rawdata2)
        
        Parameters
        ----------
        starts : jax.Array
        ends : jax.Array
        events : jax.Array
        use_sharding : bool
        
        Returns
        -------
        tuple
            (in_range, error_indices, session_indices)
        """
        # checking with broadcasting
        # events[:, None] shape: (n_events, 1)
        # starts[None, :] shape: (1, n_sessions)
        # 結果の shape: (n_events, n_sessions)
        in_range = (events[:, None] >= starts[None, :]) & \
                (events[:, None] <= ends[None, :])
        
        # getting matched range
        error_indices, session_indices = jnp.where(in_range)
        
        return in_range, error_indices, session_indices


    # @jax.jit
    def compute_matches_vmap(starts, ends, events):
        """
        optimized function with vmap
        
        Parameters
        ----------
        starts : jax.Array
        ends : jax.Array
        events : jax.Array
        
        Returns
        -------
        tuple
            (error_indices, session_indices)
        """
        def check_event(event):
            return (event >= starts) & (event <= ends)
        
        # vmap parallel vectorization
        in_range = jax.vmap(check_event)(events)
        
        error_indices, session_indices = jnp.where(in_range)
        return error_indices, session_indices


    def join_with_jax_vmap(
        df_rawdata1_1_0: pl.DataFrame,
        df_rawdata2_1_0: pl.DataFrame
    ) -> pl.DataFrame:
        """
        optimized function using vmap
        """
        df_rawdata1_1_sorted = df_rawdata1_1_0.sort(['id', 'start'])
        df_rawdata2_1_sorted = df_rawdata2_1_0.sort(['id', 'event'])

        unique_serials = df_rawdata1_1_0['id'].unique().sort()
        result_rows = []
        
        # function for comparison
        @jax.jit
        def compute_in_range(starts, ends, events):
            def check_event(event):
                return (event >= starts) & (event <= ends)
            return jax.vmap(check_event)(events)
        
        for serial in tqdm(unique_serials, desc="Processing serials (vmap)"):
            routeruns_filtered = df_routeruns_1_sorted.filter(pl.col('id') == serial)
            navstack_filtered = df_navstack_1_sorted.filter(pl.col('id') == serial)
            
            if len(navstack_filtered) == 0:
                continue
            
            starts = jnp.array(routeruns_filtered['start'].cast(pl.Int64).to_numpy())
            ends = jnp.array(routeruns_filtered['end'].cast(pl.Int64).to_numpy())
            events = jnp.array(navstack_filtered['event'].cast(pl.Int64).to_numpy())
            
            # calcuration of bool array with JIT-compiled function
            in_range = compute_in_range(starts, ends, events)
            
            # getting indices with CPU to Numpy array 
            error_indices, session_indices = jnp.where(in_range)
            error_indices_cpu = jnp.array(error_indices, dtype=jnp.int64)
            session_indices_cpu = jnp.array(session_indices, dtype=jnp.int64)
            
            # transform polars dataframe to list
            routeruns_starts = routeruns_filtered['start'].to_list()
            routeruns_ends = routeruns_filtered['end'].to_list()
            navstack_codes = navstack_filtered['code'].to_list()
            navstack_reasons = navstack_filtered['reason'].to_list()
            navstack_events = navstack_filtered['event'].to_list()
            
            # summarize results
            for err_idx, sess_idx in zip(error_indices_cpu, session_indices_cpu):
                result_rows.append({
                    'id': serial,
                    'start': routeruns_starts[int(sess_idx)],
                    'end': routeruns_ends[int(sess_idx)],
                    'error_code': navstack_codes[int(err_idx)],
                    'error_message': navstack_reasons[int(err_idx)],
                    'event': navstack_events[int(err_idx)]
                })
        
        print('*' * 80)
        print('【join with jax vmap】')
        result_df = pl.DataFrame(result_rows) if result_rows else pl.DataFrame()
        print(result_df.head(15))
        
        return result_df
    

    # simple confirmation
    if Path(combined_path).exists():
        print("【****file is existed****】")
        df_combined_1 = pl.read_csv(combined_path)
        print('【****load complete****】')
        print(df_combined_1.head(15))
    else:
        print("【**** generating error connt dataset ****】")
        df_combined_1 = join_with_jax_vmap(df_rawdata1, df_rawdata2)
        df_combined_1.write_csv(combined_path, separator=",")

    def create_error_count_matrix(df_combined_1: pl.DataFrame) -> pl.DataFrame:
        """
        generating count matrix as
        id × error_code 
        """
        # count error code
        df_error_counts_0 = (
            df_combined_1
            .group_by(['serial_no', 'error_code'])
            .agg(pl.count().alias('count'))
            .pivot(
                values='count',
                index='id',
                columns='error_code',
                aggregate_function='sum'
            )
            .fill_null(0)  # fill NULL with 0
            .sort('id')
        )

        return df_error_counts_0
    
    df_error_counts_0 = create_error_count_matrix(df_combined_1)
    
    if display_results:
        print(f"number of columns（before selection）: {len(df_error_counts_0.columns)}")
        print('*' * 80)

    # exclude unnecessary columns
    df_error_matrix = df_error_counts_0.drop([])
    
    if drop_error_columns:
        columns_to_keep = [
            col for col in df_error_matrix.columns 
            if col == 'id' or col not in drop_error_columns
        ]
        df_error_matrix_filtered = df_error_matrix.select(columns_to_keep)
    else:
        df_error_matrix_filtered = df_error_matrix
    
    if display_results:
        print(f"number of columns(after selection）: {len(df_error_matrix_filtered.columns)}")
        print('*' * 80)
        print(df_error_matrix_filtered.head(15))
    
    # final results as matrix excluded id column 
    df_error_matrix_final = df_error_matrix_filtered.drop('id')
    
    if display_results:
        print(f"number of columns as result: {len(df_error_matrix_final.columns)}")
        print('*' * 80)
        print(df_error_matrix_final.head(15))
    
    return {
        'routeruns': df_routeruns,
        'navstack': df_navstack,
        'error_matrix_with_serial': df_error_matrix_filtered,
        'error_matrix': df_error_matrix_final
    }


def plot_covariance_heatmap():
    """
    displaying heatmap of cov matrix
    """
    from preprocess import process_complete_pipeline

    # from fastapi.responses import StreamingResponse
    from io import BytesIO
    import matplotlib
    matplotlib.use('Agg')  # for environment without GUI
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    results = process_complete_pipeline(
        rawdata1_path="../data/rawdata1.csv",
        rawdata2_path="../data/rawdata2.csv",
        combined_path="../data/df_combined_1.csv",
        display_results=False
    )
    
    df_error_matrix = results['error_matrix']
    
    data_array = df_error_matrix.to_numpy()
    
    cov_matrix = jnp.cov(data_array.T)
    
    # getting error names
    error_code_names = df_error_matrix.columns
    
    # generating heatmap
    plt.figure(figsize=(14, 12))
    sns.heatmap(
        cov_matrix,
        xticklabels=error_code_names,
        yticklabels=error_code_names,
        annot=True,
        fmt='.2f',
        cmap='coolwarm',
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={'label': 'covariance'}
    )
    
    plt.title("inter-error covariance matrix", fontsize=16, pad=20)
    plt.xlabel('error name', fontsize=12)
    plt.ylabel('error name', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    # save image to byte stream
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    plt.close()

    return buf


if __name__ == "__main__":
    # usage
    results = process_complete_pipeline()
    df_rawdata1 = results['rawdata1']
    df_rawdata2 = results['rawdata2']
    df_error_matrix = results['error_matrix']
    
    print("\n preprocessing completed!")
    print(f"rawdata1: {df_rawdata1.shape}")
    print(f"rawdata2: {df_rawdata2.shape}")
    print(f"inter-error matrix: {df_error_matrix.shape}")
