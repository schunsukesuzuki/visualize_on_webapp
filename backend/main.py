from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, ConfigDict
import jax.numpy as jnp
from model import LinearRegressionJAX, load_and_train_model
import os

# added for displaying images
from fastapi.responses import StreamingResponse
from io import BytesIO
import matplotlib
matplotlib.use('Agg')  # for environment without GUI
import matplotlib.pyplot as plt
import seaborn as sns


# for displaying graphical lasso 
from typing import List, Optional
import numpy as np
from graphical_lasso import get_results_from_graphical_lasso, prepare_data, GraphicalLassoModel, GraphicalLassoConfig


# added for LKJ
from typing import Dict, Any
import sys
import base64
import json


app = FastAPI(title="covariance prediction API")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class TrainingConfig(BaseModel):
    model_config = ConfigDict(protected_namespaces=())  # mute warning from Pydantic v2
    
    data_path_routeruns: str = Field(
        default="../data/routeruns_v1.csv",
        description="Path to the routeruns CSV file"
    )
    data_path_navstack: str = Field(
        default="../data/webhooknavstack_v1.csv",
        description="Path to the navstack CSV file"
    )
    data_combined_path: str = Field(
        default="../data/df_combined_1.csv",
        description="Path to save combined CSV file"
    )

    alpha: float = Field(default=0.01, description="Regularization parameter for Graphical Lasso")
    rho: float = Field(default=1.0, description="penalty intensity by Augmented Lagrangian method")
    max_iter: int = Field(default=100, description="Maximum number of iterations")
    tol: float = Field(default=1e-4, description="Tolerance for convergence")
    test_size: float = Field(default=0.2, description="Test data ratio (0.0-1.0)")
    random_seed: int = Field(default=42, description="Random seed for reproducibility")


# global variable
model: LinearRegressionJAX = None


class PredictionRequest(BaseModel):
    visitors: float = Field(..., gt=0, description="number of visitors")


class PredictionResponse(BaseModel):
    visitors: float
    predicted_sales: float
    model_slope: float
    model_intercept: float
    r2_score: float


class ModelInfo(BaseModel):
    slope: float
    intercept: float
    is_fitted: bool
    equation: str


# @app.on_event("startup")
# async def startup_event():
#     """training model when boot App """
#     global model
    
#     data_path = "../data/sales_data.csv"
    
#     # If the data file does not exist, create it.    
#     if not os.path.exists(data_path):
#         print("データファイルが存在しないため、生成します...")
#         from generate_data import generate_sales_data
#         df = generate_sales_data()
#         os.makedirs("../data", exist_ok=True)
#         df.write_csv(data_path)
#         print("data genaration completed")
    
#     print("training model...")
#     model = load_and_train_model(data_path)
#     print("training completed")


@app.get("/")
async def root():
    """health check"""
    return {"message": "Sales Prediction API is running", "status": "ok"}


# @app.get("/model/info", response_model=ModelInfo)
# async def get_model_info():
#     """getting model info"""
#     if model is None or not model.is_fitted:
#         raise HTTPException(status_code=500, detail="model is not trained")
    
#     equation = f"sales = {model.slope:.2f} × visitors + {model.intercept:.2f}"
    
#     return ModelInfo(
#         slope=model.slope,
#         intercept=model.intercept,
#         is_fitted=model.is_fitted,
#         equation=equation
#     )


@app.post("/predict", response_model=PredictionResponse)
async def predict_sales(request: PredictionRequest):
    """来客者数から売上高を予測"""
    if model is None or not model.is_fitted:
        raise HTTPException(status_code=500, detail="モデルがまだ学習されていません")
    
    try:
        # run to predict
        visitors_array = jnp.array([request.visitors])
        predicted_sales = model.predict(visitors_array)[0]
        
        # calcuration of R² score
        import polars as pl
        df = pl.read_csv("../data/sales_data.csv")
        X = jnp.array(df["visitors"].to_numpy())
        y = jnp.array(df["sales"].to_numpy())
        r2 = model.score(X, y)
        
        return PredictionResponse(
            visitors=request.visitors,
            predicted_sales=float(predicted_sales),
            model_slope=model.slope,
            model_intercept=model.intercept,
            r2_score=r2
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"prediction error: {str(e)}")


@app.get("/api/test_dataframe")
def get_dataframe(config: TrainingConfig = TrainingConfig()):
    from preprocess import process_complete_pipeline
    import os
    if not os.path.exists(config.data_path_routeruns):
        raise HTTPException(status_code=404, detail=f"File not found: {config.data_path_rawdata1}")
    if not os.path.exists(config.data_path_navstack):
        raise HTTPException(status_code=404, detail=f"File not found: {config.data_path_rawdata2}")

    results = process_complete_pipeline(
        rawdata1_path=config.data_path_rawdata1,
        rawdata2_path=config.data_path_rawdata2,
        combined_path=config.data_combined_path
    )
    df_rawdata1 = results['rawdata1']
    df_rawdata2 = results['rawdata2']
    df_error_matrix = results['error_matrix']

    # transform Polars DataFrame to JSON
    return df_error_matrix.head(10).to_dicts()  


@app.get("/api/test_heatmap")
def get_heatmap():

    from preprocess import plot_covariance_heatmap
    buf = plot_covariance_heatmap()
    
    # return images
    return StreamingResponse(buf, media_type="image/png")


'''
Graphical Lasso 
'''
class GraphicalLassoResponse(BaseModel):
    """ results of Graphical Lasso"""
    covariance_matrix: List[List[float]]
    precision_matrix: List[List[float]]
    feature_names: List[str]
    partial_correlations: List[List[float]]
    config: dict
    convergence_info: Optional[dict] = None

class GraphicalLassoRequest(BaseModel):
    """requests of Graphical Lasso"""
    alpha: Optional[float] = 0.01
    max_iter: Optional[int] = 100
    tol: Optional[float] = 1e-4
    rho: Optional[float] = 1.0


# # @app.get("/api/test_graphical_lasso_results")
# def get_graphical_lasso_results():

#     from graphical_lasso import get_results_from_graphical_lasso

#     graphical_lasso_results_matrix_train = get_results_from_graphical_lasso()


@app.post("/api/graphical-lasso", response_model=GraphicalLassoResponse)
async def run_graphical_lasso(request: GraphicalLassoRequest):
    """
    Graphical Lassoを実行して結果を返す
    
    Parameters:
    -----------
    request : GraphicalLassoRequest
        parameters - alpha, max_iter, tol, rho
    
    Returns:
    --------
    GraphicalLassoResponse
        cov matrix, precision matrix, feature names, partial corr matrix
    """
    try:
        from preprocess import process_complete_pipeline
        
        # getting data
        results = process_complete_pipeline(
            rawdata1_path="../data/rawdata1.csv",
            rawdata2_path="../data/rawdata2.csv",
            combined_path="../data/df_combined_1.csv"          
            display_results=False
        )
        
        df_error_matrix = results['error_matrix']
        error_code_names = df_error_matrix.columns
        
        from graphical_lasso import get_results_from_graphical_lasso, prepare_data, GraphicalLassoModel, GraphicalLassoConfig

        # getting data
        X_train, X_test, df_train, df_test = prepare_data(
            df_error_matrix, 
            error_code_names
        )

        # model configuration
        config = GraphicalLassoConfig(
            alpha=request.alpha,
            max_iter=request.max_iter,
            tol=request.tol,
            rho=request.rho
        )
        
        # training model
        model = GraphicalLassoModel(config)
        model.fit(X_train, verbose=False)
        
        # getting results
        covariance_matrix = model.covariance_matrix
        precision_matrix = model.precision_matrix
        partial_correlations = model.get_partial_correlations()
        
        # NumPy to list
        covariance_list = np.array(covariance_matrix).tolist()
        precision_list = np.array(precision_matrix).tolist()
        partial_corr_list = np.array(partial_correlations).tolist()
        
        # convergence info
        convergence_info = None
        if model.convergence_history:
            last_iteration = model.convergence_history[-1]
            convergence_info = {
                "total_iterations": len(model.convergence_history),
                "final_primal_residual": last_iteration['primal_residual'],
                "final_dual_residual": last_iteration['dual_residual'],
                "converged": last_iteration['primal_residual'] < config.tol
            }
        
        return GraphicalLassoResponse(
            covariance_matrix=covariance_list,
            precision_matrix=precision_list,
            feature_names=list(error_code_names),
            partial_correlations=partial_corr_list,
            config={
                "alpha": config.alpha,
                "max_iter": config.max_iter,
                "tol": config.tol,
                "rho": config.rho
            },
            convergence_info=convergence_info
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"error occurred: {str(e)}")


'''
LKJ
'''
# global variable for cash
cached_results = None
cached_samples = None


def generate_covariance_plot(results: Dict[str, Any]) -> bytes:
    '''generating heatmap of cov matrix and return as byte array'''
    import jax.numpy as jnp
    
    mean_cov = results['mean_covariance']
    figsize = (16, 7)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # 1. cov matrix(posterior average)
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
    
    # 2. cov matrix(posterior std) - uncertainty
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
        variances = results['variances']
        ax2.bar(range(len(variances)), variances, edgecolor='black', alpha=0.7)
        ax2.set_xlabel('Feature Index', fontsize=12)
        ax2.set_ylabel('Variance', fontsize=12)
        ax2.set_title('Variance per Feature', fontsize=14, fontweight='bold')
        ax2.grid(alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # transform to byte array
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    image_bytes = buf.read()
    plt.close(fig)
    
    return image_bytes


@app.post("/api/lkj/run-mcmc")
async def run_mcmc():
    """run MCMC sampling"""

    from lkj import (
    mcmc_sampling,
    extract_covariance_from_mcmc_samples,
    print_covariance_summary
    )  

    global cached_results, cached_samples
    
    try:
        print("start MCMC sampling...")
        mcmc, elapsed_time = mcmc_sampling()
        samples = mcmc.get_samples()
        
        # save to cash 
        cached_samples = samples
        
        return {
            "status": "success",
            "message": "MCMC sampling completed",
            "elapsed_time": elapsed_time,
            "elapsed_time_minutes": elapsed_time / 60
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"MCMC sampling error: {str(e)}")


@app.get("/api/lkj/covariance-summary")
async def get_covariance_summary():
    '''getting summarized stats of cov matrix'''

    from lkj import (
    mcmc_sampling,
    extract_covariance_from_mcmc_samples,
    print_covariance_summary
    )

    global cached_results, cached_samples
    
    if cached_samples is None:
        raise HTTPException(
            status_code=400, 
            detail="MCMC is not run. please run  /api/lkj/run-mcmc in advance"
        )
    
    try:
        import jax.numpy as jnp
        
        # calcuratiing cov matrix
        cov_results = extract_covariance_from_mcmc_samples(cached_samples)
        cached_results = cov_results
        
        mean_cov = cov_results['mean_covariance']
        variances = cov_results['variances']
        
        # stats of cov matrix(non diagonal factors)
        n = mean_cov.shape[0]
        off_diag_indices = jnp.triu_indices(n, k=1)
        off_diag_covs = mean_cov[off_diag_indices]
        
        # NumPy to Python native type
        def to_python(val):
            if hasattr(val, 'item'):
                return val.item()
            return float(val)
        
        summary = {
            "matrix_shape": list(mean_cov.shape),
            "variance_statistics": {
                "mean": to_python(variances.mean()),
                "median": to_python(jnp.median(variances)),
                "min": to_python(variances.min()),
                "max": to_python(variances.max()),
                "std": to_python(variances.std())
            },
            "covariance_statistics": {
                "mean": to_python(off_diag_covs.mean()),
                "median": to_python(jnp.median(off_diag_covs)),
                "min": to_python(off_diag_covs.min()),
                "max": to_python(off_diag_covs.max()),
                "std": to_python(off_diag_covs.std())
            },
            "variances": [to_python(v) for v in variances],
            "mean_covariance_matrix": [[to_python(val) for val in row] for row in mean_cov]
        }
        
        return summary
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"cov matrix calcuration error: {str(e)}")


@app.get("/api/lkj/covariance-heatmap")
async def get_covariance_heatmap():
    """getting heatmap images of cov matrix"""
    global cached_results
    
    if cached_results is None:
        raise HTTPException(
            status_code=400,
            detail="cov matrix is not calcurated. please run /api/lkj/covariance-summary in advance"
        )
    
    try:
        image_bytes = generate_covariance_plot(cached_results)
        return StreamingResponse(BytesIO(image_bytes), media_type="image/png")
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"ヒートマップ生成エラー: {str(e)}")


@app.get("/api/lkj/covariance-heatmap-base64")
async def get_covariance_heatmap_base64():
    """getting heatmap images of cov matrix as BASE64 encoded"""
    global cached_results
    
    if cached_results is None:
        raise HTTPException(
            status_code=400,
            detail="cov matrix is not calcurated. please run /api/lkj/covariance-summary in advance"
        )
    
    try:
        image_bytes = generate_covariance_plot(cached_results)
        base64_image = base64.b64encode(image_bytes).decode('utf-8')
        
        return {
            "image": f"data:image/png;base64,{base64_image}"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"heatmap generation error: {str(e)}")


@app.delete("/api/lkj/clear-cache")
async def clear_cache():
    """sweeped cash"""
    global cached_results, cached_samples
    
    cached_results = None
    cached_samples = None
    
    return {"status": "success", "message": "sweeped cash"}




if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
