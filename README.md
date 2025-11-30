
- Note: this app is prototype version. not sophisticated. 
- This has only been tested in a local environment with RTX4060 windows PC. 
- Please be aware of this when using it as a reference for cloud computing.
- If you find any errors in the algorithm, please let me know.

# Covariance Matrix Estimation and Visualization Web Application

This project is a web application that estimates and visualizes covariance matrices using Graphical Lasso and Bayesian estimation with LKJ prior distribution.

## 📋 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Tech Stack](#tech-stack)
- [System Architecture](#system-architecture)
- [Setup](#setup)
- [Usage](#usage)
- [API Endpoints](#api-endpoints)
- [Algorithm Details](#algorithm-details)
- [Directory Structure](#directory-structure)

## 🎯 Overview

This application estimates covariance matrices from error matrices generated from route execution data and navigation stack data using two different approaches:

1. **Graphical Lasso**: Discovers conditional independence between variables through sparse precision matrix estimation
2. **LKJ Prior Bayesian Estimation**: Estimates correlation matrices using a Bayesian approach with uncertainty quantification

## ✨ Key Features

### Data Preprocessing
- CSV file loading and time-series data processing
- Matching route execution times with error events
- High-speed time range calculation using JAX (GPU-enabled)
- Error matrix generation

### Graphical Lasso
- Optimization using ADMM (Alternating Direction Method of Multipliers)
- Sparse precision matrix estimation
- Partial correlation coefficient calculation
- Convergence history tracking
- Heatmap visualization

### LKJ Bayesian Estimation
- MCMC sampling using NumPyro (NUTS)
- Correlation matrix estimation with LKJCholesky prior
- Posterior distribution statistics (mean, standard deviation, credible intervals)
- Uncertainty quantification of covariance matrices

### Frontend
- SPA built with React + TypeScript
- Real-time parameter adjustment
- Interactive heatmap display
- Multi-page feature separation

## 🛠 Tech Stack

### Test environment
- i7
- RTX4060
- wsl / ubuntu 22.04
- windows PC

### Backend
- **Language**: Python 3.x
- **Framework**: FastAPI
- **Scientific Computing**:
  - JAX / JAXLib (GPU acceleration)
  - NumPy / Polars (data processing)
  - NumPyro (Bayesian estimation)
  - SciPy
- **Visualization**: Matplotlib, Seaborn

### Frontend
- **Language**: TypeScript
- **Framework**: React 18
- **Routing**: React Router DOM
- **Animation**: Framer Motion
- **Build Tool**: Vite

### Infrastructure
- **Container**: Docker / Docker Compose
- **Ports**: 
  - Backend: 8000
  - Frontend: 3000

## 🏗 System Architecture

```
┌─────────────┐      ┌──────────────┐      ┌─────────────┐
│   Frontend  │─────▶│   Backend    │─────▶│    Data     │
│  (React)    │      │  (FastAPI)   │      │   (CSV)     │
│  Port: 3000 │◀─────│  Port: 8000  │      │             │
└─────────────┘      └──────────────┘      └─────────────┘
       │                     │
       │                     ├── Graphical Lasso (JAX)
       │                     └── LKJ MCMC (NumPyro)
       │
       └── /preprocess (Data verification)
       └── /graphical-lasso (Graphical Lasso execution)
       └── /lkj (Bayesian estimation execution)
```

## 🚀 Setup

### Requirements

- Docker & Docker Compose
- NVIDIA GPU (optional, for faster LKJ estimation)
- CUDA Toolkit (when using GPU)

### Installation Steps

1. **Clone the repository**
```bash
git clone <repository-url>
cd visualize_on_webapp
```

2. **Place data files**
```bash
mkdir -p data
# Place the following files in the data/ directory:
# - routeruns_v1.csv
# - webhooknavstack_v1.csv
```

3. **Start Docker containers**
```bash
docker-compose up --build
```

4. **Access**
- Frontend: http://localhost:3000
- Node for tsx: http://localhost:3001
- Backend API: http://localhost:8000
- API Documentation: http://localhost:8000/docs

## 📖 Usage

### 1. Data Preprocessing Verification

```
http://localhost:3001/preprocess
```

- Verify error matrix
- Display basic statistics
- Check heatmap

### 2. Graphical Lasso Execution

```
http://localhost:3001/graphical-lasso
```

**Parameter Adjustment**:
- `alpha` (0.001 ~ 1.0): L1 regularization parameter (larger = sparser)
- `rho` (0.1 ~ 10.0): ADMM penalty strength
- `max_iter`: Maximum number of iterations
- `tol`: Convergence threshold

**Display Content**:
- Covariance matrix heatmap
- Precision matrix heatmap
- Partial correlation coefficient heatmap
- Convergence information

### 3. LKJ Bayesian Estimation Execution

```
http://localhost:3001/lkj
```

**Execution Steps**:
1. Click "Run MCMC Sampling" button
2. After sampling completes, click "Get Summary" to retrieve statistics
3. Click "Show Heatmap" to display visualization

**Display Content**:
- Covariance matrix (posterior mean)
- Uncertainty (posterior standard deviation)
- Variance statistics for each variable
- Covariance statistics

## 🔌 API Endpoints

### Data Preprocessing

```http
GET /api/test_dataframe
```
Retrieve sample data from error matrix

```http
GET /api/test_heatmap
```
Retrieve heatmap image of preprocessed data

### Graphical Lasso

```http
POST /api/graphical-lasso
Content-Type: application/json

{
  "alpha": 0.01,
  "max_iter": 100,
  "tol": 0.0001,
  "rho": 1.0
}
```

**Response**:
```json
{
  "covariance_matrix": [[...]],
  "precision_matrix": [[...]],
  "feature_names": [...],
  "partial_correlations": [[...]],
  "config": {...},
  "convergence_info": {...}
}
```

### LKJ Bayesian Estimation

```http
POST /api/lkj/run-mcmc
```
Execute MCMC sampling (may take several minutes to tens of minutes)

```http
GET /api/lkj/covariance-summary
```
Retrieve summary statistics of covariance matrix

```http
GET /api/lkj/covariance-heatmap
```
Retrieve heatmap image (PNG format)

```http
GET /api/lkj/covariance-heatmap-base64
```
Retrieve Base64-encoded heatmap

```http
DELETE /api/lkj/clear-cache
```
Clear MCMC sample cache

## 📊 Algorithm Details

### Graphical Lasso

**Objective Function**:
```
minimize -log det(Θ) + trace(SΘ) + α||Θ||₁
subject to Θ ≻ 0
```

- Θ: Precision matrix (inverse of covariance matrix)
- S: Sample covariance matrix
- α: Regularization parameter controlling sparsity

**Optimization Method**: ADMM (Alternating Direction Method of Multipliers)

Augmented Lagrangian formulation:
```
L_ρ(Θ, Z, U) = -log det(Θ) + trace(SΘ) + α||Z||₁ + (ρ/2)||Θ - Z + U||²_F
```

**Parameter Meanings**:
- Small `alpha` → Dense graph (retains many correlations)
- Large `alpha` → Sparse graph (only important correlations)
- Small `rho` → Slow convergence
- Large `rho` → Fast convergence but numerically unstable

### LKJ Bayesian Estimation

**Model Structure**:
```python
L_corr ~ LKJCholesky(n_features, eta=2.0)
sigma ~ Exponential(1.0)
mu ~ Normal(0, 1)
obs ~ MultivariateNormal(mu, Σ)
```

- `L_corr`: Cholesky decomposition of correlation matrix
- `sigma`: Standard deviation of each variable
- `Σ = diag(sigma) @ R @ diag(sigma)`: Covariance matrix

**MCMC Sampling**:
- Algorithm: NUTS (No-U-Turn Sampler)
- Warmup: 300 samples
- Main sampling: 700 samples
- Target acceptance rate: 0.9

**LKJ Prior Parameter `eta`**:
- `eta = 1.0`: Uniform distribution (all correlation matrices equally likely)
- `eta > 1.0`: Favor identity matrix (prefer no correlation)
- `eta < 1.0`: Allow extreme correlations

## 📁 Directory Structure

```
visualize_on_webapp/
├── backend/                    # Backend API
│   ├── main.py                # FastAPI application main
│   ├── graphical_lasso.py     # Graphical Lasso implementation
│   ├── lkj.py                 # LKJ Bayesian estimation implementation
│   ├── preprocess.py          # Data preprocessing pipeline
│   ├── model.py               # Linear regression model (utility)
│   ├── requirements.txt       # Python dependencies
│   └── Dockerfile             # Backend container configuration
│
├── frontend/                   # Frontend UI
│   ├── src/
│   │   ├── App.tsx            # Main application
│   │   ├── preprocess.tsx     # Data preprocessing page
│   │   ├── graphicallasso.tsx # Graphical Lasso page
│   │   ├── Lkj.tsx            # LKJ estimation page
│   │   └── *.css              # Stylesheets
│   ├── package.json           # npm dependencies
│   ├── vite.config.ts         # Vite configuration
│   └── Dockerfile             # Frontend container configuration
│
├── data/                       # Data files (recommended for .gitignore)
│   ├── route.csv               # Route execution data
│   ├── navigation.csv          # Navigation stack data
│   └── df_combined.csv         # Combined data (auto-generated)
│
└── docker-compose.yml          # Docker Compose configuration
```

## ⚙️ Configuration and Customization

### Backend Configuration

**JAX GPU Settings** (`lkj.py`):
```python
os.environ['XLA_FLAGS'] = '--xla_gpu_cuda_data_dir=/usr/local/cuda'
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['XLA_CLIENT_MEM_FRACTION'] = '0.8'
```

**MCMC Parameter Tuning** (`lkj.py`):
```python
mcmc = MCMC(
    nuts_kernel,
    num_warmup=300,      # Number of warmup samples
    num_samples=700,     # Number of main samples
    num_chains=1,        # Number of chains
    progress_bar=True
)
```

### CORS Settings

Configure allowed origins in `backend/main.py`:
```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

## 🐛 Troubleshooting

### GPU-Related Errors

If GPU is not available, the system automatically falls back to CPU.
```python
# Backend: backend/lkj.py
print(f'Backend: {jax.default_backend()}')  # 'gpu' or 'cpu'
print(f'Devices: {jax.devices()}')
```

### Out of Memory Errors

If memory issues occur during MCMC sampling:
1. Reduce sample counts (`num_samples`, `num_warmup`)
2. Reduce number of chains (`num_chains`)
3. Reduce data size

### Port Conflicts

If ports 8000 or 3000 are in use, modify `docker-compose.yml`:
```yaml
services:
  backend:
    ports:
      - "8080:8000"  # Host:Container
  frontend:
    ports:
      - "3001:3000"
```

## 📊 Performance Notes

- **Graphical Lasso**: Completes in seconds
- **LKJ MCMC**: Takes several minutes to tens of minutes
- **GPU Acceleration**: Significantly speeds up MCMC sampling
- **Data Size**: Larger datasets increase preprocessing time

## 🔬 Mathematical Background

### Graphical Lasso Theory

The Graphical Lasso estimates a sparse inverse covariance matrix (precision matrix) by solving:

```
argmin_Θ { -log det(Θ) + trace(SΘ) + α||Θ||₁ }
```

The L1 penalty (||Θ||₁) encourages sparsity in the precision matrix, which corresponds to conditional independence between variables. If Θᵢⱼ = 0, then variables i and j are conditionally independent given all other variables.

**ADMM Algorithm**:
1. **Θ-update**: Solve via eigenvalue decomposition
2. **Z-update**: Apply soft-thresholding (promotes sparsity)
3. **U-update**: Update dual variable
4. Repeat until convergence

### LKJ Prior Theory

The LKJ (Lewandowski-Kurowicka-Joe) distribution is a prior for correlation matrices that:
- Ensures positive definiteness
- Allows control over correlation strength via concentration parameter η
- Has efficient sampling via Cholesky parameterization

The full model hierarchy:
```
R ~ LKJ(η)                    # Correlation matrix
σᵢ ~ Exponential(1)           # Standard deviations
Σ = diag(σ) R diag(σ)        # Covariance matrix
μᵢ ~ Normal(0, 1)             # Means
xᵢ ~ MultivariateNormal(μ, Σ) # Data likelihood
```

## 📝 Advanced Usage

### Custom Data Format

To use your own data, ensure CSV files have these columns:

**route.csv**:
- `id`: Session identifier
- `start`: Session start time (ISO format)
- `end`: Session end time (ISO format)

**navigation.csv**:
- `id`: Session identifier
- `event_time`: Event timestamp (ISO format)
- `error_code`: Error code
- `error_message`: Error reason

### Extending the Application

**Add new estimation methods**:
1. Create implementation in `backend/`
2. Add API endpoint in `main.py`
3. Create frontend component in `frontend/src/`
4. Add route in `App.tsx`

**Modify preprocessing**:
Edit `backend/preprocess.py` to customize:
- Time matching logic
- Error code filtering
- Feature engineering

## 🧪 Testing

### Backend Testing
```bash
cd backend
pip install -r requirements.txt
python -m pytest tests/  # (if tests exist)
```

### Frontend Testing
```bash
cd frontend
npm install
npm run test  # (if tests configured)
```

### Manual API Testing
```bash
# Health check
curl http://localhost:8000/

# Test Graphical Lasso
curl -X POST http://localhost:8000/api/graphical-lasso \
  -H "Content-Type: application/json" \
  -d '{"alpha": 0.01, "max_iter": 100, "tol": 0.0001, "rho": 1.0}'
```

## 📚 References

### Graphical Lasso
- Friedman, J., Hastie, T., & Tibshirani, R. (2008). Sparse inverse covariance estimation with the graphical lasso. *Biostatistics*, 9(3), 432-441.
- Boyd, S., Parikh, N., Chu, E., Peleato, B., & Eckstein, J. (2011). Distributed optimization and statistical learning via the alternating direction method of multipliers. *Foundations and Trends in Machine Learning*, 3(1), 1-122.

### LKJ Prior
- Lewandowski, D., Kurowicka, D., & Joe, H. (2009). Generating random correlation matrices based on vines and extended onion method. *Journal of Multivariate Analysis*, 100(9), 1989-2001.
- Stan Development Team (2023). LKJ Correlation Distribution. *Stan Reference Manual*.

### NumPyro & MCMC
- Phan, D., Pradhan, N., & Jankowiak, M. (2019). Composable effects for flexible and accelerated probabilistic programming in NumPyro. *arXiv preprint arXiv:1912.11554*.
- Hoffman, M. D., & Gelman, A. (2014). The No-U-Turn sampler: adaptively setting path lengths in Hamiltonian Monte Carlo. *Journal of Machine Learning Research*, 15(1), 1593-1623.

## 📄 License

Please contact the project owner regarding licensing information.

## 🤝 Contributing

Contributions are welcome! Before submitting a pull request, please ensure:

1. Code follows existing style conventions
2. Appropriate tests are included
3. Documentation is updated
4. Changes are described in the PR

### Development Workflow
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📧 Contact

For questions or issues, please create an Issue in the repository.

---

## 💡 Development Tips

- **Graphical Lasso** completes in seconds, but **LKJ MCMC** takes several minutes to tens of minutes
- When using GPU, pay attention to CUDA version compatibility
- Large datasets may require significant preprocessing time
- Monitor memory usage during MCMC sampling
- Use parameter caching to avoid redundant computations
- Start with small `alpha` values and increase gradually for Graphical Lasso
- For LKJ, begin with `eta=2.0` (moderate prior strength)

## 🔄 Version History

### Current Version
- Graphical Lasso with ADMM optimization
- LKJ Bayesian estimation with NUTS sampler
- Interactive web interface
- Real-time heatmap visualization
- Docker containerization

### Future Enhancements
- [ ] Additional covariance estimation methods
- [ ] Parallel MCMC chains
- [ ] Automated hyperparameter tuning
- [ ] Export results in multiple formats
- [ ] Performance optimization for large datasets
- [ ] Comprehensive test suite
- [ ] CI/CD pipeline integration

## 🙏 Acknowledgments

This project utilizes several excellent open-source libraries:
- **JAX**: High-performance numerical computing
- **NumPyro**: Probabilistic programming
- **FastAPI**: Modern web framework
- **React**: UI library
- **Docker**: Containerization platform
