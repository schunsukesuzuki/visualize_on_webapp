import jax.numpy as jnp
from jax import grad, jit
import polars as pl
from typing import Tuple

class LinearRegressionJAX:
    """regression model"""
    
    def __init__(self):
        self.slope = None
        self.intercept = None
        self.is_fitted = False
    
    @staticmethod
    @jit
    def predict_jit(X: jnp.ndarray, slope: float, intercept: float) -> jnp.ndarray:
        """prediction function"""
        return slope * X + intercept
    
    @staticmethod
    @jit
    def mse_loss(params: Tuple[float, float], X: jnp.ndarray, y: jnp.ndarray) -> float:
        """loss function"""
        slope, intercept = params
        predictions = slope * X + intercept
        return jnp.mean((predictions - y) ** 2)
    
    def fit(self, X: jnp.ndarray, y: jnp.ndarray, learning_rate: float = 0.001, epochs: int = 1000):
        """training model with gradient descent and regularization"""
        X_mean = jnp.mean(X)
        X_std = jnp.std(X)
        y_mean = jnp.mean(y)
        y_std = jnp.std(y)
        
        X_normalized = (X - X_mean) / X_std
        y_normalized = (y - y_mean) / y_std
        
        # parameter initialization
        slope = 0.0
        intercept = 0.0
        
        # gradient function
        grad_fn = grad(self.mse_loss, argnums=0)
        
        # gradient descent
        for epoch in range(epochs):
            params = (slope, intercept)
            grads = grad_fn(params, X_normalized, y_normalized)
            
            # update parameters
            slope -= learning_rate * grads[0]
            intercept -= learning_rate * grads[1]
            
            if epoch % 200 == 0:
                loss = self.mse_loss((slope, intercept), X_normalized, y_normalized)
                print(f"Epoch {epoch}, Loss: {loss:.6f}")
        
        # restore scaled figures
        self.slope = float(slope * y_std / X_std)
        self.intercept = float(y_mean - self.slope * X_mean)
        self.is_fitted = True
        
        print(f"\n trarining completed!")
        print(f"slope: {self.slope:.2f}")
        print(f"intercept: {self.intercept:.2f}")
    
    def predict(self, X: jnp.ndarray) -> jnp.ndarray:
        """execute prediction"""
        if not self.is_fitted:
            raise ValueError("model is not trained")
        
        return self.predict_jit(X, self.slope, self.intercept)
    
    def score(self, X: jnp.ndarray, y: jnp.ndarray) -> float:
        """calcuration of R² score"""
        if not self.is_fitted:
            raise ValueError("model is not trained"")
        
        predictions = self.predict(X)
        ss_res = jnp.sum((y - predictions) ** 2)
        ss_tot = jnp.sum((y - jnp.mean(y)) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        return float(r2)


def load_and_train_model(data_path: str = "../data/sales_data.csv") -> LinearRegressionJAX:
    # Polars
    df = pl.read_csv(data_path)
    
    # transform to jax.numpy
    X = jnp.array(df["visitors"].to_numpy())
    y = jnp.array(df["sales"].to_numpy())
    
    print(f"data loading completed: {len(df)}行")
    print(f"estimated range of visitors: {X.min():.0f} ~ {X.max():.0f}人")
    print(f"estimated range of sales: {y.min():.0f} ~ {y.max():.0f}円\n")
    
    model = LinearRegressionJAX()
    model.fit(X, y, learning_rate=0.1, epochs=2000)
    
    # pergormance evaluation
    r2 = model.score(X, y)
    print(f"R² score: {r2:.4f}")
    
    return model


if __name__ == "__main__":
    model = load_and_train_model()
    
    # test
    test_visitors = jnp.array([50.0, 100.0, 150.0])
    predictions = model.predict(test_visitors)
    
    print("\n test results:")
    for v, p in zip(test_visitors, predictions):
        print(f"estimaeted visitors: {v:.0f}人 → estimated sales: {p:.0f}")
