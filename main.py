# main.py
from core.registry import SolverRegistry
from solvers.randomized import RandomizedKaczmarz
from solvers.neural import HANN_Solver
import numpy as np

# تسجيل المحركات
SolverRegistry.register("kaczmarz", RandomizedKaczmarz)
SolverRegistry.register("hann", HANN_Solver)

if __name__ == "__main__":
    # 1. حل نظام خطي كبير بطريقة Kaczmarz
    print("=== Kaczmarz العشوائي ===")
    A = np.random.rand(100, 50)  # نظام مفرط
    b = np.random.rand(100)
    solver = SolverRegistry.get("kaczmarz")()
    x = solver.solve(A, b, max_iter=2000)
    print("الخطأ المتبقي:", np.linalg.norm(A @ x - b))
    
    # 2. حل معادلة غير خطية بطريقة HANN
    print("\n=== HANN (شبكة عصبية) ===")
    def nonlinear_sys(x):
        return np.array([
            x[0]**2 + x[1]**2 - 1,
            x[0] - x[1]
        ])
    
    solver_nn = SolverRegistry.get("hann")()
    sol = solver_nn.solve(nonlinear_sys, dim=2, epochs=1000)
    print("الحل التقريبي:", sol)
    print("التحقق (يجب أن يقترب من الصفر):", nonlinear_sys(sol))
