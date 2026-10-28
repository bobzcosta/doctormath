# solvers/randomized.py
import numpy as np
from core.base import BaseSolver

class RandomizedKaczmarz(BaseSolver):
    """
    نسخة محسنة من Kaczmarz العشوائي.
    مناسبة للأنظمة المفرطة (m > n) أو المصفوفات الكبيرة جداً.
    """
    def solve(self, A, b, max_iter=5000, tol=1e-6, seed=None):
        A = np.array(A, dtype=float)
        b = np.array(b, dtype=float)
        m, n = A.shape
        
        if seed:
            np.random.seed(seed)
        
        # تخزين معايير الصفوف لتسريع الاختيار العشوائي
        row_norms = np.sum(A**2, axis=1)
        if np.any(row_norms == 0):
            raise ValueError("المصفوفة تحتوي على صف صفري.")
        
        x = np.zeros(n)  # نقطة البداية
        
        for k in range(max_iter):
            # اختيار صف عشوائي (احتمالية تتناسب مع مربع الطول) [citation:1]
            i = np.random.choice(m, p=row_norms / np.sum(row_norms))
            
            # تحديث الحل (الإسقاط على الصف)
            a_i = A[i]
            b_i = b[i]
            x = x + ((b_i - np.dot(a_i, x)) / row_norms[i]) * a_i
            
            # شرط التوقف
            if k % 100 == 0:
                residual = np.linalg.norm(A @ x - b)
                if residual < tol:
                    print(f"تقارب عند التكرار {k}")
                    break
        
        return x

# --- اختبار سريع داخل الملف ---
if __name__ == "__main__":
    A = np.array([[3, 2], [1, 4], [2, 1]], dtype=float)
    b = np.array([5, 6, 4], dtype=float)
    solver = RandomizedKaczmarz()
    print("الحل التقريبي:", solver.solve(A, b))
