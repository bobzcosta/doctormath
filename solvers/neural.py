# solvers/neural.py
import numpy as np
from core.base import BaseSolver

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

class HANN_Solver(BaseSolver):
    """
    Homotopy Auxiliary Neural Network (HANN).
    تستخدم شبكة عصبية لحل F(x) = 0 بدون الحاجة لنقطة بداية دقيقة.
    المرجع: ScienceDirect, Oct 2026 [citation:3]
    """
    def solve(self, F, x0=None, dim=1, epochs=2000, hidden=40):
        if not TORCH_AVAILABLE:
            raise ImportError("يتطلب هذا الحل تثبيت PyTorch: pip install torch")
        
        # F: دالة تقبل مصفوفة numpy وتعيد مصفوفة numpy
        # dim: عدد المتغيرات
        
        # 1. تعريف الشبكة (4 طبقات كما في الورقة البحثية) [citation:3]
        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(1, hidden), nn.Tanh(),
                    nn.Linear(hidden, hidden), nn.Tanh(),
                    nn.Linear(hidden, hidden), nn.Tanh(),
                    nn.Linear(hidden, hidden), nn.Tanh(),
                    nn.Linear(hidden, dim)
                )
            def forward(self, t):
                return self.net(t)
        
        model = Net()
        optimizer = torch.optim.LBFGS(model.parameters(), lr=0.1, max_iter=20)
        
        # 2. نقاط التدريب (t من 0 إلى 1)
        t_vals = torch.linspace(0, 1, 100).reshape(-1, 1)
        
        # 3. دالة الخسارة (Homotopy Residual)
        def closure():
            optimizer.zero_grad()
            pred = model(t_vals)  # الشكل: [100, dim]
            
            # نفترض مسار Homotopy خطي: H(x, t) = F(x) - (1-t)*F(x0)
            # حيث x0 هي النقطة الابتدائية (يمكن أن تكون صفراً)
            x0_tensor = torch.zeros(dim) if x0 is None else torch.tensor(x0, dtype=torch.float32)
            F_x0 = torch.tensor(F(x0_tensor.numpy()), dtype=torch.float32)
            
            loss = 0
            for i, t in enumerate(t_vals):
                x_t = pred[i]
                # حساب F(x_t) باستخدام numpy (لأن F قد لا تكون differentiable)
                F_x = torch.tensor(F(x_t.detach().numpy()), dtype=torch.float32)
                residual = F_x - (1 - t) * F_x0
                loss += torch.sum(residual**2)
            
            loss.backward()
            return loss
        
        # 4. التدريب
        for epoch in range(epochs):
            optimizer.step(closure)
            if epoch % 200 == 0:
                print(f"Epoch {epoch}, Loss: {closure().item():.2e}")
        
        # 5. استخراج الحل عند t=1
        with torch.no_grad():
            final_x = model(torch.tensor([[1.0]])).numpy().flatten()
        
        return final_x

# --- اختبار سريع ---
if __name__ == "__main__":
    # مثال: حل x^2 - 2 = 0 (الجذر التربيعي لـ 2)
    def my_eq(x):
        return np.array([x[0]**2 - 2])
    
    solver = HANN_Solver()
    print("الحل:", solver.solve(my_eq, dim=1))
