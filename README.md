# doctormath
math systems

البرنامج يبحث عن "محركات الحل" في مجلد `solvers/`. كل محرك يلتزم بواجهة برمجية موحدة(`solve`).

math_solver/

├── core/

│   ├── __init__.py
│   ├── base.py           # الواجهة المجردة (BaseSolver)
│   └── registry.py       # سجل المحركات (لاستدعائها بالاسم)
├── solvers/
│   ├── __init__.py
│   ├── direct.py         # الحلول المباشرة (Gauss, LU)
│   ├── randomized.py     # Kaczmarz العشوائي (للمصفوفات الضخمة) 
│   ├── neural.py         # HANN (للمعادلات غير الخطية) 
│   └── iterative.py      # Jacobi / Gauss-Seidel / CG
├── utils/
│   └── matrix_loader.py
└── main.py

