"""
Script de Verificación de Instalación
======================================

Ejecuta este script para verificar que todo está correctamente instalado.
"""

import sys

def check_package(name, import_name=None):
    """Verifica si un paquete está instalado"""
    if import_name is None:
        import_name = name
    
    try:
        module = __import__(import_name)
        version = getattr(module, '__version__', 'desconocida')
        print(f"✅ {name:20s} {version}")
        return True
    except ImportError:
        print(f"❌ {name:20s} NO INSTALADO")
        return False

def main():
    print("=" * 70)
    print("VERIFICACIÓN DEL ENTORNO DE IA")
    print("=" * 70)
    print()
    
    # Python version
    print(f"🐍 Python: {sys.version.split()[0]}")
    print(f"   Path: {sys.executable}")
    print()
    
    # Verificar paquetes
    print("📦 Paquetes instalados:")
    print("-" * 70)
    
    packages = [
        ('NumPy', 'numpy'),
        ('Pandas', 'pandas'),
        ('Matplotlib', 'matplotlib'),
        ('Seaborn', 'seaborn'),
        ('Plotly', 'plotly'),
        ('SciPy', 'scipy'),
        ('Scikit-learn', 'sklearn'),
        ('PyTorch', 'torch'),
        ('TorchVision', 'torchvision'),
        ('TensorFlow', 'tensorflow'),
        ('Transformers', 'transformers'),
        ('OpenCV', 'cv2'),
        ('NLTK', 'nltk'),
        ('Jupyter', 'jupyter'),
        ('IPython', 'IPython'),
    ]
    
    results = [check_package(name, imp) for name, imp in packages]
    
    print()
    print("=" * 70)
    
    # Resumen
    total = len(results)
    installed = sum(results)
    
    print(f"📊 Resumen: {installed}/{total} paquetes instalados")
    
    if installed == total:
        print("🎉 ¡Excelente! Todo está correctamente instalado.")
        print("✅ Estás listo para empezar a aprender IA")
    elif installed >= total * 0.8:
        print("⚠️  Algunos paquetes opcionales faltan, pero puedes empezar.")
    else:
        print("❌ Faltan paquetes importantes.")
        print("   Ejecuta: pip install -r requirements.txt")
    
    print()
    
    # Verificaciones adicionales
    print("🔍 Verificaciones adicionales:")
    print("-" * 70)
    
    # PyTorch CUDA
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            print(f"✅ PyTorch CUDA: Disponible ({torch.cuda.get_device_name(0)})")
        else:
            print("ℹ️  PyTorch CUDA: No disponible (usando CPU)")
    except:
        print("❌ No se pudo verificar PyTorch CUDA")
    
    # Jupyter kernel
    try:
        import ipykernel
        print("✅ IPython Kernel: Instalado")
    except:
        print("⚠️  IPython Kernel: No instalado (pip install ipykernel)")
    
    print()
    print("=" * 70)
    print("Verificación completada. ¡Buena suerte con tu aprendizaje! 🚀")
    print("=" * 70)

if __name__ == "__main__":
    main()
