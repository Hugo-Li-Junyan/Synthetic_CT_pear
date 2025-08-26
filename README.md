# 🍐 Generative Model for 3D CT Pears  


A diffusion-based generative modeling framework for **3D computed tomography (CT) scans of pears**.  
This project enables the creation of realistic synthetic pear CT volumes for research in agricultural imaging, defect detection, dataset augmentation, and computer vision.

---

## ✨ Features
- 📦 3D generative model for volumetric CT data  
- 🔄 Data interpolation between samples
- 🧪 Unconditional synthetic data generation for training & evaluation  
- 👀 Tools for visualization and quality metrics  
- 🌱 Extensible to other agricultural or biological datasets  

---

## ⚙️ Installation
```bash
# Clone the repository
git clone https://github.com/Hugo-Li-Junyan/Synthetic_CT_pear.git
cd Synthetic_CT_pear

# Create and activate environment (optional)
python -m venv env
source env/bin/activate   # On Windows: env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```
## 🚀 Usage
🔧 VAE Training
```bash
  python train_vae.py --class1_dir --class2_dir --save_dir
  ```
🔧 Diffuser Training
```bash
  python train_diffuser.py --class1_dir --class2_dir --save_dir --model_id
```
🎨 Generate Synthetic 3D Pears
```bash
  python main.py --model_dir --save_dir --batch_size 2 --batches 3000
```
👀 Interpolation
```bash
  python interpolation_line.py 
```

	
## 🔬 Applications

🍏 Synthetic dataset augmentation for deep learning

🍐 X-ray radiograph simulation for automated defect detection 

🧑‍⚕️ Latent space exploration for downstream tasks