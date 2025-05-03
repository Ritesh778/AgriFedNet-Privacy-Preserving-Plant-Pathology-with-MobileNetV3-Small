# AgriFedNet: Privacy-Preserving Plant Pathology with MobileNetV3-Small


![poster](https://github.com/user-attachments/assets/30cf8ec5-5656-4751-9103-c23f4284b401)



This repository implements a **Federated Learning (FL)** system for plant disease classification using MobileNetV3-Small on the PlantVillage dataset. The system simulates decentralized learning across three clients with heterogeneous data splits (80%, 15%, 5%) and a global test set (20%), achieving **99.67% accuracy** in just three FL rounds. It highlights the power of collaborative, privacy-preserving learning in smart agriculture.

---

## 📌 Project Highlights

- **Federated Learning with Flower Framework**
- **Three clients with non-IID data**:
  - Client 0 → 80% data
  - Client 1 → 15% data
  - Client 2 → 5% data
- **Global evaluation set**: 20% of PlantVillage
- **Lightweight CNN**: MobileNetV3-Small
- **Visualizations**: Accuracy plots, confusion matrices
- **Global model outperforms standalone clients significantly**

---

## 🧠 Dataset: PlantVillage

- 20,638 RGB leaf images from 15 classes
- Available from [Kaggle](https://www.kaggle.com/datasets/emmarex/plantdisease)
- Organized as `PlantVillage/ClassName/*.jpg`


---

## ⚙️ Installation (via Conda)

### 1. Clone the repo
```bash
git clone https://github.com/Ritesh778/AgriFedNet-Privacy-Preserving-Plant-Pathology-with-MobileNetV3-Small.git
cd AgriFedNet-Privacy-Preserving-Plant-Pathology-with-MobileNetV3-Small
```

### 2. Create & activate Conda environment

#### ✅ Option A: Use `environment.yml` (recommended)
```bash
conda env create -f environment.yml
conda activate plantvillage-fl
```

#### 🛠 Option B: Manual install via `requirements.txt`
```bash
conda create -n plantvillage-fl python=3.12
conda activate plantvillage-fl
conda install pytorch=2.4.1 torchvision=0.19.1 -c pytorch
conda install numpy=1.26.4 scikit-learn=1.5.2 matplotlib=3.9.2 seaborn=0.13.2 pillow=10.4.0 prettytable=3.11.0 -c conda-forge
pip install flwr==1.11.1
```

---

## 🚀 Running the Federated Learning System

### 1. Start the Server
```bash
python server2.py
```

### 2. Run Clients (in separate terminals)
```bash
python client3.py 0  # Client 0
python client3.py 1  # Client 1
python client3.py 2  # Client 2
```

---

<!-- ## 📊 Results Summary

| Round | Loss   | Accuracy | Precision | Recall | F1-Score |
|-------|--------|----------|-----------|--------|----------|
| 1     | 0.1291 | 95.89%   | 95.72%    | 95.08% | 94.95%   |
| 2     | 0.0273 | 99.16%   | 99.12%    | 98.93% | 99.01%   |
| 3     | 0.0115 | **99.67%** | **99.71%** | **99.56%** | **99.63%** |

### 🔎 Local Client Accuracies (before FL)
- Client 0 (80%): 98.45%
- Client 1 (15%): 92.34%
- Client 2 (5%): 69.64%

FL helps boost lower-resourced clients significantly through model aggregation. -->

---

## 🖼 Visualizations

Images are saved in `/images` folder:

- `client_0_accuracy.png`, `client_1_loss.png`, etc.
- `client_0_confusion_matrix.png`, etc.

> 📷 Add your own banner image or generated graphs for better visual impact on GitHub.

---

## 📁 Folder Structure

```
plantvillage-fl/
├── client3.py
├── server2.py
├── requirements.txt
├── environment.yml
├── README.md
├── LICENSE
├── images/
│   ├── client_0_accuracy.png
│   └── ...
├── *.pkl (model checkpoints)
```

---

## 🧪 Troubleshooting

- ❗ **Dataset not found**: Check path in `client3.py`
- ⚠️ **Port in use**: Edit `localhost:8080` to a free port in both `server2.py` and `client3.py`
- 🧵 **Memory issue**: Lower `batch_size` in `client3.py`
- 🧼 **Missing flwr**: Ensure `pip` is installed: `conda install pip`

---

## 🧑‍💻 Contributing

Pull requests are welcome! Please follow PEP-8 style and include testable contributions.

---

## 📄 License

Licensed under the **MIT License**. See `LICENSE`.

---

## 📚 References

1. McMahan et al. (2017), *Communication-efficient learning with decentralized data*, AISTATS  
2. Beutel et al. (2020), *Flower: A friendly federated learning framework*, arXiv  
3. Howard et al. (2019), *Searching for MobileNetV3*, ICCV  
4. Hughes & Salathé (2015), *Open access plant disease dataset*, arXiv

---

## 📬 Contact

- Issues? Open a [GitHub issue](https://github.com/Ritesh778/AgriFedNet-Privacy-Preserving-Plant-Pathology-with-MobileNetV3-Small/issues)
- Maintainer: [Your Name or Email Here]

---

*Last Updated: May 2025*
