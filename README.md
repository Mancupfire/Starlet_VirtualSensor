-----

# Starlet\_VirtualSensor: Hybrid DeepONet for Real-time CO2 Monitoring

> **Dự án triển khai mô hình Hybrid DeepONet để tái tạo trường nồng độ CO2 3D trong phòng theo thời gian thực.**

Hệ thống sử dụng dữ liệu mô phỏng CFD (Steady-state) để huấn luyện (Offline phase) và tích hợp cảm biến vật lý (Physical Sensor) để tự động hiệu chỉnh mô hình khi vận hành (Online phase/Inference).

-----

## 📂 1. Cấu trúc Dự án

```text
DeepONet_Project/
├── checkpoints/                      # Thư mục tự động lưu model và scalers sau khi train
├── requirements.txt                  # Danh sách thư viện phụ thuộc
├── Hybrid_DeepONet.py                # Kiến trúc mạng (BranchNet, TrunkNet, HybridDeepONet)
├── DataLoader_Preprocessing.py       # Xử lý dữ liệu CSV, chuẩn hóa (Scaler), tạo dummy data
├── Training.py                       # Script huấn luyện mô hình (Training Loop)
└── Realtime_Inference.py             # Script chạy dự đoán thực tế (Real-time Prediction)
```

-----

## 🛠️ 2. Cài đặt Môi trường

**Yêu cầu:** Python 3.8 trở lên.

### Bước 1: Khởi tạo môi trường ảo (Khuyên dùng)

```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# Linux/MacOS
python3 -m venv venv
source venv/bin/activate
```

### Bước 2: Cài đặt thư viện

```bash
pip install -r requirements.txt
```

> **Lưu ý về GPU:** Nếu máy tính có GPU NVIDIA, hãy cài đặt phiên bản **PyTorch hỗ trợ CUDA** trước khi chạy lệnh trên để tối ưu hóa tốc độ huấn luyện.

-----

## 📊 3. Cấu trúc Dữ liệu (Data Structure)

Hệ thống làm việc với dữ liệu bảng phẳng (**Flat CSV**), trong đó mỗi hàng đại diện cho một điểm không gian tại một điều kiện vận hành cụ thể.

  * **Định dạng:** `.csv`
  * **Cột bắt buộc:** `x`, `y`, `z`, `u`, `CO2`, `Q_supply`, `CO2_source`, `Vs`, `Ps`

| Nhóm dữ liệu | Tên cột | Ý nghĩa Vật lý | Vai trò trong Hybrid DeepONet |
| :--- | :--- | :--- | :--- |
| **Không gian** | `x`, `y`, `z` | Tọa độ điểm đo trong phòng ($m$). | **Input (Trunk Net)**: Định danh vị trí cần dự báo. |
| **Trường Vật lý** | `u` | Vận tốc dòng khí ($m/s$). | *Mở rộng*: Hiện chưa dùng cho Baseline, giữ lại cho đa nhiệm. |
| | `CO2` | Nồng độ CO2 ($ppm$). | **Label (Ground Truth)**: Dùng để tính hàm Loss ($\mathcal{L}_{data}$). |
| **Điều kiện biên**| `Q_supply` | Lưu lượng gió cấp ($Nm^3/s$). | **Input (Branch Net)**: Tham số điều khiển chính. |
| | `CO2_source`| Cường độ nguồn thải ($kg/s$). | **Input (Branch Net)**: Thông tin nguồn phát thải. |
| **Cảm biến** | `Vs` | Giá trị CO2 ảo - Virtual. | **Input (Branch Net)**: Tham số tham chiếu từ CFD. |
| | `Ps` | Giá trị CO2 thực - Physical. | **Input (Branch Net)**: Dùng để học độ lệch thực tế. |

> **Lưu ý:** Trong tập huấn luyện (Training set), giá trị `Ps` thường được giả định bằng `Vs` (Môi trường lý tưởng).

-----

## 🚀 4. Hướng dẫn Sử dụng

### Bước 1: Chuẩn bị dữ liệu

Bạn cần file `.csv` chứa kết quả mô phỏng CFD theo cấu trúc trên.

  * Nếu **chưa có dữ liệu**, script `train.py` sẽ tự động kích hoạt hàm `generate_dummy_data` để sinh ra 10.000 mẫu giả lập tuân theo quy luật vật lý đơn giản.

### Bước 2: Huấn luyện (Training)

Chạy lệnh sau để bắt đầu quá trình huấn luyện:

```bash
python train.py --data_path "dataset.csv" --save_dir "./checkpoints" --epochs 200 --batch_size 128 --gpu_id 0
```

**Tham số:**

  * `--data_path`: Đường dẫn file CSV (VD: `D:/Data/CFD/final_data.csv`). Nếu để trống, code sẽ tạo dữ liệu giả.
  * `--save_dir`: Thư mục lưu `best_model.pth` và các file `.pkl` (scaler).
  * `--epochs`: Số vòng lặp (Mặc định: 200).
  * `--gpu_id`: ID của GPU (Mặc định: 0). Tự động chuyển về CPU nếu không tìm thấy GPU.

**Kết quả đầu ra (trong thư mục `checkpoints/`):**

1.  `best_model.pth`: Trọng số mô hình tối ưu.
2.  `scaler_u.pkl`, `scaler_y.pkl`, `scaler_target.pkl`: Các bộ chuẩn hóa dùng cho Inference.

### Bước 3: Dự đoán (Inference)

Chạy script sau để demo khả năng "Hiệu chỉnh thực tế" (Self-Calibration):

```bash
python inference.py --model_dir "./checkpoints" --gpu_id 0
```

**Cách kiểm tra:**
Mở file `inference.py`, tìm đến đoạn `if __name__ == "__main__":` và thay đổi giá trị input để thấy sự khác biệt:

  * `Ps_in`: Giá trị cảm biến thực đo được.
  * `Vs_in`: Giá trị CFD lý thuyết tại vị trí cảm biến.

-----

## 🧠 5. Kiến trúc & Logic Mô hình

### Model Architecture (`model.py`)

  * **Trunk Net:** Sử dụng hàm kích hoạt **Sine (SIREN)** để đảm bảo tính trơn và liên tục của trường không gian 3D.
  * **Output Layer:** Đi qua hàm kích hoạt **Softplus** để đảm bảo nồng độ CO2 dự báo luôn dương.

### Data Preprocessing (`data_loader.py`)

  * Sử dụng **MinMaxScaler** để đưa toàn bộ dữ liệu (Input/Output) về khoảng `[0, 1]` hoặc `[-1, 1]`. Bước này giúp mạng hội tụ nhanh và ổn định hơn.

### Inference Logic (Cơ chế Self-Calibration)

Input vector $u$ của Branch Net được cấu thành từ: $[V_s, P_s, Source, Q]$.

1.  Khi giá trị cảm biến thực $P_s$ thay đổi (khác với CFD $V_s$).
2.  Mạng Branch Net sẽ tính toán lại vector hệ số ẩn (Latent Coefficients $b$).
3.  Trường đầu ra 3D thay đổi theo công thức: $CO_2(x) = \sum (b_k \cdot t_k(x))$.

$\rightarrow$ Đây chính là cơ chế giúp mô hình tự động hiệu chỉnh toàn bộ trường nồng độ trong phòng dựa trên một điểm đo thực tế duy nhất.
