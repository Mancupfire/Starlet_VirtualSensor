# Starlet_VirtualSensor

Baseline Hybrid DeepONet for Real-time CO2 Monitoring

Dự án này triển khai mô hình Hybrid DeepONet để tái tạo trường nồng độ CO2 3D trong phòng theo thời gian thực, sử dụng dữ liệu mô phỏng CFD (Steady-state) để huấn luyện và cảm biến vật lý (Physical Sensor) để hiệu chỉnh khi vận hành.

1. Cấu trúc Thư mục

DeepONet_Project/
├── requirements.txt        # Danh sách thư viện cần cài đặt
├── model.py                # Định nghĩa kiến trúc mạng (BranchNet, TrunkNet, HybridDeepONet)
├── data_loader.py          # Xử lý dữ liệu CSV, chuẩn hóa (Scaler), tạo dataset giả lập
├── train.py                # Script huấn luyện mô hình (Training Loop)
└── inference.py            # Script chạy dự đoán thực tế (Real-time Prediction)


2. Cài đặt Môi trường

Yêu cầu: Python 3.8 trở lên.

Bước 1: Tạo môi trường ảo (Recommended)

# Windows
python -m venv venv
.\venv\Scripts\activate


Bước 2: Cài đặt các thư viện cần thiết

pip install -r requirements.txt


Lưu ý: Nếu máy có GPU NVIDIA, hãy cài đặt PyTorch bản hỗ trợ CUDA trước khi chạy lệnh trên để tận dụng sức mạnh GPU.

3. Hướng dẫn Sử dụng

Bước 1: Chuẩn bị Dữ liệu

Bạn cần có file dữ liệu dạng .csv chứa kết quả mô phỏng CFD.

Cấu trúc cột bắt buộc: x, y, z, Q_supply, CO2_source, Vs, Ps, CO2

Lưu ý: Trong tập train, Ps (Physical Sensor) thường được giả định bằng Vs (Virtual Sensor).

Nếu chưa có dữ liệu thật:
Bạn có thể tự động tạo file dữ liệu giả lập (dataset.csv) bằng cách chạy script train.py lần đầu tiên. Script này được tích hợp sẵn hàm generate_dummy_data để sinh ra 10.000 mẫu dữ liệu tuân theo quy luật vật lý đơn giản.

Bước 2: Huấn luyện Mô hình (Training)

Chạy lệnh sau để bắt đầu huấn luyện:

python train.py --data_path "dataset.csv" --save_dir "./checkpoints" --epochs 200 --batch_size 128 --gpu_id 0


Tham số:

--data_path: Đường dẫn tới file CSV dữ liệu (ví dụ: "D:/Data/CFD/final_data.csv"). Nếu để trống hoặc file không tồn tại, code sẽ tự tạo dataset.csv giả.

--save_dir: Thư mục để lưu model (best_model.pth) và các bộ chuẩn hóa (scaler_*.pkl).

--epochs: Số vòng lặp huấn luyện (Mặc định: 200).

--gpu_id: ID của GPU muốn sử dụng (Mặc định: 0). Nếu không có GPU, code tự chuyển về CPU.

Kết quả: Sau khi chạy xong, bạn sẽ thấy thư mục checkpoints/ chứa:

best_model.pth: Trọng số mô hình tốt nhất.

scaler_u.pkl, scaler_y.pkl, scaler_target.pkl: Các file dùng để chuẩn hóa dữ liệu khi chạy Inference.

Bước 3: Chạy Dự đoán (Inference)

Đây là bước demo khả năng "Hiệu chỉnh thực tế" của mô hình. Bạn chạy file inference.py để xem kết quả dự báo tại các điểm trong phòng dựa trên giá trị cảm biến giả định.

python inference.py --model_dir "./checkpoints" --gpu_id 0


Tham số:

--model_dir: Đường dẫn tới thư mục chứa model đã train (phải khớp với --save_dir ở bước trên).

Cách kiểm tra (Trong code inference.py):
Bạn có thể mở file inference.py, kéo xuống phần if __name__ == "__main__": và thay đổi các giá trị:

Ps_in: Giá trị cảm biến thật đo được.

Vs_in: Giá trị CFD lý thuyết.

Chạy lại lệnh trên để thấy sự thay đổi của kết quả dự báo CO2 toàn phòng.

4. Giải thích Logic Code (Dành cho Báo cáo)

model.py: Sử dụng hàm kích hoạt Sine (SIREN) cho Trunk Net để đảm bảo tính trơn của trường không gian. Đầu ra cuối cùng qua Softplus để đảm bảo CO2 luôn dương.

data_loader.py: Sử dụng MinMaxScaler để đưa toàn bộ dữ liệu (Input/Output) về khoảng [0,1] hoặc [-1,1]. Việc này cực kỳ quan trọng để mạng hội tụ nhanh.

Inference Logic:

Input vector u được cấu thành từ [Vs, Ps, Source, Q].

Khi Ps thay đổi (khác Vs), mạng sẽ tự động điều chỉnh vector hệ số b (Coefficients), từ đó làm thay đổi toàn bộ trường 3D đầu ra CO2 = sum(b * t). Đây chính là cơ chế "Self-Calibration".

---

4.1. Cấu trúc Dữ liệu (Data Structure)

Hệ thống được thiết kế để làm việc với dữ liệu bảng phẳng (Flat CSV), trong đó mỗi hàng đại diện cho một điểm không gian tại một điều kiện vận hành cụ thể.

**Định dạng File:** `.csv`
**Các cột bắt buộc:** `x, y, z, u, CO2, Q_supply, CO2_source, Vs, Ps`

**Giải thích chi tiết từng cột:**

| Nhóm dữ liệu | Tên cột | Ý nghĩa Vật lý | Vai trò trong Hybrid DeepONet |
| :--- | :--- | :--- | :--- |
| **Không gian (Space)** | `x`, `y`, `z` | Tọa độ điểm đo trong phòng (m). | **Input cho Trunk Net**: Định danh vị trí cần dự báo. |
| **Trường Vật lý (Field)** | `u` | Vận tốc dòng khí tại điểm đó (m/s). | *Lưu trữ mở rộng*: Hiện tại chưa dùng để train Baseline, nhưng được giữ lại để phát triển tính năng dự báo đa nhiệm sau này. |
| | `CO2` | Nồng độ CO2 tại điểm đó (ppm). | **Label (Ground Truth)**: Dùng để tính hàm Loss ($\mathcal{L}_{data}$) khi huấn luyện. |
| **Điều kiện biên (BCs)** | `Q_supply` | Lưu lượng gió cấp từ điều hòa ($Nm^3/s$). | **Input cho Branch Net**: Tham số điều khiển chính (Control Parameter). |
| | `CO2_source`| Cường độ nguồn thải (kg/s). | **Input cho Branch Net**: Thông tin về nguồn phát (1 người). |
| **Cảm biến (Sensors)** | `Vs` | Giá trị CO2 ảo (Virtual) tại điểm sensor. | **Input cho Branch Net (Baseline)**: Tham số tham chiếu từ CFD. |
| | `Ps` | Giá trị CO2 thực (Physical) tại điểm sensor. | **Input cho Branch Net (Corrector)**: Dùng để mô hình học độ lệch thực tế. |

---
