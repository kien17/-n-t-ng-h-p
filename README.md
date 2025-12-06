# 🧠 Đồ án Tổng hợp – Hướng Trí tuệ nhân tạo

Phân cụm ảnh bằng các thuật toán **K-Means, DBSCAN, OPTICS** và trực quan hóa dữ liệu bằng **PCA, t-SNE**.  
Toàn bộ chương trình được chạy thông qua file `main.py`.

---

## ⚙️ Requirements

### 1. Môi trường

- Python **>= 3.10**
- Hệ điều hành: Windows / Linux / macOS

### 2. Thư viện cần cài

Cài đặt trực tiếp bằng `pip`:

    pip install numpy opencv-python matplotlib scikit-learn

---

## 📂 Dữ liệu đầu vào

Ảnh dùng để phân cụm được đặt trong thư mục như sau:

    data/
      └── input/
            ├── img1.jpg
            ├── img2.png
            ├── ...

Kết quả phân cụm (ảnh đã chia cụm, file nhãn, hình trực quan hóa, …) sẽ được lưu vào các thư mục output mà nhóm cấu hình trong code.

---

## ▶️ Cách chạy chương trình

Đứng ở thư mục gốc của project và chạy:

- Phân cụm bằng **K-Means**  

      python main.py kmeans

- Phân cụm bằng **DBSCAN**  

      python main.py dbscan

- Phân cụm bằng **OPTICS**  

      python main.py optics

`main.py` sẽ:
- Gọi module tiền xử lý (`Preprocessing.py`) để đọc & chuẩn hóa ảnh.  
- Gọi thuật toán tương ứng trong `kMeans.py`, `dbscan.py` hoặc `optics.py`.  
- (Nếu được cấu hình) lưu nhãn cụm và/hoặc gọi `PCA_t-SNE.py` để trực quan hóa kết quả.

---

## 📌 Ghi chú

- Nếu thay đổi tên thư mục dữ liệu (ví dụ không dùng `data/input/`), cần chỉnh lại đường dẫn tương ứng trong các file Python.  
- Có thể chỉnh các tham số thuật toán (K, eps, min_samples, …) trực tiếp trong các file:
  - `kMeans.py`
  - `dbscan.py`
  - `optics.py`
