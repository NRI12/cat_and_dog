🐶🐱 Phân loại Chó và Mèo
Ứng dụng này giúp phân loại hình ảnh giữa chó và mèo với độ chính xác cao. Dưới đây là các bước để cài đặt và sử dụng ứng dụng này.

🛠️ Cài đặt Môi trường
Để cài đặt và chạy ứng dụng, bạn cần thiết lập môi trường Python phù hợp. Chúng tôi khuyến nghị sử dụng Anaconda để quản lý các gói và môi trường.

Cài đặt Anaconda: Tải và cài đặt Anaconda từ trang web chính thức.
Tạo môi trường mới: Mở Anaconda Prompt và tạo một môi trường Python mới bằng cách chạy lệnh conda create -n dogscats python=3.8.
Kích hoạt môi trường: Sử dụng lệnh conda activate dogscats.
📚 Cài đặt Thư viện
Trong môi trường dogscats, cài đặt các thư viện cần thiết:
conda install numpy pandas matplotlib scikit-learn tensorflow keras opencv
📦 Chuẩn bị Dữ liệu
Dữ liệu cho việc huấn luyện có thể tìm thấy trong bộ dữ liệu dogscats. Tải và giải nén dữ liệu vào thư mục mong muốn.

🏋️‍♂️ Huấn luyện Mô hình
Mở file train.ipynb trong môi trường Jupyter Notebook.
Điều chỉnh các tham số huấn luyện hoặc sử dụng cấu hình mặc định.
Chạy các cell trong notebook để huấn luyện mô hình.
Mô hình được huấn luyện sẽ có độ chính xác khoảng 97%.
🖥️ Sử dụng Giao Diện Người Dùng (GUI)
Sau khi huấn luyện, mô hình sẽ được lưu trong thư mục model_trained.
Đảm bảo bạn đã chỉnh sửa đường dẫn đến mô hình trong file predict.py để tránh bị lỗi.
Chạy ứng dụng GUI bằng cách thực hiện lệnh python predict.py từ terminal hoặc Anaconda Prompt.
🎯 Kết luận
Bằng cách theo dõi các bước trên, bạn có thể dễ dàng thiết lập và sử dụng ứng dụng phân loại hình ảnh cho chó và mèo.
