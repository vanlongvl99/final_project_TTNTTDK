# Trí Tuệ Nhân Tạo Trong Điều Khiển
## Tiểu luận: Emotion and Face Recognition
### Nhóm 2
- Nguyễn Văn Long Mssv 1712024
- Huỳnh Mạnh Hưng 
- Lê Hoàng Hiệp
- Hứa Hoàng Thanh Trúc
- Nguyễn Đặng Khôi Nguyên

### Môi trường:
- Ubuntu 18.04
- Python 3.6.9
- Keras 2.3.1
- Tensorflow 2.0.0-alpha0
- Sklearn 0.22.2
- mtcnn 0.1.0

### Chạy chương trình:
- Lưu ý các bạn phải cài đúng các phiên bản của các thư viện trên. Có thể chạy file `requirements.txt` hoặc tự cài riêng từng thư viện: `pip3 install -r requirements.txt`.
- Clone thư mục này về, `cd` vô forder `final_report_TTNT` tương tự như sau: `~/vanlong/ky6/TriTNT/finish_model/final_report_TTNT$`. Tạ thực hiện tất cả các lệnh tại thư mục này.
- Vì file model nặng nên không thể upload lên github, các bạn tải file mode [tại đây](https://drive.google.com/drive/folders/1sxYaEGKeChUC4NlozsdAQZT09vQYId8f?usp=sharing). Sau khi download các file model về, ta lưu các file model trong thư mục model_file. Lưu ý không chỉnh sửa tên các file model, nếu không sẽ bị 1 số lỗi gọi tên file khi chạy chương trình.
- Sau khi cài đặt xong các thư viện và tải file model về, ta chạy lệnh sau để demo kết quả:
`python3 final_source/gui_demo.py`