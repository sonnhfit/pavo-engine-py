# Pavo Engine - Todo List Cải Tiến

Danh sách dưới đây được tách theo dạng issue để theo dõi và triển khai cải tiến.

- [x] **Issue 001 - Hoàn thiện cấu trúc core engine theo module**
  - Trạng thái: Đã có các module `convert`, `perception`, `preparation`, `sequancer`, `pavolang`.

- [x] **Issue 002 - Xây dựng bộ test nền tảng cho các chức năng chính**
  - Trạng thái: Đã có test cho render, transitions, subtitle, text overlay, schema validation, video trimming.

- [x] **Issue 003 - Hoàn thiện tài liệu README cho cài đặt và sử dụng cơ bản**
  - Trạng thái: README đã có hướng dẫn cài đặt, ví dụ sử dụng, cấu trúc project và test command.

- [ ] **Issue 004 - Sửa lỗi test transition dissolve đang fail**
  - Mục tiêu: Khắc phục 2 test đang fail trong `tests/test_transitions.py` để ổn định baseline.

- [ ] **Issue 005 - Thiết lập CI tự động chạy test cho pull request**
  - Mục tiêu: Thêm GitHub Actions để tự động chạy test/build khi có PR hoặc push.

- [ ] **Issue 006 - Bổ sung quy chuẩn chất lượng mã (lint + format + typing)**
  - Mục tiêu: Thiết lập công cụ kiểm tra style/chất lượng mã và chuẩn hóa đóng góp.

- [ ] **Issue 007 - Cải thiện đóng gói và phát hành thư viện**
  - Mục tiêu: Chuẩn hóa quy trình build/release, xác thực artifact và tài liệu phát hành.

- [ ] **Issue 008 - Tăng độ phủ test cho perception/planner và các edge case**
  - Mục tiêu: Bổ sung test cho các module AI/perception và các tình huống lỗi dữ liệu đầu vào.

- [ ] **Issue 009 - Nâng cấp tài liệu docs/example theo use case thực tế**
  - Mục tiêu: Thêm walkthrough end-to-end (input assets -> timeline -> output video) và best practices.

- [ ] **Issue 010 - Tối ưu hiệu năng render cho timeline dài**
  - Mục tiêu: Đo benchmark, xác định bottleneck và đề xuất tối ưu CPU/RAM/IO.
