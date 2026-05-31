# Pavo Engine - Todo List Cải Tiến Tính Năng

Danh sách dưới đây được tách theo dạng issue để theo dõi và triển khai cải tiến tính năng mới.

- [x] **Issue 001 - Hoàn thiện cấu trúc core engine theo module**
  - Trạng thái: Đã có các module `convert`, `perception`, `preparation`, `sequancer`, `pavolang`.

- [x] **Issue 002 - Hỗ trợ nhiều loại transition cơ bản**
  - Trạng thái: Đã có fade, slide, wipe, dissolve.

- [x] **Issue 003 - Hỗ trợ text/subtitle overlay với animation**
  - Trạng thái: Đã có `fadeIn`, `fadeOut`, `slideInLeft`, `typewriter` animation.

- [ ] **Issue 004 - Xây dựng module Planner tích hợp LLM**
  - Mục tiêu: Xây dựng `pavo/planner/` để nhận prompt ngôn ngữ tự nhiên và tự động sinh timeline JSON. Cho phép người dùng mô tả video bằng lời và engine tự lên kế hoạch các strip/track.

- [ ] **Issue 005 - Thêm tính năng Emotion Detection**
  - Mục tiêu: Xây dựng `pavo/perception/emotion/` để phân tích cảm xúc từ khuôn mặt trong video, phục vụ cho việc auto-cut và highlight các khoảnh khắc biểu cảm.

- [ ] **Issue 006 - Hỗ trợ xuất video dạng GIF và WebM**
  - Mục tiêu: Mở rộng `render_video` để hỗ trợ output format `gif` và `webm` ngoài `mp4`, với các tùy chọn chất lượng/tốc độ phù hợp.

- [ ] **Issue 007 - Thêm keyframe animation cho text và image**
  - Mục tiêu: Cho phép định nghĩa nhiều keyframe (position, scale, opacity) trong một strip, thay vì chỉ hỗ trợ animation có tên cố định. Phục vụ motion graphics phức tạp.

- [ ] **Issue 008 - Hỗ trợ tự động tạo subtitle từ audio**
  - Mục tiêu: Tích hợp `pavo/perception/speech/transcriber.py` vào pipeline render để tự động sinh subtitle overlay từ file âm thanh mà không cần JSON thủ công.

- [ ] **Issue 009 - Hỗ trợ multi-audio track mixing theo timeline**
  - Mục tiêu: Cho phép nhiều strip audio chạy song song và được mix lại, thay vì chỉ có một soundtrack duy nhất. Hỗ trợ âm thanh cho từng đoạn video clip.

- [ ] **Issue 010 - Thêm hiệu ứng video nâng cao (blur, color grade, vignette)**
  - Mục tiêu: Mở rộng `SUPPORTED_MEDIA_EFFECTS` với các filter FFmpeg như Gaussian blur, LUT color grading, và vignette để tạo phong cách hình ảnh chuyên nghiệp.

- [ ] **Issue 011 - Hỗ trợ nhập timeline từ định dạng EDL/XML của các phần mềm khác**
  - Mục tiêu: Xây dựng parser chuyển đổi EDL (Final Cut / Premiere) hoặc XML sang định dạng timeline JSON của Pavo, giúp interop với các tool editing chuyên nghiệp.

- [ ] **Issue 012 - Tối ưu hiệu năng render: parallel frame processing**
  - Mục tiêu: Tận dụng multi-core để render các frame song song thay vì tuần tự, giảm thời gian render cho timeline dài (> 60 giây).
