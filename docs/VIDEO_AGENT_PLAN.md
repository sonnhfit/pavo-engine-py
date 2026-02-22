# Kế Hoạch Phát Triển Video Editing Agent

## Tổng Quan

Mục tiêu: Xây dựng một agent chỉnh sửa video cho phép người dùng mô tả yêu cầu bằng ngôn ngữ tự nhiên và agent tự động tạo ra timeline JSON rồi render thành video.

Kiến trúc hiện tại của project đã có **Layer 1 (Perception)**, **Layer 3 (Timeline Engine)** và **Layer 4 (Execution)** — phần còn thiếu là **Layer 2 (Planner/Brain)**.

---

## Kiến Trúc Tổng Thể

```
Người dùng (ngôn ngữ tự nhiên)
        ↓
[Layer 2 — Planner]   ← CẦN XÂY DỰNG
  ├── TimelineTools   (công cụ thao tác timeline)
  ├── Director        (LLM gọi công cụ, sinh JSON)
  └── VideoEditingAgent (điều phối toàn bộ pipeline)
        ↓
[Layer 3 — Timeline Engine]  ← ĐÃ CÓ
  └── Sequence / Strip / render.py
        ↓
[Layer 4 — Execution Engine] ← ĐÃ CÓ
  └── render_video / FFmpeg
        ↓
Video đầu ra (.mp4)
```

---

## Danh Sách Issue Cần Làm

### Issue 1: `[planner] Implement TimelineTools — công cụ thao tác timeline`

**Mô tả:**  
Tạo lớp `TimelineTools` trong `pavo/planner/tools.py`. Đây là tầng trung gian giữa LLM và timeline JSON. Mỗi công cụ tương ứng với một thao tác cụ thể trên timeline.

**Công việc cần làm:**
- Tạo file `pavo/planner/__init__.py`
- Tạo file `pavo/planner/tools.py`
- Định nghĩa class `TimelineTools` với các phương thức:
  - `add_video_strip(track_id, src, start, length, effect, ...)` — thêm clip video
  - `add_image_strip(track_id, src, start, length, effect, ...)` — thêm ảnh
  - `add_text_strip(track_id, text, start, length, effect, ...)` — thêm text overlay
  - `set_soundtrack(src, effect)` — đặt nhạc nền
  - `set_output_settings(fps, width, height, format)` — cấu hình đầu ra
  - `set_total_frames(n_frames)` — đặt tổng số frame
  - `get_timeline()` — trả về timeline JSON hoàn chỉnh
  - `dispatch_tool_call(tool_name, tool_args)` — dispatcher cho LLM agent
- Định nghĩa `TIMELINE_TOOL_DEFINITIONS` — danh sách schema OpenAI function-calling cho từng công cụ
- Viết unit test trong `tests/test_planner.py` (không cần API key LLM)

**Acceptance Criteria:**
- `TimelineTools` tạo ra JSON timeline đúng cấu trúc như `docs/data.json`
- `dispatch_tool_call` ném `ValueError` khi tên công cụ không hợp lệ
- `get_timeline()` trả về deep copy để tránh mutate state

---

### Issue 2: `[planner] Implement Director — LLM agent sinh timeline từ ngôn ngữ tự nhiên`

**Mô tả:**  
Tạo lớp `Director` trong `pavo/planner/director.py`. Director nhận prompt ngôn ngữ tự nhiên, chạy vòng lặp agentic với LLM (OpenAI function-calling), lần lượt gọi các công cụ của `TimelineTools` cho đến khi timeline hoàn chỉnh.

**Công việc cần làm:**
- Tạo file `pavo/planner/director.py`
- Implement class `Director`:
  - Constructor: `__init__(model, api_key, max_iterations)`
  - Phương thức chính: `plan(prompt, media_files, fps) → Dict`
  - Vòng lặp agentic: gửi message → nhận tool_calls → thực thi → gửi kết quả → lặp lại cho đến khi LLM không gọi tool nữa
  - System prompt mô tả rõ quy tắc timeline (FPS, track ID, frame numbering)
  - Lazy loading OpenAI client (chỉ import khi cần)
- Thêm `openai>=1.0.0` vào `requirements.txt`

**Acceptance Criteria:**
- `Director.plan()` trả về dict timeline hợp lệ cho `render_video()`
- Giới hạn số vòng lặp bằng `max_iterations` để tránh vòng lặp vô tận
- Raise `RuntimeError` rõ ràng khi LLM API call thất bại
- Raise `ImportError` nếu package `openai` chưa được cài đặt

---

### Issue 3: `[planner] Implement VideoEditingAgent — top-level agent API`

**Mô tả:**  
Tạo lớp `VideoEditingAgent` trong `pavo/planner/agent.py`. Đây là API cấp cao nhất, tích hợp `Director` với `render_video` để cung cấp giao diện end-to-end từ ngôn ngữ tự nhiên ra video.

**Công việc cần làm:**
- Tạo file `pavo/planner/agent.py`
- Implement class `VideoEditingAgent`:
  - Constructor: `__init__(model, api_key, fps)`
  - `plan(prompt, media_files) → Dict` — chỉ sinh timeline, không render
  - `edit(prompt, media_files, output, timeline_save_path) → str` — sinh timeline + render ra video
  - Lazy loading `Director` (tránh import openai khi không cần)
  - Hỗ trợ context manager (`with VideoEditingAgent() as agent: ...`)
- Export public API qua `pavo/planner/__init__.py`

**Acceptance Criteria:**
- `agent.plan()` trả về timeline JSON có thể truyền vào `render_video()` trực tiếp
- `agent.edit()` ghi file video ra đường dẫn `output`
- `timeline_save_path` tùy chọn: nếu truyền vào thì lưu JSON timeline ra file để debug/tái sử dụng
- File JSON tạm thời bị xóa sau khi render xong

---

### Issue 4: `[planner] Tích hợp Perception layer vào pipeline agent`

**Mô tả:**  
Cho phép agent tự động phân tích video/audio đầu vào bằng các module perception hiện có, rồi đưa kết quả phân tích vào context của LLM để sinh timeline chính xác hơn.

**Công việc cần làm:**
- Mở rộng `VideoEditingAgent.edit()` với tham số `analyze_input=True`
- Tích hợp `SceneDetector` để phát hiện scene boundaries từ video đầu vào
- Tích hợp `SpeechTranscriber` để lấy nội dung lời thoại và timestamp
- Tích hợp `SpeakerDiarization` để phân biệt người nói trong video
- Truyền kết quả phân tích vào system prompt / user message của LLM
- Ví dụ context gửi cho LLM:
  ```
  Video analysis:
  - Scenes: 3 scenes detected (0-5s, 5-12s, 12-20s)
  - Transcript: "Hello everyone [0.0s], today we discuss AI [2.1s], ..."
  - Speakers: Speaker A (0-8s), Speaker B (8-20s)
  ```

**Acceptance Criteria:**
- `analyze_input=True` tự động chạy perception pipeline trước khi gọi LLM
- Kết quả perception được format và đưa vào LLM context
- Có thể bỏ qua (`analyze_input=False`) để tiết kiệm thời gian khi không cần

---

### Issue 5: `[planner] Thêm công cụ trim và cắt ghép video cho agent`

**Mô tả:**  
Mở rộng `TimelineTools` với các công cụ hỗ trợ cắt ghép nâng cao để LLM có thể điều khiển chính xác hơn việc lấy đoạn video nào.

**Công việc cần làm:**
- Thêm tham số `video_start_frame` và `video_end_frame` vào `add_video_strip`
- Thêm tool mới `trim_video_strip(track_id, strip_index, start_frame, end_frame)` để cắt strip đã thêm
- Thêm tool `reorder_strips(track_id, new_order)` để sắp xếp lại thứ tự các strip trong track
- Cập nhật `TIMELINE_TOOL_DEFINITIONS` với schema tương ứng
- Viết unit test bổ sung

---

### Issue 6: `[planner] Hỗ trợ nhiều LLM provider (không chỉ OpenAI)`

**Mô tả:**  
Tách phần gọi LLM ra interface chung để có thể swap provider (OpenAI, Anthropic, local Ollama, v.v.) mà không cần thay đổi logic agent.

**Công việc cần làm:**
- Định nghĩa abstract interface `BaseLLMClient` trong `pavo/planner/llm_client.py`
- Implement `OpenAIClient(BaseLLMClient)`
- Implement `OllamaClient(BaseLLMClient)` cho local/offline inference
- `Director` nhận `llm_client` thay vì hard-code OpenAI
- Document cách add provider mới

---

### Issue 7: `[planner] Thêm ví dụ sử dụng agent`

**Mô tả:**  
Tạo file ví dụ trong `docs/example/` để người dùng mới có thể nhanh chóng thử agent.

**Công việc cần làm:**
- Tạo `docs/example/VIDEO_EDITING_AGENT_EXAMPLE.py` với các ví dụ:
  - Tạo slideshow từ ảnh
  - Thêm subtitle vào video
  - Cắt ghép nhiều clip với transition
- Cập nhật `README.md` với section "Using the Video Editing Agent"
- Thêm ví dụ JSON timeline output để người dùng hiểu format

---

## Thứ Tự Ưu Tiên

| Thứ tự | Issue | Phụ thuộc |
|--------|-------|-----------|
| 1 | Issue 1 — TimelineTools | Không có |
| 2 | Issue 2 — Director | Issue 1 |
| 3 | Issue 3 — VideoEditingAgent | Issue 1, 2 |
| 4 | Issue 5 — Trim tools | Issue 1 |
| 5 | Issue 4 — Perception integration | Issue 3 |
| 6 | Issue 6 — Multi-provider LLM | Issue 2 |
| 7 | Issue 7 — Examples & docs | Issue 3 |

---

## Cấu Trúc File Sau Khi Hoàn Thành

```
pavo/
└── planner/
    ├── __init__.py         # Export: VideoEditingAgent, Director, TimelineTools
    ├── tools.py            # TimelineTools + TIMELINE_TOOL_DEFINITIONS
    ├── director.py         # Director (LLM agentic loop)
    ├── agent.py            # VideoEditingAgent (top-level API)
    └── llm_client.py       # BaseLLMClient + OpenAIClient + OllamaClient (Issue 6)

docs/example/
└── VIDEO_EDITING_AGENT_EXAMPLE.py   # Ví dụ sử dụng (Issue 7)

tests/
└── test_planner.py         # Unit tests cho TimelineTools
```
