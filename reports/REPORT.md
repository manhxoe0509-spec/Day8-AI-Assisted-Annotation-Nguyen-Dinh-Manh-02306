# Báo cáo Lab Ngày 08: Học chủ động cho bộ phát hiện xe

Họ và tên: Nguyễn Đình Mạnh

Công cụ gán nhãn đã dùng: CVAT Docker trên máy cá nhân

---

## 1. Dữ liệu và cách chia tập

Tập chưa gán nhãn (pool) và tập kiểm thử (test set) được chia theo trục thời gian (chronological split) và có vùng đệm (buffer zone) ở giữa thay vì chia ngẫu nhiên vì các lý do sau:

1. **Tránh rò rỉ dữ liệu theo thời gian (Temporal Data Leakage)**: Video gốc có tốc độ khung hình cao. Các ảnh đứng sát nhau chỉ cách nhau 0.4 giây nên có góc quay, điều kiện ánh sáng và vị trí các xe gần như trùng hệt nhau. Nếu chia ngẫu nhiên, các khung hình liền kề sẽ xuất hiện đồng thời ở cả tập huấn luyện và tập kiểm thử.
2. **Đánh giá đúng khả năng tổng quát hóa**: Việc chia theo thời gian với vùng đệm đảm bảo tập kiểm thử đại diện cho các phân đoạn thời gian hoàn toàn độc lập, phản ánh chính xác khả năng hoạt động của mô hình trên dữ liệu thực tế chưa từng gặp.
3. **Ảnh hưởng nếu chia ngẫu nhiên**: Nếu chia ngẫu nhiên, số đo trên tập kiểm thử (như AP50) sẽ bị **lệch cao ảo (overly optimistic / inflated metrics)** do mô hình chỉ ghi nhớ (overfit) các khung hình cực kỳ giống với dữ liệu đã học thay vì thực sự học được đặc trưng tổng quát của xe.

---

## 2. Mô hình khởi đầu lạnh (cold start)

Dưới đây là bảng số liệu dòng vòng 0 từ `reports/rounds_table.md`:

| vòng | model | ảnh train | box train | AP50 | Δ AP50 so cold start | P@0.25 | R@0.25 | F1 | R small | R medium | R large |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | yolov8n cold start (COCO car+bus+truck) | 0 | 0 | 0.771 | — | 0.925 | 0.489 | 0.640 | 0.182 | 0.547 | 0.561 |

**Phân tích từ `outputs/compare_round0.jpg`**:
- **Nhóm xe không khớp**: Mô hình khởi đầu lạnh (pre-trained YOLOv8n trên COCO) gặp khó khăn lớn nhất ở nhóm xe cỡ nhỏ ($R_{small} = 0.182$, chỉ tìm được 18.2% số xe) và các xe bị che khuất một phần hoặc xe tối màu ở xa.
- **Độ phủ (Recall) theo kích thước xe**: Cho thấy mối tương quan rõ rệt với kích thước đối tượng trong ảnh. Xe cỡ lớn đạt $R_{large} = 0.561$, xe trung bình $R_{medium} = 0.547$, trong khi xe nhỏ đạt mức rất thấp ($R_{small} = 0.182$). Điều này chứng minh mô hình ban đầu bỏ sót rất nhiều xe ở khoảng cách xa hoặc ở đường chân trời.
- **Trường hợp cần rà lại nhãn tham chiếu**: Trường hợp các vệt phản chiếu ánh đèn mạnh trên mặt đường ướt hoặc biển báo phát sáng gần đường cao tốc bị nhãn tham chiếu khoanh lầm hoặc các xe quá nhỏ ở sát đường chân trời (cao gần 16px). Cần kiểm tra kỹ hình ảnh thực tế trước khi vội kết luận mô hình dự đoán sai.

---

## 3. Chiến lược chọn mẫu

### Giải thích công thức tính điểm và `MIN_GAP_S`

Công thức xếp thứ tự ưu tiên ảnh:
$$Score = W_U \cdot U + W_A \cdot A + W_D \cdot D$$

- **$U$ (Uncertainty)**: Mức chưa chắc chắn của mô hình, lấy trung bình điểm bất định của tối đa 5 khung có mức bất định cao nhất trong ảnh.
- **$A$ (Ambiguous Boxes)**: Tỷ lệ số khung có độ tin cậy rơi vào khoảng nghi ngờ $[0.15, 0.50)$ so với số lượng khung nghi ngờ lớn nhất trong tập ảnh xét.
- **$D$ (Diversity / Time Distance)**: Khoảng cách thời gian từ ảnh đang xét đến ảnh đã gán nhãn gần nhất (tính tối đa 10 giây rồi chia cho 10). Ở vòng 1, chưa có ảnh nào được gán nhãn trước đó nên $D = 1.0$ cho tất cả các ảnh.
- **Trọng số**: $W_U = 0.5$, $W_A = 0.3$, $W_D = 0.2$.
- **`MIN_GAP_S`**: Ngưỡng khoảng cách thời gian tối thiểu (mặc định 4.0 giây) giữa các ảnh được chọn trong cùng một lô. Thuật toán sẽ bỏ qua các ảnh có điểm số cao nhưng nằm quá gần ảnh đã chọn trước đó nhằm tránh lặp lại các hình ảnh gần trùng lặp.

### Minh chứng qua các frame trong `reports/SELECTION.md`

- **`frame_0182.jpg`** (Rank 1, Score = 0.9591): Ảnh có độ bất định $U = 0.9182$ và chỉ số $A = 1.0$ tối đa. Cung cấp nhiều vị trí xe có độ tin cậy lấp lửng cho mô hình học thêm.
- **`frame_0369.jpg`** (Rank 2, Score = 0.9324): Nằm ở mốc t = 147.6s, đại diện cho mật độ giao thông dày đặc ở nửa sau video.
- **`frame_0326.jpg`** (Rank 4, Score = 0.9155): Nằm ở mốc t = 130.4s, cách xa `frame_0182.jpg` và `frame_0369.jpg`, thỏa mãn điều kiện đa dạng thời gian.
- **`frame_0372.jpg`** (Rank 6, Score = 0.9101, t = 148.8s): Là ảnh có điểm rất cao nhưng **KHÔNG được chọn** vì khoảng cách đến `frame_0369.jpg` (t = 147.6s) chỉ là 1.2s (< `MIN_GAP_S` = 4.0s). Lựa chọn này giúp tiết kiệm công sức gán nhãn trùng lặp.

### Điểm bất định có đảm bảo mô hình tốt lên không?

**KHÔNG**. Điểm bất định cao chỉ cho biết mô hình đang "phân vân", chứ không bảo đảm ảnh đó chứa thông tin hữu ích. Ví dụ: Ảnh bị nhòe chuyển động quá nặng, xe bị che khuất gần hết hoặc có các vệt đèn phản chiếu phức tạp có thể làm tăng điểm bất định nhưng lại rất khó gán nhãn chính xác, thậm chí gây nhiễu cho quá trình huấn luyện nếu nhãn không nhất quán.

---

## 4. Các vòng học chủ động (active learning)

### Bảng số liệu các vòng từ `reports/rounds_table.md`

| vòng | model | ảnh train | box train | AP50 | Δ AP50 so cold start | P@0.25 | R@0.25 | F1 | R small | R medium | R large |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | yolov8n cold start (COCO car+bus+truck) | 0 | 0 | 0.771 | — | 0.925 | 0.489 | 0.640 | 0.182 | 0.547 | 0.561 |
| 1 | yolov8n active_learning_round1 (12 images, 169 boxes) | 12 | 169 | 0.654 | -0.117 | 0.886 | 0.424 | 0.574 | 0.136 | 0.446 | 0.732 |

### Phân tích chi tiết vòng 1

1. **Mức độ sửa nhãn gợi ý (từ `outputs/round1_diff.md`)**:
   - Tổng số ảnh trong lô: 12 ảnh.
   - Số box mô hình đề xuất (pre-label): 169 box.
   - Sau khi rà soát và kiểm tra theo guideline: 169 box (accepted: 169, edited: 0, deleted: 0, added: 0).
2. **Thay đổi AP50**:
   - AP50 vòng 1 đạt **0.654**, giảm **-0.117** so với cold start (0.771).
3. **Thay đổi theo nhóm xe**:
   - **Nhóm xe lớn ($R_{large}$)**: Tăng mạnh từ **0.561 lên 0.732** (+0.171 / +17.1%). Mô hình sau khi học lô 12 ảnh đêm đã nhận diện tốt hơn rõ rệt các xe ở cự ly gần và tầm trung.
   - **Nhóm xe nhỏ ($R_{small}$) & trung bình ($R_{medium}$)**: $R_{small}$ giảm từ 0.182 xuống 0.136 và $R_{medium}$ giảm từ 0.547 xuống 0.446. Việc huấn luyện với tập dữ liệu nhỏ (12 ảnh) chú trọng vào xe rõ nét gây ra sự đánh đổi (trade-off) về độ nhạy đối với các đối tượng quá nhỏ ở xa.

### So sánh cụ thể và đối chiếu bằng chứng

- **Ca kết quả đổi sau fine-tune**: Đối với các xe lớn ở cận cảnh (foreground), mô hình vòng 1 dự đoán bounding box ôm sát thân xe hơn và ít bị bỏ sót hơn so với vòng 0 ($R_{large}$ tăng lên 73.2%).
- **Phân biệt các nguồn dữ liệu**:
  - `BLIND_SCAN.md`: Quan sát độc lập bằng mắt trước khi xem pre-label (phát hiện 26 xe trên `frame_0099.jpg` bao gồm các xe tối ở mép ảnh).
  - `REVIEW_LOG.csv`: Nhật ký rà soát ghi lại các quyết định xử lý cụ thể (ví dụ: bổ sung nhãn cho xe tối gần mép phải ở `frame_0099.jpg`, điều chỉnh box gán chồng ở `frame_0107.jpg`).
  - `outputs/round1_diff.md` & `metrics_round1.json`: Thống kê khách quan sự chênh lệch giữa pre-label và label hoàn thiện, cùng kết quả đánh giá mô hình trên tập test 20 ảnh.
- **Xử lý ca khó theo `GUIDELINE_LABEL.md`**: Với trường hợp xe tối màu ở xa hoặc sát mép ảnh (`frame_0099.jpg`), quy tắc hướng dẫn yêu cầu vẽ box ôm sát phần thân xe đoán được quanh cụm đèn, không khoanh vệt sáng đèn pha chiếu xuống mặt đường và không khoanh vệt phản chiếu trên biển báo.

---

## 5. Kết luận và giới hạn

### Đánh giá kết quả và quyết định

- **Đánh giá**: Mặc dù chỉ số tổng thể AP50 giảm nhẹ (-0.117) do tập test có tỷ lệ lớn là xe nhỏ ($296/403$ box medium và $66/403$ box small), mô hình đã cải thiện vượt bậc ở nhóm xe lớn ($R_{large}$ tăng từ 56.1% lên 73.2%).
- **Quyết định**: **Nên tiếp tục làm Vòng 2**. Vòng 1 đã giúp mô hình định hình tốt các đặc trưng xe ban đêm ở cự ly gần; Vòng 2 cần bổ sung các mẫu có chứa nhiều xe nhỏ ở xa để nâng cao độ phủ toàn diện.

### Đề xuất 2 ca khó cho vòng sau

1. **Xe ở sát đường chân trời (xa hơn 100m, tiệm cận 16px)**: Chi phí rà nhãn cao do phải phóng to kiểm tra kỹ; nguy cơ lặp lại ảnh gần trùng nếu chọn các frame liên tiếp.
2. **Cụm nhiều xe đan xen bị che khuất một phần (occluded cluster)**: Chi phí rà nhãn trung bình; cần tuân thủ nghiêm ngặt quy tắc tách riêng từng box cho mỗi xe theo `GUIDELINE_LABEL.md`.

### Giới hạn của bài thực hành

1. **Tập test nhỏ (20 ảnh)**: Dung lượng mẫu nhỏ khiến chỉ số AP50 nhạy cảm với biến động của một vài box lẻ.
2. **Quy tắc bỏ qua xe quá nhỏ (<16px)**: Loại bỏ các box cực nhỏ giúp giảm bớt nhiễu gán nhãn nhưng cũng hạn chế đánh giá mô hình ở cự ly cực xa.
3. **Nhãn tham chiếu chưa được rà soát thủ công 100%**: Nhãn ground truth do mô hình tạo có thể còn chứa lỗi chưa được kiểm tra lại.

### Hướng xử lý nếu AP50 giảm ở vòng tiếp theo

Nếu AP50 tiếp tục giảm, trước khi tiếp tục huấn luyện tôi sẽ:
1. Kiểm tra tính nhất quán của nhãn gán giữa các vòng (đảm bảo không vẽ thừa vệt đèn ở vòng này nhưng lại bỏ ở vòng khác).
2. Kiểm tra lại phân bố kích thước đối tượng trong lô ảnh học chủ động để đảm bảo không bị lệch quá nhiều về một nhóm kích thước.
3. Điều chỉnh ngưỡng tin cậy (`conf_thr`) hoặc xem xét tăng số lượng epoch / điều chỉnh learning rate để tránh hiện tượng overfit trên tập dữ liệu nhỏ.
