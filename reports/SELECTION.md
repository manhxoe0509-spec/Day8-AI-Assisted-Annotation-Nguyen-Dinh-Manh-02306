# Vì sao chọn lô này?

Trong 50 dòng đứng đầu `outputs/selection_round1.csv`, nếu chỉ có ngân sách rà 5 ảnh, tôi sẽ ưu tiên chọn 5 frame sau:
1. `frame_0182.jpg` (Rank 1, Score = 0.9591, t_sec = 72.8s): Ảnh có độ bất định cao ($U = 0.9182$) và mật độ box độ tin cậy trung bình đạt mức tối đa ($A = 1.0$), nằm ở giữa đoạn video giúp đại diện tốt cho khu vực giữa.
2. `frame_0369.jpg` (Rank 2, Score = 0.9324, t_sec = 147.6s): Ảnh ở khoảng thời gian muộn hơn, có độ bất định cao ($U = 0.9315$) và chứa nhiều đối tượng xe ở làn bên phải.
3. `frame_0380.jpg` (Rank 3, Score = 0.9170, t_sec = 152.0s): Điểm bất định rất cao ($U = 0.9340$), cách `frame_0369.jpg` 4.4s (đạt khoảng cách `MIN_GAP_S` = 4.0s) để tránh lặp trùng dữ liệu nhưng vẫn thu thập thông tin mới.
4. `frame_0099.jpg` (Rank 8, Score = 0.9063, t_sec = 39.6s): Điểm bất định $U = 0.9460$, nằm ở đoạn đầu video (t = 39.6s) giúp đảm bảo sự đa dạng theo thời gian và chứa xe tối gần mép phải.
5. `frame_0392.jpg` (Rank 15, Score = 0.8874, t_sec = 156.8s): Có mức chưa chắc chắn cao nhất trong toàn bộ tập xét ($U = 0.9747$), chứa nhiều xe ở xa bị che khuất mà AI dễ bỏ sót (trường hợp đối tượng bất định cực đại).

---

Ba frame thuộc lô 12 ảnh model chọn và bằng chứng trong CSV/ảnh contact sheet (`outputs/selection_round1.jpg`):
1. `frame_0182.jpg` (Rank 1, Score = 0.9591, t = 72.8s, $U = 0.9182$, $A = 1.0$, $D = 1.0$): Mô hình dự đoán nhiều box có độ tin cậy từ 0.15 đến dưới 0.50 (thành phần $A$ lớn nhất). Trên contact sheet, đây là cảnh đường cao tốc đêm đông đúc với nhiều vệt đèn chiếu lọt vào vùng bất định.
2. `frame_0369.jpg` (Rank 2, Score = 0.9324, t = 147.6s, $U = 0.9315$, $A = 0.8889$, $D = 1.0$): Thuộc phân đoạn cuối video, chứa nhiều xe di chuyển ở làn bên phải với góc nhìn chéo.
3. `frame_0326.jpg` (Rank 4, Score = 0.9155, t = 130.4s, $U = 0.9310$, $A = 0.8333$, $D = 1.0$): Đại diện cho phân đoạn 130s, nơi các phương tiện di chuyển với mật độ cao và có nhiều xe phía xa.

---

Một frame có điểm cao nhưng không chọn hoặc một frame có điểm thấp vẫn nên xem, và lý do:
- Frame có điểm cao nhưng KHÔNG được chọn: `frame_0372.jpg` (Rank 6, Score = 0.9101, t = 148.8s).
- Lý do: Mặc dù đứng thứ 6 về điểm số, ảnh này bị loại bỏ do thuật toán lọc khoảng cách thời gian (`MIN_GAP_S` = 4.0s). Ảnh `frame_0369.jpg` (t = 147.6s) đã được chọn trước đó, khoảng cách giữa 2 ảnh chỉ là 1.2s (< 4.0s), dẫn đến nội dung ảnh gần như trùng lặp hoàn toàn. Việc bỏ qua ảnh này giúp tiết kiệm công sức gán nhãn trùng lặp mà không làm mất thông tin mới.

---

Điều phép chọn này chưa chứng minh về chất lượng mô hình:
- Thuật toán chọn ảnh dựa trên độ bất định ($U$) và mật độ box nghi ngờ ($A$) chỉ thể hiện **mức độ phân vân của mô hình hiện tại**, chứ KHÔNG đảm bảo ảnh đó sẽ chứa thông tin sạch hoặc giúp mô hình tốt lên sau khi fine-tune.
- Các ảnh có điểm bất định rất cao thường gặp các vấn đề như: xe bị nhòe chuyển động quá nặng, ánh đèn phản chiếu trên mặt đường nhầm thành đèn xe, xe bị che khuất gần hết hoặc ảnh quá tối. Việc bổ sung các ảnh nhiễu này nếu không gán nhãn cẩn thận theo quy chuẩn có thể gây nhiễu cho mô hình và khiến chỉ số đánh giá (AP50) không tăng hoặc giảm.
