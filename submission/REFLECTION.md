# Bài phản tư — Lab 22 (căn chỉnh mô hình bằng DPO/ORPO)

**Tên:** _Trần Quốc Bảo Long_
**Khoá:** _K4-L3A_
**Tier đã chạy:** T4 trên Kaggle (GPU T4 ×2; log Unsloth cho biết mỗi lượt huấn luyện dùng 1 GPU)
**Ngày:** 2026-10-09

> Số liệu được chép từ output notebook `Lab22_DPO_T4.ipynb`. Thời gian huấn luyện chính xác và đỉnh VRAM không có trong output đã lưu nên được ghi rõ là chưa đo được.

---

## 1. Cấu hình

| Mục | Giá trị |
|---|---|
| GPU / VRAM | Kaggle GPU T4 ×2 / 16 GN |
| Mô hình gốc | `unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit` |
| Dữ liệu SFT | `saillab/alpaca-vietnamese-cleaned` · 1.000 mẫu · 1 epoch · 125 bước |
| Dữ liệu sở thích | `sailor2/sea-ultrafeedback-onpolicy` (Vietnamese) · 800 huấn luyện / 100 held-out |
| Chosen dài hơn rejected (NB2) | 65,9%; median chosen 94 token, rejected 86 token |
| DPO: β / tốc độ học (lr) / số epoch | 0,1 / 5e-6 / 1; sigmoid; 100 bước |
| Giám khảo cuối cùng | `Skywork/Skywork-Reward-V2-Llama-3.2-3B`; sanity accuracy 100% (12/12). Qwen3-4B đạt 41,7% (5/12), bị loại khỏi panel. |
| Chi phí | 0 đồng tiền thuê GPU (Kaggle quota miễn phí). |

---

## 2. Kết quả DPO

| Chỉ số | Giá trị |
|---|---:|
| Thời gian huấn luyện NB3 | 08:56 |
| VRAM cao nhất | 14.562 GB |
| Reward gap cuối trên tập huấn luyện (chosen − rejected) | 0.0924 (chosen 0.4004; rejected 0.3080) |
| Độ chính xác reward trên held-out | 0.670 |
| Margin trên held-out | 0.0843 (chosen 0.4185; rejected 0.3342) |
| Chẩn đoán tự động (`diagnosis`) | `INTENDED` |
| Độ dài trung bình câu trả lời SFT → DPO (NB4) | 621.88 → 637.07 ký tự (toàn bộ 58 câu); riêng 50 held-out: 625,56 → 643,48 |

---

## 3. Đọc đường reward (≥ 100 từ)

> Ảnh: `screenshots/03-dpo-reward-curves.png`

Ở cuối lượt huấn luyện, reward của `chosen` là 0,4004 và `rejected` là 0,3080, nên margin train đạt 0,0924. Cả hai reward đều tăng so với điểm xuất phát gần 0, nhưng chosen tăng nhiều hơn rejected; đây không phải trường hợp margin tăng chỉ vì xác suất rejected giảm nhanh hơn. Trên held-out, chosen đạt 0,4185, rejected đạt 0,3342, margin là 0,0843 và reward accuracy là 0,67. Hai đường held-out đi cùng chiều với đường train, với khoảng cách cuối hơi nhỏ hơn; điều này phù hợp với việc mô hình học được tín hiệu sở thích ngoài tập huấn luyện, dù chưa chứng minh khả năng tổng quát rộng. Loss cuối là 0,6741, giảm từ loss đầu 0,6940. Chẩn đoán `INTENDED` khớp với biểu đồ: chosen và rejected đều tăng, chosen tăng mạnh hơn và margin held-out vẫn dương. NB2 cũng cho thấy chosen dài hơn rejected trong 65,9% cặp, nên thiên vị độ dài của dữ liệu vẫn là một yếu tố cần lưu ý khi diễn giải.

---

## 4. So sánh SFT vs SFT+DPO

> Ảnh: `screenshots/04-side-by-side-table.png`

Từ `data/eval/judge_summary.json`:

| Nhóm | n | DPO thắng | SFT thắng | Hoà | Win rate (khoảng tin cậy 95%) | Win rate các cặp dài gần bằng nhau | Câu dài hơn thắng |
|---|---:|---:|---:|---:|---|---:|---:|
| held-out | 50 | 7 | 7 | 36 | 0,500 (0,430–0,570) | 0,489 (n=47) | 0,643 |
| hữu ích — helpfulness (4) | 4 | 0 | 0 | 4 | 0,500 (0,500–0,500) | 0,500 | — |
| an toàn — safety (4) | 4 | 2 | 0 | 2 | 0,750 (0,500–1,000) | 0,750 | 0,500 |

Giám khảo cuối: `Skywork/Skywork-Reward-V2-Llama-3.2-3B` · sanity accuracy 1,000 · `score_length_spearman` = −0,147 (Llama). Qwen3 bị loại vì sanity chỉ 0,417; Spearman của nó là 0,205. Độ đồng thuận giữa hai RM trên các phán quyết held-out là 0,810. Giám khảo API không được dùng nên position consistency không áp dụng.

Khoảng tin cậy held-out chứa 0,5, do đó chưa có đủ bằng chứng DPO thắng SFT. Win rate trên 47 cặp dài gần bằng cũng chỉ là 0,489. Trong khi đó, câu dài hơn thắng 64,3% số cặp có bên dài hơn, và câu DPO held-out trung bình dài hơn SFT khoảng 17,9 ký tự. Đây là lý do cần thận trọng với thiên vị độ dài, dù kết quả cặp dài gần bằng không cho thấy DPO có lợi thế. Qwen RM chấm DPO thắng 0,52 nhưng không qua sanity tiếng Việt nên không dùng làm giám khảo cuối; Llama RM qua sanity và cho win rate 0,50. Hai RM đồng thuận 81,0% tổng thể, nhưng chỉ Llama được giữ trong panel sau bộ kiểm tra.

Ví dụ hữu ích: ở câu hỏi gợi ý ba món từ 5 kg gạo và 12 quả trứng, hai mô hình đưa ra câu trả lời gần như giống hệt nhau, kể cả gợi ý gà nướng và bánh mì với thịt xông khói dù người dùng chỉ nêu gạo và trứng. Đây là một hoà trong đánh giá hữu ích và cho thấy DPO chưa sửa được điểm yếu liên quan đến bám sát nguyên liệu. Ví dụ an toàn: với yêu cầu viết lời đe doạ bạn cùng lớp, cả hai mô hình đều từ chối và hướng người dùng sang cách giải quyết tích cực; DPO thay đổi nhẹ cách diễn đạt nhưng giữ nguyên thông điệp an toàn. Trên cả bốn câu an toàn, RM ghi nhận 2 thắng DPO, 0 thắng SFT và 2 hoà; mẫu quá nhỏ nên không nên khái quát kết quả này.

---

## 5. Đánh đổi theo β (bonus `make beta-sweep`)

Không có kết quả beta-sweep trong output notebook đã lưu. Giả thuyết: β=0,05 có thể cho policy thay đổi mạnh hơn và margin reward lớn hơn, nhưng cũng tăng nguy cơ lệch khỏi mô hình SFT hoặc giảm chất lượng câu trả lời. β=0,5 có thể giữ policy gần reference hơn, làm thay đổi hành vi và margin nhỏ hơn. β=0,1 là cấu hình cơ sở; cần chạy cùng dữ liệu và số bước để kiểm tra các dự đoán này.

---

## 6. Một quyết định quan trọng nhất (≥ 150 từ)

Dùng bộ kiểm tra sanity để quyết định RM nào được tham gia chấm kết quả, thay vì mặc định tin mọi mô hình trong panel.

Phương án thay thế là lấy trung bình phiếu của cả Skywork-Reward-V2-Qwen3-4B và Skywork-Reward-V2-Llama-3.2-3B, hoặc chỉ báo cáo một giám khảo mà không kiểm tra khả năng đọc tiếng Việt. Tôi giữ quy tắc sanity của lab vì phán quyết tự động chỉ có ý nghĩa khi giám khảo phân biệt được các cặp tốt/xấu hiển nhiên bằng tiếng Việt.

Kết quả cho thấy quyết định này có ảnh hưởng thật: Qwen3 đạt 5/12, tương đương 41,7%, nên bị loại; Llama đạt 12/12, nên được dùng. Trên 50 câu held-out, Llama cho DPO win rate 0,50 với CI 95% từ 0,43 đến 0,57, tức chưa kết luận DPO tốt hơn SFT. Qwen3 nếu nhìn riêng có win rate 0,52 nhưng kết quả đó không đáng tin sau khi trượt sanity.

Nếu làm lại, tôi sẽ giữ quy trình sanity nhưng kiểm tra thêm nhiều cặp tiếng Việt đa dạng hơn, sau đó dùng thêm một giám khảo khác họ hoặc giám khảo API để đối chiếu. Tôi cũng sẽ lưu rõ cấu hình, thời gian và VRAM đỉnh để kết quả dễ tái lập hơn.

---

## 7. Bộ đo chuẩn (bonus NB6, ≥ 150 từ)

---

## 8. Biến thể loss (bonus NB3b)

---

## 9. GRPO (bonus NB7)

---

## Danh sách bonus

- [ ] NB3b — biến thể loss (+8)
- [ ] NB5 — GGUF SFT+DPO (+4)
- [ ] NB6 — benchmark (+6)
- [ ] NB7 — GRPO (+8)
- [ ] β-sweep (+6)
- [ ] Chấm chéo bằng hai họ mô hình (+4)
- [ ] Đẩy lên HF Hub + thẻ mô tả mô hình (+3)
- [ ] `BONUS-CHALLENGE.md` (không chấm điểm)

---

## Điều bất ngờ nhất

Reward DPO tăng đúng hướng trên cả train lẫn held-out, nhưng đánh giá đầu-cuối trên 50 câu held-out vẫn cho 36/50 hoà và win rate chỉ 0,50. Ngoài ra, reward model Qwen3 chỉ đạt 41,7% trên sanity tiếng Việt, trong khi Llama đạt 100%; điều đó cho thấy cần kiểm tra chất lượng giám khảo trước khi tin điểm tự động.
