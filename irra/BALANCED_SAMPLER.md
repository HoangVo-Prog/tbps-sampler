# Balanced mixed sampler, version 2

## Phương pháp

Mỗi batch ưu tiên một số cặp cùng PID từ hai `image_id` khác nhau.
Các vị trí còn lại ưu tiên PID chưa có trong batch, để giữ nhiều identity âm.
Khi có cache embedding cố định, quota singleton mặc định là:

- 25% hard: lấy từ các negative có hạng cosine 1 đến 10.
- 25% semi-hard: lấy từ hạng 11 đến 64.
- 50% random: lấy PID còn dữ liệu, rồi chọn một bản ghi.

"Semi-hard" ở đây là dải hạng, không phải điều kiện margin của triplet loss.
Mining dùng cả ảnh→caption, caption→ảnh và caption→caption. PID của negative
phải khác PID của query; mỗi PID singleton chỉ được chọn một lần trong batch.
Query chưa được mining ưu tiên trước để các mẫu khó không tập trung vào một anchor.
Positive khác ảnh chưa chắc khác ngữ nghĩa caption; phiên bản này chưa tối ưu
cosine giữa các caption positive. Negative gần nhau theo encoder cũng có thể
có mô tả giống nhau nhưng khác PID, nên không ưu tiên hard ở mọi vị trí.
Khi pool hard/semi-hard đã hết mẫu khả dụng, sampler chuyển sang random và ghi
số lần fallback. Cache không được cập nhật trong quá trình huấn luyện.

## Coverage và cân bằng PID hiếm

Hai yêu cầu "mỗi dòng đúng một lần" và "số lượt/PID bằng nhau" không thể cùng
đạt nếu dataset có số dòng/PID khác nhau. Vì vậy có hai chế độ riêng:

- `--sampler-exposure coverage` (mặc định): mọi dòng đúng một lần, không bỏ
  phần dư và không oversample. Chế độ này giữ phân bố PID gốc.
- `--sampler-exposure oversample`: mọi dòng gốc vẫn xuất hiện; PID hiếm được
  bổ sung lượt lấy mẫu để đạt quota bên dưới. Epoch có thể dài hơn.

Với `n_i` là số **bản ghi** của PID i, `n_max` là số bản ghi của PID nhiều nhất:

```text
quota_i = ceil(n_i * (n_max / n_i) ** rarity_power)
```

`rarity_power=0` giữ nguyên số dòng; `0.5` giảm chênh lệch;
`1` cho mọi PID đúng `n_max` lượt. Ví dụ `[2, 8, 18, 32]` thành
`[32, 32, 32, 32]` với power 1. Quota tính theo số bản ghi, không phải số ảnh.
Trên RSTPReid cân bằng sẵn, oversample không tạo thêm lượt lấy mẫu.

Phần dư được dùng hết. Nếu không đủ PID/ảnh để giữ quota positive và singleton,
sampler nới ràng buộc và ghi `relaxed_batches`, `pair_shortfall_batches`.
Mẫu trùng ảnh ở phần dư được thử đổi với singleton trong batch trước, giữ
nguyên quota PID và các cặp positive cũ. Trường hợp không sửa được vẫn được
báo trong `unresolved_same_image_positive_pairs`.
Không bảo đảm mọi anchor đều có positive ngoài chính nó.

## Notebook hiện tại

Sau khi lấy source mới, giữ lệnh Cell 2 và Cell 7 hiện tại:

- Cell 2 đo cấu trúc batch; chưa có cache nên mining bị tắt và được ghi rõ.
- Cell 6 vẫn train encoder bằng random, batch 64 nếu dùng T4 16GB.
- Cell 7 audit batch 128, tự tạo cache cosine từ checkpoint và chạy thêm
  `balanced_mixed_metadata` (không mining) và `balanced_mixed` (có mining).

Cell 7 xuất cache ở:

```text
../results/sampler_feature_audit/negative_neighbors.npz
```

Đồng thời, thư mục `composition/` có summary, batch CSV và PID exposure CSV
cho **từng sampler, từng seed**. Các bảng feature dùng trung bình có trọng số
theo số anchor, để batch cuối nhỏ không bị tính ngang một batch đầy đủ.

Để audit cấu trúc có mining sau Cell 7, thêm vào lệnh Cell 2:

```bash
--sampler-mining-cache ../results/sampler_feature_audit/negative_neighbors.npz
```

Để kiểm chứng cân bằng trên dataset lệch tần suất, thêm:

```bash
--sampler-exposure oversample --rarity-power 1
```

Khi train bằng sampler mới, thêm các tham số này vào lệnh train hiện tại:

```bash
--sampler balanced_mixed \
--positive_pairs_per_batch 4 \
--sampler-mining-cache ../results/sampler_feature_audit/negative_neighbors.npz \
--sampler-exposure coverage
```

Cache chỉ dùng training split, kiểm tra fingerprint của PID, image ID, tên ảnh,
caption và thứ tự dòng. Cache của dataset khác hoặc khác thứ tự dòng sẽ bị từ chối.
Hiện hỗ trợ train single-GPU; chưa hỗ trợ DDP cho sampler này.

## Diễn giải kết quả

Hard mining và metric cosine dùng cùng encoder sẽ tăng coverage theo chính
encoder đó. Đây là kiểm chứng cấu trúc sampling, không phải bằng chứng độc lập
về ngữ nghĩa hay chất lượng model. JSON ghi
`mining_same_checkpoint_as_evaluation` để phân biệt trường hợp này.
Muốn kiểm chứng bằng encoder khác, cung cấp `--sampler-mining-cache` đã tạo từ
encoder A trong khi chạy feature audit bằng checkpoint B.

Cache chỉ lưu các neighbor gần nhất, không lưu ma trận cosine toàn bộ dataset.
Khi dữ liệu bị tiêu thụ dần, tỷ lệ mining thực tế có thể nhỏ hơn quota dự kiến;
đọc `hard_draws`, `semi_hard_draws`, `mining_fallback_draws` trong diagnostics.

## Kiểm thử

```bash
python -m unittest discover -s irra/tests -v
python irra/tests/integration_sampler_check.py -v
```

Các test NumPy kiểm tra coverage, phần dư, cân bằng quota PID, khác ảnh,
seed/epoch, cache, lọc cùng PID, mapping ảnh/caption và negative mining.
Lệnh thứ hai cần các dependency IRRA, dùng PyTorch thật trên CPU để kiểm tra
DataLoader và đường đi cosine → cache → sampler → cả hai loại audit.
Kết quả test dữ liệu giả lập không thay thế kết quả audit RSTPReid trên Kaggle.
