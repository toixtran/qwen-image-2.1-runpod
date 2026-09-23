# qwen-image-2.1-runpod

[![Runpod](https://api.runpod.io/badge/toixtran/qwen-image-2.1-runpod)](https://console.runpod.io/hub/listing/toixtran/qwen-image-2.1-runpod)

Worker RunPod Serverless cho model [Qwen/Qwen-Image-2.1](https://huggingface.co/Qwen/Qwen-Image-2.1). Một endpoint làm được ba việc:

- **Text-to-image**: sinh ảnh từ prompt, độ phân giải tới 2K.
- **Chỉnh sửa ảnh**: dùng 1–10 ảnh tham chiếu, ví dụ đổi nền, ghép nhiều người vào một ảnh, giữ nguyên nhân vật hoặc sản phẩm.
- **Ảnh trong suốt (RGBA)**: sinh ảnh có kênh alpha, chỉnh sửa layer trong suốt.

## Cấu trúc

```
handler.py            # Handler RunPod, dùng diffusers QwenImage21Pipeline
requirements.txt
Dockerfile            # Build image để deploy thủ công
test_input.json       # Input mặc định khi chạy handler ở local
.runpod/
  hub.json            # Cấu hình RunPod Hub
  tests.json          # Test mà RunPod Hub chạy khi publish
  Dockerfile
examples/client.py    # Client mẫu: gọi endpoint và lưu ảnh
```

## Yêu cầu GPU

Tổng trọng số khoảng **33 GB** (text encoder Qwen3-VL 17.5 GB, transformer 14.2 GB, VAE 1.4 GB).

| GPU | Cấu hình |
|---|---|
| 48 GB (L40S, A6000, RTX 6000 Ada) | `ENABLE_CPU_OFFLOAD=false`, khuyến nghị |
| 80 GB (A100, H100) | `ENABLE_CPU_OFFLOAD=false`, nhanh nhất |
| 24 GB (4090, L4, A5000) | `ENABLE_CPU_OFFLOAD=true`, chậm hơn nhiều |

Nên để container disk từ 80 GB trở lên, hoặc gắn **Network Volume** để không phải tải lại model mỗi lần cold start.

## Deploy

### Cách 1: RunPod Hub

Đưa repo này lên GitHub, rồi publish lên RunPod Hub. Hub đọc cấu hình từ thư mục `.runpod/` và có sẵn hai preset: GPU 48 GB+ và GPU 24 GB có offload.

### Cách 2: Build image rồi tạo endpoint

```bash
# Image nhẹ, model được tải khi worker khởi động
docker build --platform linux/amd64 -t <dockerhub-user>/qwen-image-2.1-runpod:latest .

# Hoặc đóng gói sẵn model vào image (~45 GB), cold start nhanh hơn
docker build --platform linux/amd64 --build-arg BAKE_MODEL=true \
  -t <dockerhub-user>/qwen-image-2.1-runpod:baked .

docker push <dockerhub-user>/qwen-image-2.1-runpod:latest
```

Sau đó vào RunPod Console → Serverless → New Endpoint, chọn image vừa push, chọn GPU 48 GB và đặt các biến môi trường bên dưới.

### Nơi lưu model

Handler tự chọn thư mục cache theo thứ tự sau:

1. Biến `HF_HOME`, nếu bạn đặt.
2. `/models/huggingface`, nếu model đã được bake vào image.
3. `/runpod-volume/huggingface`, nếu có gắn Network Volume.
4. Cache mặc định của Hugging Face.

## Biến môi trường

| Biến | Mặc định | Mô tả |
|---|---|---|
| `HF_MODEL` | `Qwen/Qwen-Image-2.1` | Repo model trên Hugging Face |
| `PRECISION` | `bf16` | `bf16`, `fp16` hoặc `fp32` |
| `ENABLE_CPU_OFFLOAD` | `false` | Bật `enable_model_cpu_offload()` cho GPU ít VRAM |
| `MAX_PIXELS` | `4227072` (2752×1536) | Giới hạn số pixel `width*height` |
| `MAX_IMAGES_PER_JOB` | `4` | Giới hạn `num_images` trong một job |
| `HF_TOKEN` | – | Token Hugging Face, không bắt buộc |
| `HF_HOME` | tự chọn | Thư mục cache model |

## Input

| Trường | Kiểu | Mặc định | Mô tả |
|---|---|---|---|
| `prompt` | string | **bắt buộc** | Mô tả ảnh cần sinh, hoặc chỉ dẫn chỉnh sửa |
| `images` | string[] | – | 1–10 ảnh tham chiếu, dạng URL, base64 hoặc data URI |
| `image` / `image_url` / `image_base64` | string | – | Cách viết ngắn khi chỉ có một ảnh |
| `aspect_ratio` | string | `1:1` (text-to-image) | `1:1`, `4:3`, `3:4`, `3:2`, `2:3`, `16:9`, `9:16`, lấy theo kích thước khuyến nghị của model card |
| `resolution` | string | `2k` | `2k` dùng bảng kích thước của model card (~4 MP). `1k` dùng bản nhỏ (~1 MP, nhanh hơn khoảng 4 lần), xem bảng bên dưới |
| `width`, `height` | int | – | Kích thước tùy chỉnh, được làm tròn xuống bội số của 32. Bị bỏ qua nếu có `aspect_ratio` |
| `num_inference_steps` | int | `40` | Số bước khử nhiễu, từ 1 đến 100 |
| `seed` | int | ngẫu nhiên | Truyền `-1` hoặc bỏ trống để lấy seed ngẫu nhiên |
| `num_images` | int | `1` | Số ảnh sinh ra |
| `transparent` | bool | `false` | Tự thêm template prompt RGBA của model card |
| `negative_prompt` | string | – | Chỉ có tác dụng khi `true_cfg_scale > 1` |
| `true_cfg_scale` | float | `1.0` | Model được thiết kế để chạy không cần CFG. Đặt lớn hơn 1 sẽ chậm gấp khoảng 2 lần |
| `output_resolution` | int | `1024` | Cạnh chuẩn để resize ảnh tham chiếu, và để tính kích thước ảnh ra khi chỉnh sửa mà không truyền `width`/`height` |
| `use_kv_cache` | bool | `true` | Cache KV của prefix. Bật hay tắt sẽ cho ra ảnh khác nhau dù cùng seed |
| `output_format` | string | `png` | `png`, `jpeg` hoặc `webp`. JPEG không có alpha nên ảnh được ghép lên nền trắng |
| `quality` | int | `95` | Chất lượng nén cho jpeg và webp |

### Bảng kích thước theo `resolution`

| `aspect_ratio` | `2k` (model card) | `1k` |
|---|---|---|
| 1:1 | 2048×2048 | 1024×1024 |
| 4:3 | 2400×1792 | 1184×896 |
| 3:4 | 1792×2400 | 896×1184 |
| 3:2 | 2528×1696 | 1248×832 |
| 2:3 | 1696×2528 | 832×1248 |
| 16:9 | 2752×1536 | 1376×768 |
| 9:16 | 1536×2752 | 768×1376 |

Ví dụ: `{"prompt": "...", "aspect_ratio": "16:9", "resolution": "1k"}` sinh ảnh 1376×768.

Khi chỉnh sửa ảnh mà không truyền `width`/`height`/`aspect_ratio`, kích thước ảnh ra được suy từ tỉ lệ của ảnh tham chiếu **cuối cùng**, với diện tích khoảng `output_resolution²`.

## Output

```json
{
  "images": [
    { "image": "<base64>", "format": "png", "width": 2048, "height": 2048, "mode": "RGBA" }
  ],
  "seed": 42,
  "prompt": "...",
  "task": "text-to-image",
  "inference_time": 31.4
}
```

Nếu input sai, handler trả về `{"error": "..."}`.

## Ví dụ

**Text-to-image**

```json
{ "input": {
  "prompt": "A neon shop sign that reads \"QWEN IMAGE 2.1\", rainy night, reflections on wet pavement",
  "aspect_ratio": "16:9",
  "seed": 42
} }
```

**Chỉnh sửa ảnh**

```json
{ "input": {
  "prompt": "Change the background to a sunset beach",
  "image_url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/cat.png"
} }
```

**Dùng nhiều ảnh tham chiếu**

```json
{ "input": {
  "prompt": "A group photo of the people in image 1, image 2 and image 3 standing in a park",
  "images": ["https://.../a.jpg", "https://.../b.jpg", "https://.../c.jpg"],
  "aspect_ratio": "3:2"
} }
```

**Ảnh trong suốt**

```json
{ "input": { "prompt": "A cute cartoon dragon sticker.", "transparent": true } }
```

### Gọi bằng curl

```bash
curl -X POST https://api.runpod.ai/v2/$RUNPOD_ENDPOINT_ID/run \
  -H "Authorization: Bearer $RUNPOD_API_KEY" -H "Content-Type: application/json" \
  -d '{"input": {"prompt": "A red fox in the snow", "aspect_ratio": "16:9"}}'
```

### Gọi bằng client mẫu

```bash
export RUNPOD_API_KEY=... RUNPOD_ENDPOINT_ID=...
python examples/client.py "A red fox in the snow" --aspect-ratio 16:9
python examples/client.py "Change the background to a sunset beach" --image input.png
```

Ảnh 2K ở dạng PNG base64 có thể nặng vài MB. Nếu gặp giới hạn kích thước payload, hãy dùng `output_format: "webp"` hoặc `"jpeg"`.

## Chạy thử ở local (cần GPU)

```bash
pip install -r requirements.txt
python handler.py                    # tự đọc test_input.json
python handler.py --test_input '{"input": {"prompt": "a cat", "width": 1024, "height": 1024}}'
python handler.py --rp_serve_api     # mở API tại http://localhost:8000
```

## License

Code trong repo này dùng tự do. Model Qwen-Image-2.1 tuân theo [Qwen Research License](https://huggingface.co/Qwen/Qwen-Image-2.1/blob/main/LICENSE), cần kiểm tra điều khoản trước khi dùng cho mục đích thương mại.
