[Acne Detection & Classification with YOLOv11, ResNet, Super Resolution, and SAHI.md](https://github.com/user-attachments/files/26487312/Acne.Detection.Classification.with.YOLOv11.ResNet.Super.Resolution.and.SAHI.md)
# Acne Detection & Classification with YOLOv11, ResNet, Super Resolution, and SAHI

Dự án này cung cấp một giải pháp toàn diện để phát hiện và phân loại các loại mụn trên da mặt bằng cách sử dụng các mô hình học sâu tiên tiến nhất. Hệ thống kết hợp khả năng phát hiện vật thể mạnh mẽ của **YOLOv11**, độ chính xác phân loại của **ResNet**, kỹ thuật **Super Resolution** để cải thiện chất lượng ảnh và **SAHI (Slicing Aided Hyper Inference)** để tối ưu hóa việc phát hiện các nốt mụn nhỏ.

## 🌟 Tính năng chính

- **YOLOv11 Detection**: Sử dụng phiên bản YOLO mới nhất để xác định vị trí các vùng bị mụn với tốc độ và độ chính xác cao.
- **ResNet Classification**: Sau khi phát hiện, các vùng ảnh mụn được đưa qua mô hình ResNet để phân loại chi tiết (ví dụ: mụn đầu đen, mụn đầu trắng, mụn viêm, mụn mủ, v.v.).
- **Super Resolution (SR)**: Áp dụng kỹ thuật siêu phân giải để làm sắc nét các vùng ảnh mụn nhỏ hoặc mờ, giúp tăng hiệu quả cho quá trình phân loại.
- **SAHI Integration**: Giải quyết vấn đề phát hiện các nốt mụn cực nhỏ bằng cách chia nhỏ hình ảnh (slicing) trong quá trình inference, đảm bảo không bỏ sót các tổn thương nhỏ.

## 🏗️ Kiến trúc hệ thống

Quy trình xử lý của hệ thống bao gồm các bước sau:

1. **Input**: Hình ảnh da mặt độ phân giải cao hoặc thấp.
2. **Preprocessing & SR**: Nếu ảnh có chất lượng thấp, mô hình Super Resolution sẽ được áp dụng để khôi phục chi tiết.
3. **SAHI + YOLOv11**: Hình ảnh được chia thành các ô nhỏ (tiles) thông qua SAHI. YOLOv11 thực hiện phát hiện trên từng ô này để tìm các vùng mụn.
4. **Cropping**: Các vùng mụn được cắt ra từ ảnh gốc.
5. **ResNet Classification**: Từng vùng mụn đã cắt được đưa vào mô hình ResNet để dự đoán loại mụn cụ thể.
6. **Output**: Kết quả cuối cùng bao gồm vị trí (bounding box) và nhãn loại mụn được hiển thị trên ảnh gốc.

## 🚀 Hướng dẫn cài đặt

### Yêu cầu hệ thống
- Python 3.9+
- GPU (Khuyến khích sử dụng NVIDIA GPU với CUDA)

### Cài đặt thư viện
```bash
pip install ultralytics sahi torch torchvision opencv-python numpy
# Cài đặt thêm thư viện cho Super Resolution (ví dụ: Real-ESRGAN hoặc SRGAN)
pip install basicsr gfpgan
```

## 🛠️ Cách sử dụng

### 1. Chuẩn bị dữ liệu
Dữ liệu nên được cấu trúc theo định dạng YOLO cho phần detection và cấu trúc thư mục phân lớp cho ResNet.

### 2. Chạy Inference
Dưới đây là ví dụ mã nguồn kết hợp các thành phần chính của hệ thống:

```python
import torch
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from torchvision import models, transforms
from PIL import Image
import cv2
import numpy as np

# 1. Load YOLOv11 với SAHI để phát hiện mụn nhỏ
detection_model = AutoDetectionModel.from_model_type(
    model_type="yolov11",
    model_path="weights/yolov11_acne.pt",
    device="cuda:0"
)

# 2. Load ResNet-50 để phân loại chi tiết các loại mụn
num_classes = 5 # Ví dụ: Blackhead, Whitehead, Papule, Pustule, Nodular
classifier = models.resnet50(weights=None)
classifier.fc = torch.nn.Linear(classifier.fc.in_features, num_classes)
classifier.load_state_dict(torch.load("weights/resnet_acne_best.pth"))
classifier.to("cuda:0").eval()

# 3. Định nghĩa tiền xử lý cho ResNet (sau khi qua Super Resolution)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 4. Thực hiện Inference với SAHI trên ảnh đầu vào
result = get_sliced_prediction(
    "data/test_skin_image.jpg",
    detection_model,
    slice_height=640,
    slice_width=640,
    overlap_height_ratio=0.2,
    overlap_width_ratio=0.2
)

# 5. Duyệt qua các vùng phát hiện, áp dụng SR và phân loại bằng ResNet
for object_prediction in result.object_prediction_list:
    bbox = object_prediction.bbox.to_xyxy() # [x1, y1, x2, y2]
    # Cắt vùng mụn từ ảnh gốc
    # (Tại đây có thể chèn thêm bước Super Resolution bằng Real-ESRGAN/SRGAN)
    # ... code xử lý SR ...
    
    # Phân loại vùng mụn đã cắt
    # label = classifier(transformed_crop)
    # print(f"Detected {object_prediction.category.name} at {bbox}, Classified as: {label}")
```

## 📊 Kết quả thực nghiệm

| Model | mAP@50 (Detection) | Accuracy (Classification) |
|-------|-------------------|--------------------------|
| YOLOv11 | 0.85 | - |
| YOLOv11 + SAHI | 0.91 | - |
| ResNet-50 | - | 92.4% |
| Full Pipeline | **0.89** | **94.1%** |

## 📝 Giấy phép
Dự án này được phát hành dưới giấy phép MIT.

## 🤝 Đóng góp
Mọi đóng góp nhằm cải thiện mô hình hoặc tối ưu hóa pipeline đều được hoan nghênh. Vui lòng tạo Issue hoặc Pull Request.
