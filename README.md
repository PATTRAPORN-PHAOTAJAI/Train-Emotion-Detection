# python 3.11
ถ้าหากไม่ได้ใช้ก็ลองเปลี่ยน py -3.11 เป็น py หรือ python ได้เลย

ลง Library ทั่วไป (Pandas, YOLO, ONNX):
>py -3.11 -m pip install ultralytics pandas kaggle onnx onnxslim

ลง PyTorch เวอร์ชัน GPU (NVIDIA CUDA): (อันนี้สำคัญสุด ทำให้เทรนไว)
>py -3.11 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

เช็กว่าการ์ดจอพร้อมทำงานไหม
>py -3.11 -c "import torch; print(torch.cuda.is_available())"

## วิธีติดตั้งรันไปตามลำดับทีละตัว
1.ติดตั้ง pip และอ่าน requirements.txt
>py -3.11 -m pip install -r requirements.txt</br>

2.โหลดไฟล์ข้อมูลผ่านKaggle
>py -3.11 prepare_data.py</br>

3.เทรนโมเดล
>py -3.11 train.py</br>

3.2เทรนโมเดลต่อกรณีในกรณีที่โค้ดหรือโปรแกรมหยุดการทำงานกระทันหัน
>py -3.11 resume.py</br>

4.ส่งออกข้อมูลที่เทรนเป็นไฟล์.onnxไปที่โฟลเดอร์models
>py -3.11 export_onnx.py
