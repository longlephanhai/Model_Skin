import torch
import numpy as np
from PIL import Image
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer


device = torch.device(
    'cuda') if torch.cuda.is_available() else torch.device('cpu')

model_path = './RealESRGAN_x4plus.pth'


model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                num_block=23, num_grow_ch=32, scale=4)


upsampler = RealESRGANer(
    scale=4,
    model_path=model_path,
    model=model,
    tile=0,
    pre_pad=0,
    half=False
)

img = Image.open('./test-2.png').convert('RGB')
img = np.array(img)

output, _ = upsampler.enhance(img, outscale=4)

output_img = Image.fromarray(output)
output_img.save('./test_output.jpg')