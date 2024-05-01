# import kornia
import cv2
import torch.nn as nn
import torch
import torch.nn.functional as F
import kornia.color.lab

# @torch.no_grad()
# def image2grad(image, der_operator='sobel', out_shape=(224,224)):
#     grad = kornia.filters.spatial_gradient(
#         image, mode=der_operator, order=1, normalized=True).squeeze(0)
#     grad = torch.norm(grad, dim=2)
#     grad = F.interpolate(grad, size=out_shape, mode='area')
#     grad = torch.mean(grad, dim=1, keepdim=True)
#     grad = grad.repeat(1,3,1,1)
#     return grad

def prepare_simplified_sketch(x):
    x = x * 2 - 1
    x *= -1
    x = F.interpolate(x, size=(224,224), mode='bicubic')
    x[x > 0] = 1.0
    x[x <= 0] = -1.0
    return x

class SobelConv(nn.Module):
    def __init__(self):
        super(SobelConv, self).__init__()
        kernel_v = [[0, -1, 0],
                    [0, 0, 0],
                    [0, 1, 0]]
        kernel_h = [[0, 0, 0],
                    [-1, 0, 1],
                    [0, 0, 0]]
        kernel_h = torch.FloatTensor(kernel_h).unsqueeze(0).unsqueeze(0)
        kernel_v = torch.FloatTensor(kernel_v).unsqueeze(0).unsqueeze(0)
        self.weight_h = nn.Parameter(data=kernel_h, requires_grad=False)
        self.weight_v = nn.Parameter(data=kernel_v, requires_grad=False)
        self.immean = 0.9664114577640158
        self.imstd = 0.0858381272736797

    def get_gray(self,x):
        ''' 
        Convert image to its gray one.
        '''
        gray_coeffs = [65.738, 129.057, 25.064]
        convert = x.new_tensor(gray_coeffs).view(1, 3, 1, 1) / 256
        x_gray = x.mul(convert).sum(dim=1)
        return x_gray.unsqueeze(1)

    def gradient(self, x):
        x_v = F.conv2d(x, self.weight_v, padding=1)
        x_h = F.conv2d(x, self.weight_h, padding=1)
        x = torch.sqrt(torch.pow(x_v, 2) + torch.pow(x_h, 2) + 1e-6)
        return x

    def binary_image2sketch(self, x, thr=0.2):
        edge = self.gradient(self.get_gray(x))
        edge[edge<thr] = 0
        edge[edge>=thr] = 1
        return edge

    def __call__(self, x, amplify_ratio=3.5, normalize=True):

        if x.shape[1] == 3:
            x = self.get_gray(x)

        x = self.gradient(x)
        
        xa = 1 - 3.75*x
        if normalize is True:
            return (xa - self.immean) / self.imstd
        else:
            return xa


### Will introduce some artifacts and training leads to Nan
### input is rgb from [0,1]
### output is lab where l ranges from -1 to 1
# def rgb_to_lab(srgb):
    
# 	srgb_pixels = torch.reshape(srgb, [-1, 3])

# 	linear_mask = (srgb_pixels <= 0.04045).type(torch.FloatTensor).to(srgb.device)
# 	exponential_mask = (srgb_pixels > 0.04045).type(torch.FloatTensor).to(srgb.device)
# 	rgb_pixels = (srgb_pixels / 12.92 * linear_mask) + (((srgb_pixels + 0.055) / 1.055) ** 2.4) * exponential_mask
	
# 	rgb_to_xyz = torch.tensor([
# 				#    X        Y          Z
# 				[0.412453, 0.212671, 0.019334], # R
# 				[0.357580, 0.715160, 0.119193], # G
# 				[0.180423, 0.072169, 0.950227], # B
# 			]).type(torch.FloatTensor).to(srgb.device)
	
# 	xyz_pixels = torch.mm(rgb_pixels, rgb_to_xyz)
	

# 	# XYZ to Lab
# 	xyz_normalized_pixels = torch.mul(xyz_pixels, torch.tensor([1/0.950456, 1.0, 1/1.088754]).type(torch.FloatTensor).to(srgb.device))

# 	epsilon = 6.0/29.0

# 	linear_mask = (xyz_normalized_pixels <= (epsilon**3)).type(torch.FloatTensor).to(srgb.device)

# 	exponential_mask = (xyz_normalized_pixels > (epsilon**3)).type(torch.FloatTensor).to(srgb.device)

# 	fxfyfz_pixels = (xyz_normalized_pixels / (3 * epsilon**2) + 4.0/29.0) * linear_mask + ((xyz_normalized_pixels+0.000001) ** (1.0/3.0)) * exponential_mask
# 	# convert to lab
# 	fxfyfz_to_lab = torch.tensor([
# 		#  l       a       b
# 		[  0.0,  500.0,    0.0], # fx
# 		[116.0, -500.0,  200.0], # fy
# 		[  0.0,    0.0, -200.0], # fz
# 	]).type(torch.FloatTensor).to(srgb.device)
# 	lab_pixels = torch.mm(fxfyfz_pixels, fxfyfz_to_lab) + torch.tensor([-16.0, 0.0, 0.0]).type(torch.FloatTensor).to(srgb.device)
# 	#return tf.reshape(lab_pixels, tf.shape(srgb))
# 	return torch.reshape(lab_pixels, srgb.shape)



def unit_test():
    import torchvision
    sobel_conv = SobelConv().cuda()
    x_path = '/home/vis/sunyasheng/Dataset/FFHQ_sample/00007.png'
    x_np = cv2.imread(x_path)/255.
    x = torch.from_numpy(x_np).cuda().float()
    x = x.permute(2,0,1).unsqueeze(0)
    sobel = sobel_conv(x, normalize=False)
    torchvision.utils.save_image(sobel/255, 'sobel_torch.jpg')
    import pdb; pdb.set_trace()





if __name__ == '__main__':
    # unit_test()
    img_path = '/home/vis/sunyasheng/Dataset/FFHQ_sample/00000.png'
    bgr = cv2.imread(img_path)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb_ts = torch.from_numpy(rgb)
    rgb_ts = rgb_ts / 255.
    rgb_ts = rgb_ts.unsqueeze(0)
    rgb_ts = rgb_ts.permute(0,3,1,2)
    # lab = rgb_to_lab(rgb_ts)
    lab = kornia.color.lab.rgb_to_lab(rgb_ts)
    import pdb; pdb.set_trace();
