## Predefined functions for Angular Spectrum based diffraction simulaiton
# first established at 2025. 02.25
# first upload at 2025. 02.28
# Contact : Yong Guk Kang

import torch
import torch.nn.functional as F
import torch.nn as nn
import matplotlib.pyplot as plt
import scipy.io as sio
import cv2
import math
import os
import datetime
import pickle
import tifffile
import numpy as np
import h5py


## predefined functions

def to_tensor(x):
    return x if isinstance(x, torch.Tensor) else torch.tensor(x)

def _ensure_single_device(*args):
    # 하나라도 GPU에 있으면 해당 GPU device를 타겟으로 설정
    target_device = 'cpu'
    for arg in args:
        if isinstance(arg, torch.Tensor) and arg.device.type == 'cuda':
            target_device = arg.device
            break
    # 모든 텐서를 target_device로 이동
    new_args = []
    for arg in args:
        if isinstance(arg, torch.Tensor):
            new_args.append(arg.to(target_device))
        else:
            new_args.append(arg)
    return tuple(new_args)
## simulation related ##########################################

def image_grid(dimX, dimY, Nx, Ny):
    xx = torch.linspace(-(dimX*0.5), (dimX*0.5), Nx)
    yy = torch.linspace(-(dimY*0.5), (dimY*0.5), Ny)
    XX, YY = torch.meshgrid(xx, yy, indexing='xy')
    return XX, YY
                       
def fourier_grid(dimX, dimY, Nx, Ny):
    # dX, dY = dimX/Nx, dimY/Ny
    # Frequency space coordinates
    fx = torch.fft.fftshift(torch.fft.fftfreq(Nx, d=(dimX/Nx)))
    fy = torch.fft.fftshift(torch.fft.fftfreq(Ny, d=(dimY/Ny)))
    FX, FY = torch.meshgrid(fx, fy, indexing='xy')
    return  FX, FY

def propASKernel(FX, FY, xRad, yRad, k, dz):
    FX, FY, xRad, yRad, k, dz = _ensure_single_device(FX, FY, xRad, yRad, k, dz)

    #  propagation_kernel(FX, FY, xDeg, yDeg, wavelen, dz), xDeg, yDeg = 0 means normal kernel    
    # H(f_x, f_y) = exp{ i*(2π/wavelen)*dz * sqrt(1 - [wavelen * f_x]^2 - [wavelen * f_y]^2) }
    # f_x_tilted = fx + FXoff = fx - (sin(X)/lambda)    
    # wavelen = (2*torch.pi)/k0
    # FXoff, FYoff = -torch.sin(xDeg)/wavelen, -torch.sin(yDeg)/wavelen
    # wavelen = (2*torch.pi)/k0
    FXoff, FYoff = -k*torch.sin(xRad)/(2*torch.pi), -k*torch.sin(yRad)/(2*torch.pi)
    FX_tilted, FY_tilted = FX-FXoff, FY-FYoff
    # root_arg = torch.sqrt(torch.clamp (1 - ((FX_tilted*(wavelen))**2 + (FY_tilted*(wavelen))**2), min=0) )
    root_arg = torch.sqrt(torch.clamp (k**2 - ((FX_tilted*(2*torch.pi))**2 + (FY_tilted*((2*torch.pi)))**2), min=0) )

    Amp =  ( root_arg > 0 )
    prop_op = Amp*torch.exp(1j*dz*root_arg)
    return prop_op

def propASKernelEvanescent(FX, FY, xRad, yRad, k, dz):
    FX, FY, xRad, yRad, k, dz = _ensure_single_device(FX, FY, xRad, yRad, k, dz)

    #  propagation_kernel(FX, FY, xDeg, yDeg, wavelen, dz), xDeg, yDeg = 0 means normal kernel    
    # H(f_x, f_y) = exp{ i*(2π/wavelen)*dz * sqrt(1 - [wavelen * f_x]^2 - [wavelen * f_y]^2) }
    # f_x_tilted = fx + FXoff = fx - (sin(X)/lambda)    
    wavelen = (2*torch.pi)/k
    FXoff, FYoff = -torch.sin(xRad)/wavelen, -torch.sin(yRad)/wavelen
    FX_tilted, FY_tilted = FX-FXoff, FY-FYoff
    root_arg = torch.sqrt(torch.clamp (k**2 - ((FX_tilted*(2*torch.pi))**2 + (FY_tilted*((2*torch.pi)))**2), min=0) )
    # root_arg = torch.sqrt(torch.clamp (1 - ((FX_tilted*(wavelen))**2 + (FY_tilted*(wavelen))**2), min=0) )
    # Amp =  ( root_arg > 0 )
    # prop_op = torch.exp(1j*k0*dz*root_arg)
    prop_op = torch.exp(1j*dz*root_arg)

    return prop_op

def FT(inputfield):
    input_fft = torch.fft.fftshift(torch.fft.fft2(inputfield))
    return input_fft

def IFT(inputfield):
    input_ifft = torch.fft.ifft2(torch.fft.ifftshift(inputfield))
    return input_ifft


## Input E Field related ##########################################
def sourceField(XX, YY, xalpha, ybeta, WD, k0, sourceType='Sphere'):
    # def sourceField(XX, YY, xalpha, ybeta, WD, wavelen, sourceType = 'Plane' or 'Sphere'):
    
    if sourceType=='Plane':       
        xa, yb = to_tensor(xalpha), to_tensor(ybeta)    
        FX, FY = k0 * torch.sin(xa), k0 * torch.sin(yb)
        phase = k0 * (torch.sin(xa) * XX + torch.sin(yb) * YY)
        Amp = torch.ones_like(XX)
        
    elif sourceType=='Sphere':
        xa, yb = to_tensor(-1*xalpha), to_tensor(-1*ybeta) 
        xs, ys = WD * torch.tan(xa), WD * torch.tan(yb)
        zs = WD
        pS = torch.tensor([xs,ys,zs])    
        # if isSphericSource==True: 
        # pS = (pS)/torch.norm(pS)*WD
        R_ = torch.sqrt((XX-pS[0])**2 + (YY-pS[1])**2 + pS[2]**2)
        phase = k0 * R_
        Amp =  (1 / R_)        
    else:
        error('source type is Plane or Sphere')
        
    outputfield = Amp*torch.exp(1j * phase)    
    return outputfield
    
    
#### Optical elements #############################################
def spherical_lens_by_radius(XX, YY, Aperture, nMat, R_req):
       # Focal length and radius for given fLens and WD
#     f_req = 1 / ((1 / fLens) + (1 / WD))
#     R_req = f_req * (n - 1)
#     t = R_req
    # h = lambda r, R, t: t - (R - torch.sqrt(R**2 - r**2))

    fLens = R_req / (nMat - 1)
    hLens = R_req - torch.sqrt(torch.clamp((R_req**2 - (Aperture*0.5)**2), min=0.0 )) # max Height of lens Profile
    r = torch.sqrt(XX**2 + YY**2) # Grid
    rCircle = r < (Aperture*0.5) # Aperture
    heightmap = (hLens - (R_req - torch.sqrt(torch.clamp((R_req**2 - r**2), min=0.0 ))))*rCircle
    # heightmap[torch.isnan(heightmap)] = torch.tensor(heightmap)

    # imagesc(rCircle, title='rCircle')
    # imagesc(heightmap, title='heightmap')
    # imagesc(heightmap*rCircle, title='heightmap*rCircle')
    # outputfield = Amp*torch.exp(1j * k0 * heightmap)    
    return heightmap, fLens
    
def spherical_lens(XX,YY, Aperture, nMat, fLens):
       # Focal length and radius for given fLens and WD
#     f_req = 1 / ((1 / fLens) + (1 / WD))
#     R_req = f_req * (n - 1)
#     t = R_req
    # h = lambda r, R, t: t - (R - torch.sqrt(R**2 - r**2))

    R_req = fLens * (nMat - 1)    
    hLens = R_req - torch.sqrt(torch.clamp((R_req**2 - (Aperture*0.5)**2), min=0.0 )) # max Height of lens Profile
    print(f'hLens : {hLens}')
    r = torch.sqrt(XX**2 + YY**2) # Grid
    rCircle = r < (Aperture*0.5) # Aperture
    heightmap = (hLens - (R_req - torch.sqrt(torch.clamp((R_req**2 - r**2), min=0.0))))*rCircle

    # imagesc(rCircle, title='rCircle')
    # imagesc(heightmap, title='heightmap')
    # imagesc(heightmap*rCircle, title='heightmap*rCircle')
    # outputfield = Amp*torch.exp(1j * k0 * heightmap)    
    print(f'designed R_req : {R_req}')
    return heightmap, R_req
    
    
def spatial_grating(XX, YY, T, angle):
    f = 1.0 / T

    rotated_coords = XX * torch.cos(angle) + YY * torch.sin(angle)
    # 위상 계산: 2*pi*f*x'
    phase = 2 * torch.pi * f * rotated_coords
    # sinusoidal grating 생성
    return torch.sin(phase)

# def ideal_lens_heightmap(XX, YY, Aperture, fLens, nMat, wavelength):
#     """
#     이상적인 구면파 위상을 이용하여 렌즈의 높이 맵을 계산합니다.
    
#     입력:
#       - XX, YY: 2D meshgrid (좌표 배열)
#       - Aperture: 렌즈의 전체 지름 (aperture)
#       - fLens: 렌즈의 초점 거리
#       - nMat: 렌즈 재질의 굴절률
#       - wavelength: 사용 파장
      
#     계산:
#       1. 파수 k0 = 2π/wavelength
#       2. 각 점에 대해 r = sqrt(XX^2 + YY^2)를 구함.
#       3. 이상적인 구면파 위상: φ(x,y) = k0*(sqrt(r^2 + fLens^2) - fLens)
#       4. 렌즈의 위상 변화는 Δφ = k0*(nMat-1)*h(x,y) 이므로,
#          h(x,y) = Δφ / [k0*(nMat-1)]
#       5. Aperture 내의 영역만 사용 (mask 적용)
      
#     출력:
#       - heightmap: 렌즈 표면의 높이 맵
#       - mask: Aperture mask
#     """
#     # 파수 계산
#     k0 = 2 * torch.pi / wavelength
    
#     # 좌표 r 계산
#     r2 = XX**2 + YY**2
#     r = torch.sqrt(r2)
    
#     # Aperture 마스크: r < Aperture/2인 영역만 남김
#     mask = (r < (Aperture / 2)).float()
    
#     # 이상적인 구면파 위상 지연 계산
#     phase_delay = k0 * (torch.sqrt(r2 + fLens**2) - fLens)
    
#     # 높이 맵 계산: h(x,y) = phase_delay / [k0*(nMat-1)]
#     heightmap = phase_delay / (k0 * (nMat - 1))
    
#     # Aperture 외부는 0으로 처리
#     heightmap = heightmap * mask
#     return heightmap, mask


def cylinderic_lens(XX, YY, Aperture, nMat, fLensX, fLensY):
    
    R_reqX = fLensX * (nMat - 1)    
    R_reqY = fLensY * (nMat - 1)    
    R_req_min = torch.min(R_reqX,R_reqY)
    hLens = R_req_min - torch.sqrt(torch.clamp((R_req_min**2 - (Aperture*0.5)**2), min=0.0 )) # max Height of lens Profile
    print(f'hLens : {hLens}')
    r = torch.sqrt(XX**2 + YY**2) # Grid
    rCircle = r < (Aperture*0.5) # Aperture
    sag_x = R_reqX - torch.sqrt(torch.clamp(R_reqX**2 - XX**2, min=0.0))
    sag_y = R_reqY - torch.sqrt(torch.clamp(R_reqY**2 - YY**2, min=0.0))

    heightmap = (sag_x+sag_y )* rCircle
    return heightmap

def cutoff_filter_kernel(FX, FY, cutoffVal):
    cutoff_kernel = ((FX**2 + FY**2) <= cutoffVal**2)
    return cutoff_kernel

#### for convenience #############################################
def subpixel_shift(im, dx, dy):
    """
    Parameters
    ----------
    im : torch.Tensor
        입력 2D 이미지 (실수 혹은 복소수). 크기는 (M, N)이어야 함.
    dx : float
        x 방향(열)으로의 shift (pixel 단위, 소수점 가능).
    dy : float
        y 방향(행)으로의 shift (pixel 단위, 소수점 가능).        
    Returns
    -------
    shifted_image : torch.Tensor
        주파수 도메인에서 위상 보정을 거친 후 역 FFT를 취한 이미지.
        (복소수 텐서; 필요시 torch.real()을 이용하여 실수부만 취할 수 있음)
    """
    device = im.device
   
    sy, sx = im.shape    
    yy = torch.arange(- (sy // 2), sy - (sy // 2), device=device)
    yy = torch.fft.ifftshift(yy)
    xx = torch.arange(- (sx // 2), sx - (sx // 2), device=device)
    xx = torch.fft.ifftshift(xx)
    XX, YY = torch.meshgrid(xx, yy, indexing='xy')  
    phase_ramp = torch.exp(-1j * 2 * torch.pi * ((XX * dx) / sx + (YY * dy) / sy))    
    shifted_image = torch.fft.ifft2(torch.fft.fft2(im) * phase_ramp)    
    return shifted_image
  
def imageCropPad(M_img, target_size=None, this_size=None):
    """
    Crops or zero-pads the input image `M_img` to match the target size, accounting for odd dimensions.
    Args:
        M_img (4D tensor): Input image to crop or pad (Batch, Channel, Height, Width).
        target_size (tuple): The target size (height, width) for the output tensor.
        this_size (tuple, optional): The current size of the image (height, width). 
                                     If None, it will automatically use the size of M_img.
    Returns:
        result_img (4D tensor): The cropped or padded image with the target size.
    """
    if this_size is None:
        this_size = M_img.shape[-2:]  # Automatically set this_size from input image dimensions (Height, Width)

    if target_size is None:
        raise ValueError("target_size must be provided.")

    # Check if we need to crop or pad
    if this_size[0] >= target_size[0] and this_size[1] >= target_size[1]:
        # Center crop the image
        top = (this_size[0] - target_size[0]) // 2
        bottom = top + target_size[0]
        left = (this_size[1] - target_size[1]) // 2
        right = left + target_size[1]
        result_img = M_img[..., top:bottom, left:right]
    
    else:
        # Zero pad the image
        v_pad_up = max((target_size[0] - this_size[0]) // 2, 0)
        v_pad_down = (target_size[0] - this_size[0]) - v_pad_up
        h_pad_left = max((target_size[1] - this_size[1]) // 2, 0)
        h_pad_right = (target_size[1] - this_size[1]) - h_pad_left

        padding = [h_pad_left, h_pad_right, v_pad_up, v_pad_down]  # Left, right, top, bottom padding
        result_img = F.pad(M_img, padding, mode='constant', value=0)

    return result_img

def imageResize(mask, upScaleFactor):
    output = F.interpolate(mask.unsqueeze(0).unsqueeze(0), scale_factor=upScaleFactor, mode='bicubic', align_corners=False).squeeze(0).squeeze(0)
    return output

def imagesc(image, title=None, cmap=None, figsize=None, vmin=None, vmax=None):
    """ 
    Displays an image (grayscale or color) in JupyterLab with a colorbar.
    
    Args:
        image (numpy array): The image to display. Can be grayscale or color (RGB).
        title (str, optional): The title of the image. Defaults to an empty string if not provided.
        cmap (str, optional): The colormap to use for grayscale images. Defaults to 'viridis' if not specified.
        vmin (float, optional): The minimum data value for colormap normalization. If None, the data's min is used.
        vmax (float, optional): The maximum data value for colormap normalization. If None, the data's max is used.
    """    
    # Set default title and cmap if not provided
    if title is None:
        title = ""  # Default to no title
    if cmap is None:
        cmap = 'viridis'  # Default colormap
    if figsize is None:
        figsize=6
        
    # If image is a torch.Tensor, move to CPU and convert to numpy
    if isinstance(image, torch.Tensor):
        if image.is_cuda:
            image = image.cpu()  # Move to CPU if on CUDA
        image = image.detach().numpy()  # Convert to numpy
    sy, sx = image.shape
    # Initialize the figure
    plt.figure(figsize=(figsize, figsize))    
    # Check if image is grayscale or RGB
    if len(image.shape) == 2:  # Grayscale image
        img_plot = plt.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax)
        plt.colorbar(img_plot,fraction = 0.05*(sy/sx), pad=0.04)
    elif len(image.shape) == 3 and image.shape[2] == 3:  # RGB image
        img_plot = plt.imshow(image)
        # No colorbar for RGB images as the pixel values are RGB triplets
    else:
        raise ValueError("Unsupported image format! Provide a grayscale (2D) or RGB (3D) image.")
    
    plt.title(title)  # Use the provided or default title
    # plt.axis('off')  # Optionally turn off the axis
    plt.show()
    
    
### Minimal Sampling rate estimation
def estimateSampRate(dimX,dimY, xs, ys, zs, wavelen, Demodulated=False):    
    # estimateSampRate(dimX,dimY, xs, ys, zs, wavelen, Demodulated==False):
    # sqrt((XX-dx)**2+(YY-dy)**2+zz**2)의 gradient를 analytic하게 구해서 필요한 sampling rate 결정    
    # sr = 2.7e-6 # sampling rate 최대 필요값 도출시 이용할 sampling rate, 너무 크지만 않으면 됨
    
    # nx, ny = int(dimX/sr), int(dimY/sr)
    nx, ny = 512, int(512*dimY/dimX)
    Xx,Yy = image_grid(dimX, dimY, nx, ny)

    if Demodulated==True:
        r = torch.sqrt((xs)**2 + (ys)**2  + zs**2)
        xc, yc = xs/zs , ys/zs 
        #carrier = torch.exp(1j * k * (X * sin(thetax) + Y * sin(thetay)))?
        #carrier = torch.exp(1j * k * (X * tan(thetax) + Y * tan(thetay)))?
    else:
        xc, yc = 0 , 0
    
    g = torch.sqrt((Xx-xs)**2 + (Yy-ys)**2  + zs**2)

    gradx = (Xx-xs)/g  + xc 
    grady = (Yy-ys)/g  + yc
    # grad = torch.sqrt(grady**2+gradx**2)
    # Absgrad = torch.abs(grady)+torch.abs(gradx)
    Absgrad = torch.max(torch.abs(grady), torch.abs(gradx))

    max_phase_gradient = torch.pi*2*((torch.max(Absgrad))/wavelen)

    #인접 픽셀간 위상차이가  Nyq 기준으로 pi이하여야 함
    # Nyquist 조건: Δs <= π / (max_grad)
    ds = torch.pi / max_phase_gradient
    # print (f"{ds/um:.2f}um")
    return ds

def estimateSampRate_from_angle(dimX, dimY, xalpha, yalpha, WD, wavelen, Demodulated=False):
    """
    estimateSampRate_from_angle(dimX, dimY, xalpha, yalpha, WD, wavelen):
    반환:
      최소 샘플링 간격 (미터 단위)
    """
    # 점광원 위치 계산 (기존 sphericalwave 함수의 방식)
    xs = WD * torch.tan(-xalpha)
    ys = WD * torch.tan(-yalpha)
    zs = WD  # 마스크 평면이 z=0이라 가정    
    return estimateSampRate(dimX, dimY, xs, ys, zs, wavelen, Demodulated)

def estimateSampRate_from_heightmap(heightmap: torch.Tensor, dx: float, wavelen: float) -> float:
    """
    heightmap (Tensor): 2D 높이 분포 (예: shape = [ny, nx])
    dx, dy (float): 각 픽셀의 실제 물리적 크기(단위 m 등)
    wavelen (float): 파장 (예: m 단위)
    
    반환값:
      ds (float): Nyquist 조건을 만족하는 최소 샘플링 간격 (m 단위)
    """
    # heightmap.shape = [ny, nx]
    # PyTorch의 gradient 함수 사용 (버전에 따라 spacing 인자가 없을 수 있음)
    # grad_y, grad_x = torch.gradient(heightmap, spacing=(dy, dx))
    
    # spacing 매개변수를 지원하지 않는 버전의 PyTorch라면, 다음과 같이 직접 계산:
    grad_y = (heightmap[1:, :] - heightmap[:-1, :]) / dx  # y방향 차분
    grad_x = (heightmap[:, 1:] - heightmap[:, :-1]) / dx  # x방향 차분

    # 유효 범위(1픽셀 줄어든)에서 절댓값의 최대값을 찾기 위해
    # 편의상 같은 크기로 맞추고, 2D상의 모든 위치에서의 기울기 크기를 계산
    # grad_y: shape=(ny-1, nx), grad_x: shape=(ny, nx-1)
    # 크기가 다르므로 아래처럼 중간 영역을 맞춰서 계산할 수도 있음
    ny, nx = heightmap.shape
    min_ny = min(grad_y.shape[0], grad_x.shape[0])
    min_nx = min(grad_y.shape[1], grad_x.shape[1])

    # 공통 영역에서 기울기 크기를 계산
    # 여기서는 간단히 |grad_x| + |grad_y| 형태(혹은 sqrt(grad_x^2 + grad_y^2)) 사용 가능
    # 필요에 따라 정의 변경 가능
    abs_grad = torch.max(torch.abs(grad_y[:min_ny, :min_nx]), torch.abs(grad_x[:min_ny, :min_nx]))

    # 높이 h(x,y)에 대응하는 위상 기울기는 (2π/λ)*grad(h)
    # 따라서 위상 기울기의 최대값을 찾는다
    max_phase_gradient = (2.0 * torch.pi / wavelen) * torch.max(abs_grad)

    # Nyquist 조건: Δs * (max_phase_gradient) <= π  ->  Δs <= π / max_phase_gradient
    ds = torch.pi / max_phase_gradient

    return ds.item()


def heightmapSkew(heightmap, px_size, thXrad, thYrad, WD, sourceType = 'Plane', heightmapDirection = 'Front'):
    """
    PyTorch version of heightmapSkew.
    
    Parameters:
        heightmap (torch.Tensor): 2D array representing the original heightmap (H, W)
        px_size (float): Pixel size
        thXrad (float): Tilt angle around x-axis in radians
        thYrad (float): Tilt angle around y-axis in radians
    
    Returns:
        torch.Tensor: Tilted heightmap
    """
      # Get the size of the heightmap
    Ny, Nx = heightmap.shape
    DimX, DimY = Nx*px_size, Ny*px_size
    mH = torch.max(heightmap)
    xx = torch.linspace(-(Nx / 2), (Nx / 2), Nx, device=heightmap.device) * px_size
    yy = torch.linspace(-(Ny / 2), (Ny / 2), Ny, device=heightmap.device) * px_size
    XX, YY = torch.meshgrid(xx, yy, indexing='xy')

    if heightmapDirection == 'Front':
         # heightmap = torch.max(heightmap)-heightmap
        thXrad = -thXrad
        thYrad = -thYrad
        offset = mH
    else:
        offset = 0

    
    # 점광원 위치 계산 (기존 sphericalwave 함수의 방식, WD를 멀리 놓아서 Plane wave 근사)
    xs = WD * torch.tan(thXrad)
    ys = WD * torch.tan(thYrad)
    zs = WD  # 마스크 평면이 z=0이라 가정
    
    if sourceType=='Plane':
        dx = (heightmap-offset) * (xs/zs)
        dy = (heightmap-offset) * (ys/zs)  
        # imagesc(dx,title='dx Plane')
    elif sourceType=='Sphere': #spherical wave
        # 점광원 위치 계산 (기존 sphericalwave 함수의 방식)    
        dx = (heightmap-offset) * ((xs-XX)/zs)
        dy = (heightmap-offset) * ((ys-YY)/zs)
        # imagesc(dx,title='dx Sphere')
    else:
        print('source type plane으로 가정')
        dx = (heightmap-offset) * (xs/zs)
        dy = (heightmap-offset) * (ys/zs)  
    # medianH = 0.0 * (torch.max(heightmap) - torch.min(heightmap))
#     # Calculate displacement in x and y directions
#     # thXrad를 pixelwise로  
# #     dx = (heightmap - medianH) * (xs/zs)
# #     dy = (heightmap - medianH) * (ys/zs)

    
    # Compute new coordinates
    x_new = XX + dx
    y_new = YY + dy

    # Normalize coordinates to [-1, 1] for grid_sample
    x_norm = 2.0 * (x_new/DimX)
    y_norm = 2.0 * (y_new/DimY)

    # Stack and reshape for grid_sample
    grid = torch.stack((x_norm, y_norm), dim=-1).unsqueeze(0)

    # Interpolate using grid_sample (equivalent to interp2)
    heightmap_tilted = F.grid_sample(heightmap.unsqueeze(0).unsqueeze(0).float(), grid.float(), mode='bicubic', align_corners=True)
    heightmap_tilted = heightmap_tilted.squeeze()

    # Fill NaN values with the minimum height
    heightmap_tilted[torch.isnan(heightmap_tilted)] = torch.min(heightmap).float()
    
    return heightmap_tilted.double()



def heightmapSliceNew(wholeMask, numSlices=10, direction = 'Front'):
    """
    주어진 2D heightmap(단위: m)을 dZinterval (m) 간격으로 슬라이싱하여,
    각 슬라이스는 해당 구간의 높이값을 반환합니다.
    
    예)
       1D heightmap = [1, 2, 3.1, 2, 1] (m 단위)
       dZinterval = 1.1
       결과:
           slice 0 = clamp([1,2,3.1,2,1] - 0, 0, 1.1) = [1.0, 1.1, 1.1, 1.1, 1.0]
           slice 1 = clamp([1,2,3.1,2,1] - 1.1, 0, 1.1) = [0, 0.9, 1.1, 0.9, 0]
           slice 2 = clamp([1,2,3.1,2,1] - 2.2, 0, 1.1) = [0, 0, 0.9, 0, 0]
           
    Parameters:
        wholeMask (torch.Tensor): 입력 heightmap, shape=(nY, nX), 단위 m
        dZinterval (float or 0-dim torch.Tensor): 슬라이스 간격 (m)
    
    Returns:
        slices (torch.Tensor): shape=(nY, nX, nSlices)로 stacking된 각 슬라이스
        nSlices (int): 생성된 슬라이스 수
    """
    # 입력 heightmap을 음수가 없도록 최소값을 빼줌
    heightMask = wholeMask - wholeMask.min()
    nSlices = numSlices
    
    if direction == 'None':
        direction = 'Front'


    # 최대 높이 T와 dZinterval을 이용해 슬라이스 수 계산 (나머지가 있으면 한 장 추가)
    max_val = heightMask.max()
    dZinterval = max_val / nSlices
    
    nY, nX = heightMask.shape
    slices = torch.zeros((nY, nX, nSlices), dtype=heightMask.dtype, device=heightMask.device)
    for i in range(nSlices):
        # 각 슬라이스는 (heightMask - i*dZinterval)을 [0, dZinterval]으로 클램핑
        slice_i = torch.clamp(heightMask - i * dZinterval, min=0, max=dZinterval)
        if direction == 'Front':
            slices[:, :, nSlices-1-i] = slice_i
        else:
            slices[:, :, i] = slice_i

        
    return slices, nSlices, dZinterval

def genAngularResponseKernel(FX, FY, wavelen, cutoffTheta):

    # Gaussian에서 cutoff 조건: exp(-cutoffTheta^2/(2*sigma^2)) = 0.5
    sigma = cutoffTheta / torch.sqrt(2 * torch.log(torch.tensor(2.0)))

    # 2D 그리드 (일반적인 FFT grid; 순서: (Nx, Ny))
    # FX, FY = fourier_grid(DimX, DimY, Nx, Ny)
    # wave number: k = 2π/λ

    # 각 성분의 입사각 계산: 
    # x방향: theta_x = arcsin(FX/k), y방향: theta_y = arcsin(FY/k)
    # 클램핑으로 arcsin의 정의역 [-1,1]을 보장합니다.
    theta_x = torch.asin(torch.clamp(FX * wavelen, -1.0, 1.0))
    theta_y = torch.asin(torch.clamp(FY * wavelen, -1.0, 1.0))
    
    # 2D에서 전체 입사각은 두 성분의 합성 각도로 정의 (radially symmetric)
    theta_total = torch.sqrt(theta_x**2 + theta_y**2)
    
    # Gaussian 필터 생성 (theta 도메인에서 정의)
    kernel = torch.exp(-theta_total**2 / (2 * sigma**2))
    
    # 최대값 1 (수직 입사: theta=0)로 정규화
    kernel /= kernel.max()
    
    if cutoffTheta==0:
        kernel = torch.ones_like(kernel)
    
    return kernel

def fresnelNumber(Diameter, wavelen, dz):
    return Diameter**2/(4*wavelen*dz)

def compute_tamura_cog(image):
    """
    Compute Tamura's Coefficient of Gradients (CoG) for a given image.
    """
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=image.device).unsqueeze(0).unsqueeze(0)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32, device=image.device).unsqueeze(0).unsqueeze(0)
    
    image = image.unsqueeze(0).unsqueeze(0)  # Convert to (1,1,H,W) format for convolution
    grad_x = F.conv2d(image, sobel_x, padding=1).squeeze()
    grad_y = F.conv2d(image, sobel_y, padding=1).squeeze()
    
    gradient_magnitude = torch.sqrt(grad_x ** 2 + grad_y ** 2)
    cog = torch.sum(gradient_magnitude) / torch.sum(image + 1e-6)  # Avoid division by zero

    return cog.item()

def mmNormalize(data):
    if isinstance(data, torch.Tensor):
        data = (data - torch.min(data)) / (torch.max(data) - torch.min(data)+1e-6)
    else:
        data = (data - np.min(data)) / (np.max(data) - np.min(data)+1e-6)

    # noise term prevents the zero division
    return data


def stdNormalize(data):
    if isinstance(data, torch.Tensor):
        data = (data-torch.mean(data))/torch.std(data) 
    else:
        data = (data-np.mean(data))/np.std(data) 
    # noise term prevents the zero division
    return data


def save8bitImg(fileName, toSave, scaling=True):
    if scaling==True:
        toSave = mmNormalize(toSave)
        toSave = np.array(toSave*255, dtype='uint8')
    else:
        toSave =  np.array(toSave, dtype='uint8')

    return cv2.imwrite(f'{fileName}',toSave)
    
