# filters.py
import cv2
import numpy as np

def apply_monochrome(image):
    """Apply monochrome filter"""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # Enhance contrast
    gray = cv2.convertScaleAbs(gray, alpha=1.2, beta=10)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

def apply_vivid(image):
    """Apply vivid filter (increase saturation and sharpness)"""
    # Convert to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
    
    # Increase saturation
    hsv[:, :, 1] = hsv[:, :, 1] * 1.5
    hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
    
    # Convert back to RGB
    vivid = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
    
    # Apply sharpening
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    sharpened = cv2.filter2D(vivid, -1, kernel)
    
    return sharpened

def apply_impressionist(image):
    """Apply impressionist filter"""
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(image, (15, 15), 0)
    
    # Enhance edges
    edges = cv2.Canny(image, 100, 200)
    edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    
    # Combine blurred image with edges
    result = cv2.addWeighted(blurred, 0.7, edges, 0.3, 0)
    
    return result

def apply_warm(image):
    """Apply warm tone filter"""
    # Create warming filter
    warming_filter = np.array([[1.2, 0, 0],
                              [0, 1.1, 0],
                              [0, 0, 0.9]], dtype=np.float32)
    
    # Apply filter
    warm_image = cv2.transform(image, warming_filter)
    warm_image = np.clip(warm_image, 0, 255)
    
    # Add slight vignette
    rows, cols = image.shape[:2]
    kernel_x = cv2.getGaussianKernel(cols, cols/3)
    kernel_y = cv2.getGaussianKernel(rows, rows/3)
    kernel = kernel_y * kernel_x.T
    mask = 255 * kernel / np.linalg.norm(kernel)
    mask = cv2.merge([mask, mask, mask])
    
    warm_image = warm_image * (mask / 255.0)
    warm_image = np.clip(warm_image, 0, 255).astype(np.uint8)
    
    return warm_image

def apply_cool(image):
    """Apply cool tone filter"""
    # Create cooling filter
    cooling_filter = np.array([[0.9, 0, 0],
                              [0, 1.0, 0],
                              [0, 0, 1.2]], dtype=np.float32)
    
    # Apply filter
    cool_image = cv2.transform(image, cooling_filter)
    cool_image = np.clip(cool_image, 0, 255)
    
    return cool_image

def apply_nostalgic(image):
    """Apply nostalgic/vintage filter"""
    # Apply sepia tone
    kernel = np.array([[0.393, 0.769, 0.189],
                       [0.349, 0.686, 0.168],
                       [0.272, 0.534, 0.131]])
    nostalgic = cv2.transform(image, kernel)
    nostalgic = np.clip(nostalgic, 0, 255)
    
    # Add vignette effect
    rows, cols = image.shape[:2]
    kernel_x = cv2.getGaussianKernel(cols, cols/2)
    kernel_y = cv2.getGaussianKernel(rows, rows/2)
    kernel = kernel_y * kernel_x.T
    mask = kernel / np.max(kernel)
    mask = cv2.merge([mask, mask, mask])
    
    nostalgic = nostalgic * mask
    nostalgic = np.clip(nostalgic, 0, 255).astype(np.uint8)
    
    return nostalgic

def apply_dramatic(image):
    """Apply dramatic filter (high contrast, darkened shadows)"""
    # Convert to LAB color space
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    
    # Split channels
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE to L channel
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    l = clahe.apply(l)
    
    # Merge channels
    lab = cv2.merge((l, a, b))
    
    # Convert back to RGB
    dramatic = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    
    # Increase contrast
    dramatic = cv2.convertScaleAbs(dramatic, alpha=1.2, beta=0)
    
    return dramatic
def apply_filter(image, filter_name):
    """Apply a filter based on filter_name"""
    if filter_name == "monochrome":
        return apply_monochrome(image)
    elif filter_name == "vivid":
        return apply_vivid(image)
    elif filter_name == "impressionist":
        return apply_impressionist(image)
    elif filter_name == "warm":
        return apply_warm(image)
    elif filter_name == "cool":
        return apply_cool(image)
    elif filter_name == "nostalgic":
        return apply_nostalgic(image)
    elif filter_name == "dramatic":
        return apply_dramatic(image)
    else:
        return image  # If filter not found, return original image
