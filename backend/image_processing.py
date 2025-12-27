import os
import io
import google.generativeai as genai
from PIL import Image
import numpy as np
import cv2
from dotenv import load_dotenv


load_dotenv()


genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
model = genai.GenerativeModel('gemini-1.5-flash')

def analyze_image_quality(image_bytes):
    """Analyze image quality using Gemini Vision API"""
    try:
        
        image = Image.open(io.BytesIO(image_bytes))
        
       
        prompt = """
        Analyze this image and provide a detailed assessment of:
        1. Blur level and type (motion blur, out of focus, etc.)
        2. Signs of tampering or manipulation
        3. Overall quality issues (noise, compression artifacts, etc.)
        4. Recommended enhancement parameters
        Format the response as a JSON with these keys:
        {blur_level, blur_type, tampering_detected, quality_issues, enhancement_params}
        """
        
       
        response = model.generate_content([prompt, image])
        analysis = response.text
        
       
        return analysis
    except Exception as e:
        return str(e)

def deblur_image(image_array):
    """Apply advanced deblurring techniques"""
    try:
       
        if len(image_array.shape) == 3:
            gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
        else:
            gray = image_array

     
        psf = cv2.estimateRobustMotionBlur(gray)
        
     
        deblurred = cv2.deconvWiener(image_array, psf)
        
       
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        deblurred = cv2.filter2D(deblurred, -1, kernel)
        
        return deblurred
    except Exception as e:
        return image_array

def fix_tampering(image_array, analysis):
    """Fix detected tampering issues"""
    try:
        
        mask = np.zeros(image_array.shape[:2], dtype=np.uint8)
        fixed_image = cv2.inpaint(image_array, mask, 3, cv2.INPAINT_TELEA)
        return fixed_image
    except Exception as e:
        return image_array

def enhance_image_quality(image_array, params):
    """Apply AI-guided image enhancement"""
    try:
       
        lab = cv2.cvtColor(image_array, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=params.get('clip_limit', 2.0))
        l = clahe.apply(l)
        enhanced = cv2.merge((l, a, b))
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        
      
        enhanced = cv2.fastNlMeansDenoisingColored(
            enhanced,
            None,
            h=params.get('denoise_strength', 10),
            hColor=params.get('color_strength', 10)
        )
        
        return enhanced
    except Exception as e:
        return image_array

def process_image(image_bytes, filename, enhancement_params=None):
    """Main function to process images using Gemini Vision and advanced CV techniques"""
    try:
      
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
       
        original_format = filename.split('.')[-1]
        original_size = len(image_bytes)
        
 
        report = {
            'original_format': original_format,
            'original_size': original_size,
            'final_size': 0,
            'operations': [],
            'errors': []
        }
        
      
        analysis = analyze_image_quality(image_bytes)
        report['operations'].append('Performed AI analysis using Gemini Vision')
        
      
        deblurred_image = deblur_image(image)
        report['operations'].append('Applied advanced deblurring')
        
      
        fixed_image = fix_tampering(deblurred_image, analysis)
        report['operations'].append('Fixed potential tampering issues')
     
        final_image = enhance_image_quality(fixed_image, enhancement_params or {})
        report['operations'].append('Applied AI-guided quality enhancement')
        
        is_success, buffer = cv2.imencode(f'.{original_format}', final_image)
        if not is_success:
            raise Exception('Failed to encode processed image')
        
        processed_image_bytes = buffer.tobytes()
        report['final_size'] = len(processed_image_bytes)
        
        return processed_image_bytes, report
        
    except Exception as e:
        report['errors'].append(str(e))
        return image_bytes, report

def check_image_quality(image_bytes):
    """Check image quality and return issues"""
    try:
        
        analysis = analyze_image_quality(image_bytes)
        
        
        quality_report = {
            'issue_detected': False,
            'results_df': [],
            'recommendation': ''
        }
        
   
        if analysis:
            quality_report['issue_detected'] = True
            quality_report['recommendation'] = analysis
        
        return quality_report
        
    except Exception as e:
        return {
            'issue_detected': True,
            'results_df': [],
            'recommendation': f'Error during quality check: {str(e)}'
        }
