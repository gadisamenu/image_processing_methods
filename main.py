import matplotlib.pyplot as plt
import matplotlib.widgets as widgets
import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog

def convert_to_grayscale(image):
    """Converts RGB image to grayscale."""
    if len(image.shape) == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        return image


def negative_image(image):
  """Inverts the intensity values of an image (creates a negative).

  Args:
      image: A NumPy array representing the image (RGB or grayscale).

  Returns:
      A NumPy array representing the negative image.
  """
  return 255 - image

def gamma_correction(image, gamma=1.0):
    # normalize image to range 0-1
    image_normalized = image / 255.0

    # apply gamma correction formula
    image_corrected = 255.0 * np.power(image_normalized, 1.0/gamma)

    # clip values that exceed the valid range due to numerical errors
    image_corrected = np.clip(image_corrected, 0, 255)

    # convert image back to uint8 data type
    image_corrected = image_corrected.astype(np.uint8)

    return image_corrected



def logarithmic_transformation(image, c=1):
    # apply logarithmic transformation formula
    image_transformed = c * np.log(np.where(image == 0, 1e-10, image) + 1) / np.log(256)

    # clip values that exceed the valid range due to numerical errors
    image_transformed = np.clip(image_transformed, 0, 255)

    # convert image back to uint8 data type
    image_transformed = image_transformed.astype(np.uint8)

    return image_transformed


def contrast_stretching(image, low=0, high=255):
  old_min = np.min(image)
  old_max = np.max(image)
  
  # Handle case where min and max are the same (avoid division by zero)
  if old_min == old_max:
    return image.copy()
  
  # Clip low intensity values to low
  clipped_image = np.clip(image, low, high)
  
  return ((clipped_image - old_min) * (high - low) / (old_max - old_min)) + low




def histogram_equalization(image):
    # image = convert_to_grayscale(image)
    if len(image.shape) == 2:  # grayscale image
        equalized_image = cv2.equalizeHist(image)
    elif len(image.shape) == 3 and image.shape[2] == 3:  # color image
        # split the image into its three color channels
        b, g, r = cv2.split(image)

        # equalize the histogram of each channel separately
        b_eq = cv2.equalizeHist(b)
        g_eq = cv2.equalizeHist(g)
        r_eq = cv2.equalizeHist(r)

        # merge the three channels back into a single image
        equalized_image = cv2.merge((b_eq, g_eq, r_eq))
    else:
        return image
    return equalized_image



def intensity_level_slicing(image, low_threshold, high_threshold):
    
    image = convert_to_grayscale(image)
    row, column = image.shape
    # Create an zeros array to store the sliced image
    img1 = np.zeros((row,column),dtype = 'uint8')


    # Loop over the input image and if pixel value lies in desired range set it to 255 otherwise set it to 0.
    for i in range(row):
        for j in range(column):
            if image[i,j]>low_threshold and image[i,j]< high_threshold:
                img1[i,j] = 255
            else:
                img1[i,j] = 0
                
    return img1
        


def bit_plane_slicing(image, plane_number):
    if len(image.shape) == 3:
        image  = convert_to_grayscale(image)
    
    return (image >> plane_number) & 1 



def load_image(file_path):
    image = cv2.imread(file_path, cv2.IMREAD_COLOR)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)



class Main:
    def __init__(self):
        self.image  = None
        self.log = 1
        self.bit_pln= 0
        self.int_level_sl_l = 0
        self.int_level_sl_h = 255
        self.gm = 0.1
        self.cnt_str_l = 0
        self.cnt_str_h = 255
        
    
    def run(self):
            
        def update(val, ax,type):
            # Update image intensity based on slider value
            if type == 'gamma':
                self.gm = val
                adjusted_img = gamma_correction(self.image, val)
                
            elif type == 'log':
                self.log = val
                adjusted_img = logarithmic_transformation(self.image, val)
                
            elif type == 'cont_stretch_l' or type == 'cont_stretch_h':
                if (type == 'cont_stretch_l'):
                    self.cnt_str_l = int(val)
                else:
                    self.cnt_str_h = int(val)
                adjusted_img = contrast_stretching(self.image, self.cnt_str_l, self.cnt_str_h)
                
            elif type == 'bit_plane':
                self.bit_pln = int(val)
                adjusted_img = bit_plane_slicing(self.image, self.bit_pln)
                
            elif type == 'intensity_l' or type == 'intensity_h':
                if (type == 'intensity_l'):
                    self.int_level_sl_l =  int(val)
                else:
                    self.int_level_sl_h = int(val)                   
                adjusted_img = intensity_level_slicing(self.image, self.int_level_sl_l, self.int_level_sl_h )
        
            ax.images[0].set_array(adjusted_img)
            fig.canvas.draw_idle()
            
        def updateAll():
            original_image_axes.imshow(self.image)
            negative_image_axes.imshow(negative_image(self.image))
            gm_crr_axes.imshow(gamma_correction(self.image,self.gm))
            log_tr_axes.imshow(logarithmic_transformation(self.image,self.log))
            cnt_str_axes.imshow(contrast_stretching(self.image,self.cnt_str_l,self.cnt_str_h))
            histogram_equalization_axes.imshow(histogram_equalization(self.image))
            int_lc_axes.imshow(intensity_level_slicing(self.image,self.int_level_sl_l,self.int_level_sl_h))
            bp_slc_axes.imshow(bit_plane_slicing(self.image,self.bit_pln))
            
            fig.canvas.draw_idle()
            

        fig = plt.figure(figsize=(10, 7))  # set the size of the figure to 10x5 inches
        axes = fig.subplots(2, 4)  # create a 2x4 grid of subplots
        
        def open_file_browser(event):
            root = tk.Tk()
            root.withdraw()
            file_path = filedialog.askopenfilename()
            if (file_path):
                self.image = load_image(file_path)
                updateAll()
            
        # create a button
        button_ax = fig.add_axes([0.45, 0.95, 0.1, 0.060])  # left, bottom, width, height
        button = plt.Button(button_ax, 'Open file')

        # set the callback function for the button
        button.on_clicked(open_file_browser)
        
        
        # Display the images in the subplots
        original_image_axes = axes[0, 0]
        if self.image: original_image_axes.imshow(self.image)
        original_image_axes.axis('off')
        original_image_axes.set_title('Original Image')

        negative_image_axes = axes[0, 1]
        if self.image: negative_image_axes.imshow(negative_image(self.image))
        negative_image_axes.axis('off')
        negative_image_axes.set_title('Negative Image')


        gm_crr_axes = axes[0, 2]
        if self.image: gm_crr_axes.imshow(gamma_correction(self.image,self.gm))
        gm_crr_axes.axis('off')
        gm_crr_axes.set_title('Gamma Correction')

        slider_ax = fig.add_axes([gm_crr_axes.get_position().x0, gm_crr_axes.get_position().y0 - 0.04, gm_crr_axes.get_position().width, 0.02])
        slider = widgets.Slider(slider_ax, 'gm', 0.1, 10.0, valinit=self.gm)
        slider.on_changed(lambda val, gm_crr_axes=gm_crr_axes,type='gamma': update(val, gm_crr_axes,type))


        log_tr_axes = axes[0, 3]
        if self.image: log_tr_axes.imshow(logarithmic_transformation(self.image,self.log))
        log_tr_axes.axis('off')
        log_tr_axes.set_title('Logarithmic Transformation')

        slider_ax_lg = fig.add_axes([log_tr_axes.get_position().x0 + 0.05, log_tr_axes.get_position().y0 - 0.04, log_tr_axes.get_position().width, 0.02])
        slider_lg = widgets.Slider(slider_ax_lg, 'c', 0, 255, valinit=self.log)
        slider_lg.on_changed(lambda val, log_tr_axes=log_tr_axes,type='log': update(val, log_tr_axes,type))



        cnt_str_axes = axes[1, 0]
        if self.image: cnt_str_axes.imshow(contrast_stretching(self.image,self.cnt_str_l,self.cnt_str_h))
        cnt_str_axes.axis('off')
        cnt_str_axes.set_title('Contrast Stretching')
        slider_ax_cl = fig.add_axes([cnt_str_axes.get_position().x0 - 0.08, cnt_str_axes.get_position().y0 - 0.08, cnt_str_axes.get_position().width, 0.02])
        slider_cl = widgets.Slider(slider_ax_cl, 'l', 0, 255, valinit=self.cnt_str_l,valstep=1)
        slider_cl.on_changed(lambda val, cnt_str_axes=cnt_str_axes,type='cont_stretch_l': update(val, cnt_str_axes,type))

        slider_ax_ch = fig.add_axes([cnt_str_axes.get_position().x0 - 0.08, cnt_str_axes.get_position().y0 - 0.11, cnt_str_axes.get_position().width, 0.02])
        slider_ch = widgets.Slider(slider_ax_ch, 'h', 0, 255, valinit=self.cnt_str_h,valstep=1)
        slider_ch.on_changed(lambda val, cnt_str_axes=cnt_str_axes,type='cont_stretch_h': update(val, cnt_str_axes,type))

        histogram_equalization_axes = axes[1, 1]
        if self.image: histogram_equalization_axes.imshow(histogram_equalization(self.image))
        histogram_equalization_axes.axis('off')
        histogram_equalization_axes.set_title('Histogram Equalization')


        int_lc_axes = axes[1, 2]
        if self.image: int_lc_axes.imshow(self.image)
        int_lc_axes.axis('off')
        int_lc_axes.set_title('Intensity Level Slicing')

        slider_ax_it_l = fig.add_axes([int_lc_axes.get_position().x0, int_lc_axes.get_position().y0 - 0.08, int_lc_axes.get_position().width, 0.02])
        slider_it_l = widgets.Slider(slider_ax_it_l, 'l', 0, 255, valinit=self.int_level_sl_l,valstep=1)
        slider_it_l.on_changed(lambda val, int_lc_axes=int_lc_axes,type='intensity_l': update(val, int_lc_axes,type))

        slider_ax_it_h = fig.add_axes([int_lc_axes.get_position().x0 - 0., int_lc_axes.get_position().y0 - 0.11, int_lc_axes.get_position().width, 0.02])
        slider_it_h = widgets.Slider(slider_ax_it_h, 'h', 0, 255, valinit=self.int_level_sl_h,valstep=1)
        slider_it_h.on_changed(lambda val, int_lc_axes=int_lc_axes,type='intensity_h': update(val, int_lc_axes,type))

        bp_slc_axes = axes[1, 3]
        if self.image: bp_slc_axes.imshow(self.image)
        bp_slc_axes.axis('off')
        bp_slc_axes.set_title('Bit Plane Slicing')
        slider_ax_bp = fig.add_axes([bp_slc_axes.get_position().x0 + 0.05, bp_slc_axes.get_position().y0 - 0.08, bp_slc_axes.get_position().width, 0.02])
        slider_bp = widgets.Slider(slider_ax_bp, 'n', 0, 8, valinit=self.bit_pln,valstep=1)
        slider_bp.on_changed(lambda val, bp_slc_axes=bp_slc_axes,type='bit_plane': update(val, bp_slc_axes,type))

            
        fig.subplots_adjust(wspace=0.1, hspace=0.1)
        fig.tight_layout(h_pad=1.0, w_pad=1.0)
        plt.show()




if __name__ == "__main__":
    main = Main()
    main.run()
