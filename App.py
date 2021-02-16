import numpy as np
import tkinter as tk
from PIL import Image, ImageTk
import cv2
import math
import time

class MainWindow():
    def __init__(self, window, cap): 
        
        # Vars
        self.window = window
        self.window.geometry("1170x620")
        self.cap = cap
        self.interval = 200 # Interval in ms to get the latest frame
        self.width = 250
        self.height = 250
        self.flag_noise = False
        self.flag_bc = False
        self.flag_loc = False
        self.flag_skin_hsv = False
        self.flag_skin_ycc = False
        self.flag_detect = False
          
        # First Frame zero
        zero_img = np.zeros([self.height, self.width, 3],dtype=np.uint8)
        zero_img.fill(255)
        zero_img = self.resize_image(zero_img)
        self.image_zero = zero_img
        self.image = self.img_to_photo(self.image_zero)
        self.image_save = zero_img
        self.image_proc = self.img_to_photo(self.image_zero)
        self.image_proc_focus = self.img_to_photo(self.image_zero)
        self.image_proc_focus_gesture = self.img_to_photo(self.image_zero)
        
        # Labels
        self.label = tk.Label(self.window, text="HAND GESTURES DETECTION", font=("Calibri", 16))
        self.label.grid(row=0, column=0, padx=10, pady=5, sticky="sw")
        self.label = tk.Label(self.window, text="Original Frames", font=("Calibri", 12), relief="groove", width=35)
        self.label.grid(row=1, column=0, padx=10, pady=5, sticky="sw")
        self.label = tk.Label(self.window, text="Processed Frames", font=("Calibri", 12), relief="groove", width=35)
        self.label.grid(row=1, column=2, padx=10, pady=5, sticky="sw")
        self.label = tk.Label(self.window, text="Localization Focus", font=("Calibri", 12), relief="groove", width=35)
        self.label.grid(row=3, column=0, padx=10, pady=5, sticky="sw")
        self.label = tk.Label(self.window, text="Gesture Detected", font=("Calibri", 12), relief="groove", width=35)
        self.label.grid(row=3, column=2, padx=10, pady=5, sticky="sw")
        
        # Videos canvas
        '''self.width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)'''
        self.canvas = tk.Canvas(self.window, width=self.width, height=self.height)
        self.canvas2 = tk.Canvas(self.window, width=self.width, height=self.height)
        self.canvas3 = tk.Canvas(self.window, width=self.width, height=self.height)
        self.canvas4 = tk.Canvas(self.window, width=self.width, height=self.height)
        self.canvas.grid(row=2, column=0, padx=10)
        self.canvas2.grid(row=2, column=2, padx=10) 
        self.canvas3.grid(row=4, column=0, padx=10) 
        self.canvas4.grid(row=4, column=2, padx=10) 
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.image)
        self.canvas2.create_image(0, 0, anchor=tk.NW, image=self.image_proc)
        self.canvas3.create_image(0, 0, anchor=tk.NW, image=self.image_proc_focus)
        self.canvas4.create_image(0, 0, anchor=tk.NW, image=self.image_proc_focus_gesture)
        
        # Btn start
        self.btn_start = tk.Button(self.window, text="START", width=20, command=self.start_video)
        self.btn_start.grid(row=0, column=4, padx=10)
        self.btn_reset = tk.Button(self.window, text="RESET", width=20, command=self.reset_flags)
        self.btn_reset.grid(row=0, column=5, padx=10)
        self.btn_reset = tk.Button(self.window, text="SAVE SCREEN", width=20, command=self.save)
        self.btn_reset.grid(row=0, column=6, padx=10)
        
        
    ############################## Core
    def start_video(self):
        # Launch
        self.image_bc = self.resize_image(cv2.flip(cv2.cvtColor(self.cap.read()[1], cv2.COLOR_BGR2RGB), 1)) # set background
        #cv2.namedWindow("Background", cv2.WINDOW_AUTOSIZE)
        #cv2.imshow("Background", cv2.cvtColor(self.image_bc, cv2.COLOR_RGB2BGR))
        
        # Text description
        self.Tdesc = tk.Text(self.window, height = 6, width = 40) 
        self.Tdesc.place(x=620, y=60)
        fact = """1- Depending on lighting scene use:
    - YCC when bad lighting
    - HSV for high lighting
    - Background when others don't work
[Reset button = reset all filters, and takes photo of the background]"""
        self.Tdesc.insert(tk.END, fact)

        # Button skin color        
        self.btn_noise = tk.Button(self.window, text="YCC skin detection", width=20, command=self.set_flag_skin_ycc)
        self.btn_noise.place(x=715, y=165) 
        self.btn_bc = tk.Button(self.window, text="HSV skin detection", width=20, command=self.set_flag_skin_hsv)
        self.btn_bc.place(x=715, y=195)
        # Button background cancelling 
        self.btn_hsv = tk.Button(self.window, text="Background cancelling", width=20, command=self.set_flag_bc)
        self.btn_hsv.place(x=715, y=225)
        
        self.Tdesc2 = tk.Text(self.window, height = 3, width = 40) 
        self.Tdesc2.place(x=620, y=280)
        fact = """3- Add noise reduction if necessary
4- Once filter chose, localize the hand"""
        self.Tdesc2.insert(tk.END, fact)
        
        # Button noise reduction  
        self.btn_ycc = tk.Button(self.window, text="Noise reduction", width=20, command=self.set_flag_noise)
        self.btn_ycc.place(x=715, y=340)
        # Button localization 
        self.btn_loc = tk.Button(self.window, text="Localization", width=20, command=self.set_flag_loc)
        self.btn_loc.place(x=715, y=370)
        
        
        self.Tdesc3 = tk.Text(self.window, height = 2, width = 40) 
        self.Tdesc3.place(x=620, y=430)
        fact = """5- Once hand localized use the gesture detection"""
        self.Tdesc3.insert(tk.END, fact)
        
        # Button gesture detection
        self.btn_loc = tk.Button(self.window, text="Detect gesture", width=30, command=self.set_flag_detect)
        self.btn_loc.place(x=680, y=490)
        # label detected
        self.label = tk.Label(self.window, text="Label Detected", font=("Calibri", 12), relief="groove", width=27)
        self.label.place(x=680, y=520)
        self.T = tk.Text(self.window, height = 1, width = 27) 
        self.T.place(x=680, y=540)
        
        self.update_image() # Update Frames on canvas
        
        
    def update_image(self):
        # Get the latest frame and convert image format
        self.image = self.resize_image(cv2.flip(cv2.cvtColor(self.cap.read()[1], cv2.COLOR_BGR2RGB), 1))
        self.image_proc = self.resize_image(cv2.flip(cv2.cvtColor(self.cap.read()[1], cv2.COLOR_BGR2RGB), 1))
        self.image_proc_focus = self.image_zero
        self.image_proc_focus_gesture = self.image_zero
        # Check effects on proc image
        self.activate_effects()
        # Rescale adn convert to photo
        self.image = self.img_to_photo(self.image)
        self.image_proc = self.img_to_photo(self.image_proc)
        self.image_proc_focus = self.img_to_photo(self.image_proc_focus)
        self.image_proc_focus_gesture = self.img_to_photo(self.image_proc_focus_gesture)
        # Update image
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.image)
        self.canvas2.create_image(0, 0, anchor=tk.NW, image=self.image_proc)
        self.canvas3.create_image(0, 0, anchor=tk.NW, image=self.image_proc_focus)
        self.canvas4.create_image(0, 0, anchor=tk.NW, image=self.image_proc_focus_gesture)
        # Repeat every 'interval' ms
        self.window.after(self.interval, self.update_image)
    
    ############################## Actions activation
    def activate_effects(self):
        flag_mask = False
        if self.flag_bc:
            self.image_proc = self.backgroud_cancelling(self.image_bc, self.image_proc)
        if self.flag_skin_hsv:
            self.image_proc = self.skin_hsv_detection(self.image_proc)
        if self.flag_skin_ycc:
            self.image_proc = self.skin_ycc_detection(self.image_proc)
        if self.flag_noise:
            self.image_proc = self.noise_reduction(self.image_proc)
        if self.flag_loc:
            image_copy = self.image.copy()
            image_proc_copy = self.image_proc.copy()
            #self.image_proc, boundrec, contours, contour_output, clean_filled = self.localization(self.image_proc)
            #self.image_proc_focus = self.resize_image_boundrec(image_copy, boundrec)
            self.image_proc_focus, boundrec, contours, contour_output, clean_filled = self.localization(self.image_proc)
            image_proc_copy = self.resize_image_boundrec(clean_filled, boundrec)
            self.image_save = image_proc_copy
            if self.flag_detect:
                self.image_proc_focus_gesture, label = self.gesture_detection(image_proc_copy, contours)
                # resize on bounding box
                #self.image_proc_focus_gesture = image_copy
                # Write result
                fact = str(label) 
                self.T.delete('1.0', tk.END)
                self.T.insert(tk.END, fact)
                
            # What to visualize
            
                
    def set_flag_bc(self):
        if self.flag_bc:
            self.flag_bc = False
        else:
            self.flag_bc = True
    def set_flag_skin_hsv(self):
        if self.flag_skin_hsv:
            self.flag_skin_hsv = False
        else:
            self.flag_skin_hsv = True
    def set_flag_skin_ycc(self):
        if self.flag_skin_ycc:
            self.flag_skin_ycc = False
        else:
            self.flag_skin_ycc = True
    def set_flag_noise(self):
        if self.flag_noise:
            self.flag_noise = False
        else:
            self.flag_noise = True
    def set_flag_loc(self):
        if self.flag_loc:
            self.flag_loc = False
        else:
            self.flag_loc = True
    def set_flag_detect(self):
        if self.flag_detect:
            self.flag_detect = False
        else:
            self.flag_detect = True
    def reset_flags(self):
        self.flag_noise = False
        self.flag_bc = False
        self.flag_skin_hsv = False
        self.flag_skin_ycc = False
        self.flag_loc = False
        self.flag_detect = False
        self.image_bc = self.resize_image(cv2.flip(cv2.cvtColor(self.cap.read()[1], cv2.COLOR_BGR2RGB), 1)) # set background
        
        
        
    ############################## Actions
    @staticmethod
    def noise_reduction(img):
        #img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        #img = cv2.GaussianBlur(img, (5, 5), 0)
        red = 4
        # large size object detection against small size noise must have large erosion and dilation size
        erosion_size = red
        dilation_size = red
        element = np.ones((erosion_size, erosion_size))
        img = cv2.erode(img, element, iterations = 2)
        img = cv2.dilate(img, element, iterations = 2)
        #img = cv2.GaussianBlur(img, (3, 3), 0)
        return img
    @staticmethod
    def backgroud_cancelling(bc, image):
        #bc = cv2.cvtColor(bc, cv2.COLOR_RGB2GRAY)
        #image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        diff = cv2.absdiff(bc, image)
        diff = cv2.cvtColor(diff, cv2.COLOR_RGB2GRAY)
        #diff = bc - image
        ret, thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY |cv2.THRESH_OTSU)
        return thresh
    @staticmethod
    def skin_hsv_detection(img):
        original = img.copy()
        rgba = cv2.cvtColor(img, cv2.COLOR_RGB2RGBA) #Red-Green-Blue
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV) #Hue-Saturation-Value
        ycc = cv2.cvtColor(img, cv2.COLOR_RGB2YCR_CB) #Luminance-Chrominance
        # Bounds
        hsv_lower = np.array([0, 48, 80]) 
        hsv_upper = np.array([20, 250, 250])
        mask_hsv = cv2.inRange(hsv, hsv_lower, hsv_upper)
        blurred = cv2.blur(mask_hsv, (2,2))
        result = blurred
        return result
    @staticmethod
    def skin_ycc_detection(img):
        original = img.copy()
        ycc = cv2.cvtColor(img, cv2.COLOR_RGB2YCR_CB) #Luminance-Chrominance
        # Bounds
        ycc_lower = np.array([0, 133, 77],np.uint8)
        ycc_upper = np.array([200, 173, 127],np.uint8) # 235
        mask_ycc = cv2.inRange(ycc, ycc_lower, ycc_upper)
        blurred = cv2.blur(mask_ycc, (2,2))
        result = blurred
        return result
    @staticmethod
    def localization(thres_output): # need input from processed image of a treshold
        contours, hierarchy = cv2.findContours(thres_output, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contour_output = cv2.cvtColor(np.zeros(np.shape(thres_output), dtype='uint8'), cv2.COLOR_GRAY2BGR)
        clean_filled = cv2.cvtColor(np.zeros(np.shape(thres_output), dtype='uint8'), cv2.COLOR_GRAY2BGR)
        if (len(contours) > 0):
            # Find largest contour
            max_id = max(enumerate(contours), key=lambda x : cv2.contourArea(x[1]))[0]
            max_size = cv2.contourArea(contours[max_id])
            boundrec = cv2.boundingRect(contours[max_id])
            cv2.drawContours(contour_output, contours, max_id, (255, 0, 0), cv2.FILLED)
            cv2.drawContours(clean_filled, contours, max_id, (255, 255, 255), cv2.FILLED)
            cv2.drawContours(contour_output, contours, max_id, (0, 0, 255), 2)
            cv2.rectangle(contour_output, boundrec, (0, 255, 0), 2)
            #contours = np.delete(contours, max_id)
        return contour_output, boundrec, contours, contour_output, clean_filled
    
    ############################## Tools
    def save(self):
        cv2.imwrite("img-saved.jpg", self.image_save)
    @staticmethod
    def resize_image(img):
        # Resize
        height, width, *_ = img.shape
        HEIGHT = 250   
        imgScale = HEIGHT/height
        newX, newY = img.shape[1]*imgScale, img.shape[0]*imgScale
        img = cv2.resize(img, (int(newX), int(newY)))
        img = img[50:int(img.shape[1])+50, 0:int(img.shape[0])]
        return img
    @staticmethod 
    def resize_image_boundrec(image_copy, boundrec):
            x = boundrec[0]
            y = boundrec[1]
            w = boundrec[2]
            h = boundrec[3]
            image_copy = image_copy[y:y+h, x:x+w]
            height, width, *_ = image_copy.shape
            if width >= height:
                WIDTH = 250   
                imgScale = WIDTH/width
            else:
                HEIGHT = 250   
                imgScale = HEIGHT/height
            newX, newY = image_copy.shape[1]*imgScale, image_copy.shape[0]*imgScale
            image_copy = cv2.resize(image_copy, (int(newX), int(newY)))
            return image_copy
    @staticmethod
    def img_to_photo(img):
        # Transform to photo
        photo = Image.fromarray(img) # to PIL format
        photo = ImageTk.PhotoImage(photo) # to ImageTk format
        return photo
    @staticmethod
    def gesture_detection(img, contours):
        templates = ['5.jpg','4.jpg','3.jpg','2.jpg','1.jpg','up.jpg','down.jpg']
        max_result = 0
        max_ind = 0
        h_img, w_img, *_ = img.shape
        for i in range(len(templates)): 
            temp_original = cv2.imread(templates[i])
            if "down" in templates[i] or "up" in templates[i]:
                max_range = 1
            else:
                max_range = 16
            for y in range(max_range):
                if y == 0:
                    temp = temp_original 
                elif y == 1:
                    temp = cv2.flip(temp, 1) 
                elif y == 2:
                    temp = cv2.flip(temp, -1)  
                elif y == 3:
                    temp = cv2.flip(temp, 0)  
                elif y == 4:
                    temp = cv2.rotate(temp, cv2.ROTATE_180) 
                elif y == 5:
                    temp = cv2.rotate(temp_original, cv2.ROTATE_90_CLOCKWISE) 
                elif y == 6:
                    temp = cv2.rotate(temp, cv2.ROTATE_90_COUNTERCLOCKWISE)
                elif y == 7:
                    temp = cv2.flip(temp, 1) 
                    temp = cv2.rotate(temp, cv2.ROTATE_180) 
                elif y == 8:
                    temp = cv2.flip(temp, -1) 
                    temp = cv2.rotate(temp, cv2.ROTATE_180) 
                elif y == 9:
                    temp = cv2.flip(temp, 0)  
                    temp = cv2.rotate(temp, cv2.ROTATE_180)  
                elif y == 10:
                    temp = cv2.flip(temp, 1) 
                    temp = cv2.rotate(temp_original, cv2.ROTATE_90_CLOCKWISE)
                elif y == 11:
                    temp = cv2.flip(temp, -1) 
                    temp = cv2.rotate(temp_original, cv2.ROTATE_90_CLOCKWISE) 
                elif y == 12:
                    temp = cv2.flip(temp, 0) 
                    temp = cv2.rotate(temp_original, cv2.ROTATE_90_CLOCKWISE)
                elif y == 13:
                    temp = cv2.flip(temp, 1) 
                    temp = cv2.rotate(temp, cv2.ROTATE_90_COUNTERCLOCKWISE)  
                elif y == 14:
                    temp = cv2.flip(temp, -1) 
                    temp = cv2.rotate(temp, cv2.ROTATE_90_COUNTERCLOCKWISE) 
                elif y == 15:
                    temp = cv2.flip(temp, 0) 
                    temp = cv2.rotate(temp, cv2.ROTATE_90_COUNTERCLOCKWISE)
                    
                h_temp, w_temp, *_ = temp.shape
                # Since we are already localizing the hand, I need for matching the same size template
                width = w_img
                height = h_img
                h, w = temp.shape[:2]
                pad_bottom, pad_right = 0, 0
                ratio = w / h
                if h > height or w > width:
                    interp = cv2.INTER_AREA # shrinking image algorithm
                else:
                    interp = cv2.INTER_CUBIC # stretching image algorithm
                w = width
                h = round(w / ratio)
                if h > height:
                    h = height
                    w = round(h * ratio)
                pad_bottom = abs(height - h)
                pad_right = abs(width - w)
                scaled_img = cv2.resize(temp, (w, h), interpolation=interp)
                temp = cv2.copyMakeBorder(scaled_img,0,pad_bottom,0,pad_right,borderType=cv2.BORDER_CONSTANT,value=[0,0,0])
                result = cv2.matchTemplate(img, temp, cv2.TM_CCOEFF_NORMED)
                
                if result[0][0] > max_result:
                    max_result = result[0][0]
                    max_ind = i
                
        '''cv2.namedWindow("Match", cv2.WINDOW_AUTOSIZE)
        cv2.imshow("Match", templates[i])'''
        label = templates[max_ind].replace(',jpg', '') + " - " + str(max_result)
        match_temp = templates[max_ind]
        return cv2.imread(templates[max_ind]), label
        
        
        



if __name__ == "__main__":
    root = tk.Tk()
    MainWindow(root, cv2.VideoCapture(0))
    root.mainloop()