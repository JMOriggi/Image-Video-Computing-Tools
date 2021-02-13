import numpy as np
import tkinter as tk
from PIL import Image, ImageTk
import cv2
import math


class MainWindow():
    def __init__(self, window, cap): 
        
        # Vars
        self.window = window
        self.window.geometry("950x650")
        self.cap = cap
        self.interval = 200 # Interval in ms to get the latest frame
        self.width = 250
        self.height = 250
        self.flag_noise = False
        self.flag_bc = False
        self.flag_loc = False
        self.flag_skin_hsv = False
        self.flag_skin_ycc = False
          
        # First Frame zero
        zero_img = np.zeros([self.height, self.width, 3],dtype=np.uint8)
        zero_img.fill(255)
        zero_img = self.resize_image(zero_img)
        self.image_zero = zero_img
        self.image = self.img_to_photo(self.image_zero)
        self.image_proc = self.img_to_photo(self.image_zero)
        self.image_proc_focus = self.img_to_photo(self.image_zero)
        
        # Labels
        self.label = tk.Label(self.window, text="HAND GESTURES DETECTION", font=("Calibri", 16))
        self.label.grid(row=0, column=0, padx=10, pady=5, sticky="sw")
        self.label = tk.Label(self.window, text="Original Frames", font=("Calibri", 12), relief="groove", width=35)
        self.label.grid(row=1, column=0, padx=10, pady=5, sticky="sw")
        self.label = tk.Label(self.window, text="Processed Frames", font=("Calibri", 12), relief="groove", width=35)
        self.label.grid(row=1, column=2, padx=10, pady=5, sticky="sw")
        self.label = tk.Label(self.window, text="Localization Focus", font=("Calibri", 12), relief="groove", width=35)
        self.label.grid(row=3, column=2, padx=10, pady=5, sticky="sw")
        
        # Videos canvas
        '''self.width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)'''
        self.canvas = tk.Canvas(self.window, width=self.width, height=self.height)
        self.canvas.grid(row=2, column=0, padx=10)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.image)
        self.canvas2 = tk.Canvas(self.window, width=self.width, height=self.height)
        self.canvas2.grid(row=2, column=2, padx=10) # to RGB
        self.canvas2.create_image(0, 0, anchor=tk.NW, image=self.image_proc)
        self.canvas3 = tk.Canvas(self.window, width=self.width, height=self.height)
        self.canvas3.grid(row=4, column=2, padx=10) # to RGB
        self.canvas3.create_image(0, 0, anchor=tk.NW, image=self.image_proc_focus)
        
        # Btn start
        self.btn_start = tk.Button(self.window, text="START", width=20, command=self.start_video)
        self.btn_start.grid(row=0, column=4, padx=10)
        self.btn_reset = tk.Button(self.window, text="RESET", width=20, command=self.reset_flags)
        self.btn_reset.grid(row=0, column=5, padx=10)
        
        
    ############################## Core
    def start_video(self):
        # Launch
        self.image_bc = self.resize_image(cv2.flip(cv2.cvtColor(self.cap.read()[1], cv2.COLOR_BGR2RGB), 1)) # set background
        #cv2.namedWindow("Background", cv2.WINDOW_AUTOSIZE)
        #cv2.imshow("Background", cv2.cvtColor(self.image_bc, cv2.COLOR_RGB2BGR))
        
        self.update_image() # Update Frames on canvas
        
        # Button noise reduction           
        self.btn_noise = tk.Button(self.window, text="Noise reduction", width=20, command=self.set_flag_noise)
        self.btn_noise.place(x=725, y=75)
        # Button background cancelling  
        self.btn_bc = tk.Button(self.window, text="Background cancelling", width=20, command=self.set_flag_bc)
        self.btn_bc.place(x=725, y=125)
        # Button skin color
        self.btn_hsv = tk.Button(self.window, text="HSV skin detection", width=20, command=self.set_flag_skin_hsv)
        self.btn_hsv.place(x=725, y=175)
        self.btn_ycc = tk.Button(self.window, text="YCC skin detection", width=20, command=self.set_flag_skin_ycc)
        self.btn_ycc.place(x=725, y=225)
        # Button localization 
        self.btn_loc = tk.Button(self.window, text="Localization", width=20, command=self.set_flag_loc)
        self.btn_loc.place(x=725, y=275)
        
        
    def update_image(self):
        # Get the latest frame and convert image format
        self.image = self.resize_image(cv2.flip(cv2.cvtColor(self.cap.read()[1], cv2.COLOR_BGR2RGB), 1))
        self.image_proc = self.resize_image(cv2.flip(cv2.cvtColor(self.cap.read()[1], cv2.COLOR_BGR2RGB), 1))
        self.image_proc_focus = self.image_zero
        # Check effects on proc image
        self.activate_effects()
        # Rescale adn convert to photo
        self.image = self.img_to_photo(self.image)
        self.image_proc = self.img_to_photo(self.image_proc)
        self.image_proc_focus = self.img_to_photo(self.image_proc_focus)
        # Update image
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.image)
        self.canvas2.create_image(0, 0, anchor=tk.NW, image=self.image_proc)
        self.canvas3.create_image(0, 0, anchor=tk.NW, image=self.image_proc_focus)
        # Repeat every 'interval' ms
        self.image_prev = self.image_proc
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
            self.image_proc, boundrec = self.localization(self.image_proc)
            x = boundrec[0]
            y = boundrec[1]
            w = boundrec[2]
            h = boundrec[3]
            image_copy = self.image.copy()
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
            self.image_proc_focus = image_copy
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
    def reset_flags(self):
        self.flag_noise = False
        self.flag_bc = False
        self.flag_skin_hsv = False
        self.flag_skin_ycc = False
        self.flag_loc = False
        self.start_video()
        
        
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
        img = cv2.GaussianBlur(img, (3, 3), 0)
        return img
    @staticmethod
    def backgroud_cancelling(bc, image):
        bc = cv2.cvtColor(bc, cv2.COLOR_RGB2GRAY)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        diff = cv2.absdiff(bc, image)
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
        result = mask_hsv
        return result
    @staticmethod
    def skin_ycc_detection(img):
        original = img.copy()
        ycc = cv2.cvtColor(img, cv2.COLOR_RGB2YCR_CB) #Luminance-Chrominance
        # Bounds
        ycc_lower = np.array([0, 133, 77],np.uint8)
        ycc_upper = np.array([235, 173, 127],np.uint8)
        mask_ycc = cv2.inRange(ycc, ycc_lower, ycc_upper)
        result = mask_ycc
        return result
    @staticmethod
    def localization(thres_output): # need input from processed image of a treshold
        contours, hierarchy = cv2.findContours(thres_output, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contour_output = cv2.cvtColor(np.zeros(np.shape(thres_output), dtype='uint8'), cv2.COLOR_GRAY2BGR)
        if (len(contours) > 0):
            # Find largest contour
            max_id = max(enumerate(contours), key=lambda x : cv2.contourArea(x[1]))[0]
            max_size = cv2.contourArea(contours[max_id])
            boundrec = cv2.boundingRect(contours[max_id])
            cv2.drawContours(contour_output, contours, max_id, (255, 0, 0), cv2.FILLED)
            cv2.drawContours(contour_output, contours, max_id, (0, 0, 255), 2)
            cv2.rectangle(contour_output, boundrec, (0, 255, 0), 2)
            contours = np.delete(contours, max_id)
            '''if (len(contours) > 1):
                # Find largest contour
                max_id = max(enumerate(contours), key=lambda x : cv2.contourArea(x[1]))[0]
                max_size = cv2.contourArea(contours[max_id])
                if max_size > 100:
                    boundrec = cv2.boundingRect(contours[max_id])
                    cv2.drawContours(contour_output, contours, max_id, (255, 0, 0), cv2.FILLED, 8)
                    cv2.drawContours(contour_output, contours, max_id, (0, 0, 255), 2, 8)
                    cv2.rectangle(contour_output, boundrec, (0, 255, 0), 1, 8, 0)'''
        return contour_output, boundrec
    
    ############################## Tools
    @staticmethod
    def resize_image(img):
        # Resize
        height, width, *_ = img.shape
        HEIGHT = 250   
        imgScale = HEIGHT/height
        newX, newY = img.shape[1]*imgScale, img.shape[0]*imgScale
        img = cv2.resize(img, (int(newX), int(newY)))
        img = img[50:int(img.shape[1])+50, 0:int(img.shape[0])]
        '''w = img.shape[1]
        h = img.shape[0]
        mid_x, mid_y = int(w/2), int(h/2)
        cw2, ch2 = int(500/2), int(500/2)
        crop_img = img[mid_y-ch2:mid_y+ch2, mid_x-cw2:mid_x+cw2]'''
        return img
    @staticmethod
    def img_to_photo(img):
        # Transform to photo
        photo = Image.fromarray(img) # to PIL format
        photo = ImageTk.PhotoImage(photo) # to ImageTk format
        return photo
    @staticmethod
    def calculateFingers(res, drawing):
        #  convexity defect
        hull = cv2.convexHull(res, returnPoints=False)
        if len(hull) > 3:
            defects = cv2.convexityDefects(res, hull)
            if defects is not None:
                cnt = 0
                for i in range(defects.shape[0]):  # calculate the angle
                    s, e, f, d = defects[i][0]
                    start = tuple(res[s][0])
                    end = tuple(res[e][0])
                    far = tuple(res[f][0])
                    a = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
                    b = math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
                    c = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
                    angle = math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c))  # cosine theorem
                    if angle <= math.pi / 2:  # angle less than 90 degree, treat as fingers
                        cnt += 1
                        cv2.circle(drawing, far, 8, [211, 84, 0], -1)
                if cnt > 0:
                    return True, cnt+1
                else:
                    return True, 0
        return False, 0
        



if __name__ == "__main__":
    root = tk.Tk()
    MainWindow(root, cv2.VideoCapture(0))
    root.mainloop()