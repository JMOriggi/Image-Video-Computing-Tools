import numpy as np
import cv2
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import time, os, random, string, argparse
from pathlib import Path
from PIL import ImageTk, Image


class Page(tk.Tk):

    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)
        self.title("Image Processing")
        
        container=tk.Frame(self)
        container.grid()

        # menu
        self.option_add('*tearOff', False)
        menubar = tk.Menu(self)
        self.config(menu = menubar)
        file = tk.Menu(menubar)
        help_ = tk.Menu(menubar)

        menubar.add_cascade(menu = file, label = "File")
        file.add_command(label = 'Open...', command=lambda:self.show_image())
        menubar.add_cascade(menu = help_, label = "Help")
        help_.add_command(label = 'About', command=lambda:self.about())

        # title
        self.empty_name=tk.Label(self, text="Image Processing", font=("Arial", 16))
        self.empty_name.grid(row=0, column=0, pady=5, padx=10, sticky="sw")

        # intro
        self.intro_lbl = tk.Label(self, text="""Sandbox to manipulate images with different filters."""
                                  , font=("Arial", 11), fg="#202020")
        self.intro_lbl.grid(row=1, column=0, columnspan=3, padx=10, pady=10, sticky="nw")

        # select image                          
        self.browse_lbl = tk.Label(self, text="Select Image ", font=("Arial", 10), fg="#202020")
        self.browse_lbl.grid(row=3, column=0, columnspan=3, padx=10, pady=10, sticky="w")

        self.browse_entry=tk.Entry(self, text="", width=30)
        self.browse_entry.grid(row=3, column=0, columnspan=3, padx=200, pady=10, sticky="w")

        self.browse_btn = tk.Button(self, text="     Browse     ", bg="#ffffff", relief="raised", width=10, command=lambda:self.show_image())
        self.browse_btn.grid(row=3, column=0, padx=110, pady=10, columnspan=3, sticky="w")
        
        
        self.T = tk.Text(self, height = 6, width = 45)
        Fact = """1. Browse an Image
2. Try a filter button
3. Apply button
4. Another button filter (added to previous)
--> Use Localization filter only after gray filter is applied"""
        self.T.insert(tk.END, Fact) 
        self.T.grid(row=7, column=0, padx=10, pady=0, columnspan=3,  sticky="w")
        
        # file info
        self.lbl_filename = tk.Label(self, text="File Name: ", font=("Arial", 10), fg="#202020")
        self.lbl_filesize = tk.Label(self, text="File Size: ", font=("Arial", 10), fg="#202020") 

        self.label_text_x = tk.StringVar()
        self.lbl_filename_01 = tk.Label(self, textvariable=self.label_text_x, font=("Arial", 10),fg="#202020")
        
        self.text_file_size=tk.StringVar()
        self.lbl_filesize_01 = tk.Label(self, textvariable=self.text_file_size, font=("Arial", 10), fg="#202020")
        
        # place holder for document thumbnail
        self.lbl_image = tk.Label(self, image="")
        self.lbl_image.grid(row=8, column=0, pady=25, padx=10, columnspan=3, sticky="nw")
        
        # Image processed
        self.lbl_image2 = tk.Label(self, image="")
        self.lbl_image2.grid(row=8, column=2, pady=25, padx=10, columnspan=3, sticky="nw")

        # status text 
        #self.label_text_progress = tk.StringVar()
        #self.scan_progress = tk.Label(self, textvariable=self.label_text_progress, font=("Arial", 10),fg="#0000ff")
                
        
        # Buttons
        self.noise_entry_lb = tk.StringVar()
        self.noise_entry=tk.Entry(self,textvariable=self.noise_entry_lb, width=5)
        self.tresh_entry_lb = tk.StringVar()
        self.tresh_entry=tk.Entry(self,textvariable=self.tresh_entry_lb, width=5)
        self.bnr_bt = tk.Button(self, text="   Noise red   ", bg="#ffffff", relief="raised", width=10, command=lambda:self.binary_noise_reduction(self.proc_image, int(self.noise_entry.get()), False))
        self.loc_bt = tk.Button(self, text="  Localization  ", bg="#ffffff", relief="raised", width=10, command=lambda:self.binary_contours(self.proc_image, int(self.tresh_entry.get()), False))
        self.gray_bt = tk.Button(self, text="  Gray  ", bg="#ffffff", relief="raised", width=10, command=lambda:self.gray())
        self.tune_bt = tk.Button(self, text="  Tune noise and localization ", bg="#ffffff", relief="raised"
                                 , width=25, command=lambda:self.tune())
        self.clear_bt = tk.Button(self, text="     Clear      ", bg="#ffffff", relief="raised", width=10, command=lambda:self.clear())
        self.apply_bt = tk.Button(self, text="     <- Apply      ", bg="#ffffff", relief="raised", width=10, command=lambda:self.apply())
        
        # text area to place text
        #self.ocr_text = tk.Text(self, height=25, width=38)  
       
    
    def show_image(self):
        global path
        
        # open file dialog
        self.path = filedialog.askopenfilename(defaultextension="*.jpg", filetypes = (("JPG", "*.jpg"),("PNG","*.png")))
        self.browse_entry.delete(0, tk.END)
        self.browse_entry.insert(0, self.path)
        
        self.noise_entry_lb.set("10")
        self.tresh_entry_lb.set("88")
        #entry = self.noise_entry.get()
        #self.label_text_progress.set("Add different filters. For noise reduction and localization find the correct value of sensibility first. Localization filter only works after applying gray filter.");
        #self.scan_progress.grid(row=16, column=2, padx=10, pady=0, columnspan=3, sticky="w")

        # resize image
        cv_img = cv2.cvtColor(cv2.imread(self.path), cv2.COLOR_BGR2RGB)
        photo = self.image_to_photo(cv_img)
        self.source_image = cv_img
        self.proc_image = cv_img
        
        # show image
        self.lbl_image.configure(image=photo)
        self.lbl_image.image=photo

        # show image2
        white = np.zeros([1000,800,3],dtype=np.uint8)
        white.fill(255)
        photo = self.image_to_photo(white)
        self.lbl_image2.configure(image=photo)
        self.lbl_image2.image=photo

        # show file information
        self.lbl_filename.grid(row=4, column=0, pady=0, padx=10, columnspan=3, sticky="nw")
        self.lbl_filename_01.grid(row=4, column=0, pady=0, padx=85, columnspan=3, sticky="nw")
        self.lbl_filesize.grid(row=5, column=0, pady=0, padx=10, sticky="nw")
        self.lbl_filesize_01.grid(row=5, column=0, pady=0, padx=75, columnspan=3, sticky="nw")

        # add buttons
        pad_mid = 70;       
        self.clear_bt.grid(row=9, column=0, padx=pad_mid, pady=3, columnspan=3, sticky="w") 
        self.apply_bt.grid(row=8, column=1, padx=0, pady=3, columnspan=3, sticky="w") 
        self.gray_bt.grid(row=9, column=2, padx=10, pady=3, columnspan=3, sticky="w") 
        self.tune_bt.grid(row=10, column=3, padx=0, pady=3, columnspan=3, sticky="w")     
        self.bnr_bt.grid(row=11, column=2, padx=10, pady=3, columnspan=3, sticky="w")
        self.noise_entry.grid(row=11, column=3, padx=0, pady=3, columnspan=3, sticky="w")
        self.loc_bt.grid(row=12, column=2, padx=10, pady=3, columnspan=3, sticky="w")
        self.tresh_entry.grid(row=12, column=3, padx=0, pady=3, columnspan=3, sticky="w")
        #self.ocr_text.grid(row=8, column=0, padx=350, pady=26, columnspan=3, sticky="w")
        
        # set the filename
        self.label_text_x.set(os.path.basename(self.path))
        
        # set the filesize
        self.text_file_size.set(os.path.getsize(self.path))
   
    
        
    @staticmethod
    def image_to_photo(cv_img):
        height, width, *_ = cv_img.shape
        HEIGHT = 300   
        imgScale = HEIGHT/height
        newX, newY = cv_img.shape[1]*imgScale, cv_img.shape[0]*imgScale
        newimg = cv2.resize(cv_img, (int(newX), int(newY)))
        photo = ImageTk.PhotoImage(image = Image.fromarray(newimg))
        return photo
    
    def clear(self):
        photo = self.image_to_photo(self.source_image)
        self.lbl_image.configure(image=photo)
        self.lbl_image.image=photo
        white = np.zeros([1000,800,3],dtype=np.uint8)
        white.fill(255)
        photo = self.image_to_photo(white)
        self.lbl_image2.configure(image=photo)
        self.lbl_image2.image=photo
        self.proc_image = self.source_image
        self.effect_image = self.source_image
        
   
    def apply(self):
        self.proc_image = self.effect_image
        photo = self.image_to_photo(self.effect_image)
        self.lbl_image.configure(image=photo)
        self.lbl_image.image=photo
        self.lbl_image2.pack_forget()
        
    
    def gray(self):
        self.effect_image = cv2.cvtColor(self.proc_image, cv2.COLOR_BGR2GRAY)
        photo = self.image_to_photo(self.effect_image)
        self.lbl_image2.configure(image=photo)
        self.lbl_image2.image=photo
       

    def blur(self):
        # Convert image to gray and blur
        self.effect_image = cv2.blur(self.proc_image, (3, 3))
        photo = self.image_to_photo(self.effect_image)
        self.lbl_image2.configure(image=photo)
        self.lbl_image2.image=photo
        
        
   # Clean from noise (2 type of noise in binary img: inside the object, in the background)
    def binary_noise_reduction(self, img, red, flag_print):
        # large size object detection against small size noise must have large erosion and dilation size
        erosion_size = red
        dilation_size = red
        element = np.ones((erosion_size,erosion_size))
        img = cv2.erode(img, element)
        img = cv2.erode(img, element)
        img = cv2.dilate(img, element)
        img = cv2.dilate(img, element)
        img_proc = cv2.dilate(img, element)  
        if flag_print:
            cv2.namedWindow("Noise_reduction", cv2.WINDOW_AUTOSIZE)
            cv2.imshow("Noise_reduction", img_proc)
        else:
            self.effect_image = img_proc
            photo = self.image_to_photo(img_proc)
            self.lbl_image2.configure(image=photo)
            self.lbl_image2.image=photo
            
       
    # Binaries the image & find contours
    def binary_contours(self, img, thresh,flag_print):
        _, thres_output = cv2.threshold(img, thresh, 255, 0)
        #cv2.namedWindow('Thres')
        #cv2.imshow('Thres', thres_output)
        contours, hierarchy = cv2.findContours(thres_output, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contour_output = cv2.cvtColor(np.zeros(np.shape(thres_output), dtype='uint8'), cv2.COLOR_GRAY2BGR)
        for i in range(len(contours)):
            size = cv2.contourArea(contours[i])
            if size < 90:
                continue
            boundrec = cv2.boundingRect(contours[i])
            cv2.drawContours(contour_output, contours, i, (255, 0, 0), cv2.FILLED, 8)
            cv2.drawContours(contour_output, contours, i, (0, 0, 255), 2, 8)
            cv2.rectangle(contour_output, boundrec, (0, 255, 0), 1, 8, 0)
        if flag_print:
            cv2.namedWindow("Contours", cv2.WINDOW_AUTOSIZE)
            cv2.imshow("Contours", contour_output)
        else:
            self.effect_image = contour_output
            photo = self.image_to_photo(contour_output)
            self.lbl_image2.configure(image=photo)
            self.lbl_image2.image=photo
            
        
    def tune(self):
        cv2.namedWindow("Reference", cv2.WINDOW_AUTOSIZE)
        cv2.imshow("Reference", self.proc_image)
        cv2.createTrackbar("Noise:", "Reference", 2, 100, lambda x: self.binary_noise_reduction(self.proc_image, x, True))
        cv2.createTrackbar("Threshold:", "Reference", 30, 255, lambda x: self.binary_contours(self.proc_image, x, True))


    def about(self):
        # show about message
        messagebox.showinfo(title = 'About', message = 'Made by Juan Manuel Origgi inspired by a project by Rick Torzynski')
        

if __name__ == "__main__":
    app = Page()
    app.geometry("800x800")
    app.mainloop()
