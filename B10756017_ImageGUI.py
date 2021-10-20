# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 11:40:42 2020

@author: JunYi
"""
import tkinter as tk
import numpy as np
import cv2
from PIL import Image,ImageTk
from tkinter import messagebox,filedialog
#import sys
#import matplotlib.pyplot as plt

Gcheck = False
Bcheck = False

def open_file():
    global img,img_c
    win.filename = filedialog.askopenfilename(initialdir = "/",title = "Select file",filetypes = (("jpeg files","*.jpg"),("png files","*.png"),("all files","*.*")))
    if win.filename != "":
        btn_sv.config(state="normal")
        btn_re.config(state="normal")
        btn_gray.config(state="normal")
        btn_salt.config(state="normal")
        btn_gasuss.config(state="normal")
        img = cv2.imdecode(np.fromfile(win.filename,dtype=np.uint8),1)  #可讀中文檔
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB) #TK的顯示為RGB格式
        img = cv2.resize(img,(1024,768), cv2.INTER_AREA)
        img_c = Image.fromarray(img)
        img_c = ImageTk.PhotoImage(image=img_c)
        lb_img.imgtk = img_c
        lb_img.configure(image=img_c)
        return img

def save_file():
    global img_s,Gcheck,Bcheck
    win.filesave = filedialog.asksaveasfilename(initialdir = "/",title = "Select file",filetypes = (("jpeg files","*.jpg"),("png files","*.png")))
    if win.filesave !="":
        name = str(win.filesave)
        name = name.split(".")
        name = name[len(name)-1]    #讀取副檔名
        str1 = en4.get()
        size = str1.split("x") #將str以x為界分割成array
        num = len(size)
        if str1 == "":    
            if Gcheck == False and Bcheck == False:
                img_s = cv2.cvtColor(img_s,cv2.COLOR_RGB2BGR)
        elif num == 1:
            if Gcheck == False and Bcheck == False:
                img_s = cv2.cvtColor(img_s,cv2.COLOR_RGB2BGR)
                img_s = cv2.resize(img_s,(int(size[0]),int(size[0])), cv2.INTER_AREA)
            else:
                img_s = cv2.resize(img_s,(int(size[0]),int(size[0])), cv2.INTER_AREA)
        elif(num > 2):
            messagebox.showerror("Error", "Please Input one or two Value")
            return
        else:
            if Gcheck == False and Bcheck == False:
                img_s = cv2.cvtColor(img_s,cv2.COLOR_RGB2BGR)
                img_s = cv2.resize(img_s,(int(size[0]),int(size[1])), cv2.INTER_AREA)
            else:
                img_s = cv2.resize(img_s,(int(size[0]),int(size[1])), cv2.INTER_AREA)
        if name == "jpg":
            cv2.imencode('.jpg',img_s)[1].tofile(win.filesave)
        else:
            cv2.imencode('.png',img_s)[1].tofile(win.filesave) #[0]為bool值

def Quit():
    win.destroy()
    
def recover(): #還原圖像
    global img_g,img_gs,img_c,img_s,Gcheck,Bcheck
    img_gs = img
    img_g = img
    img_s = img
    Gcheck = False
    Bcheck = False
    lb_img.configure(image=img_c)
    
def img_gray(img):
    global img_g,img_s,Gcheck
    btn_thresh.config(state="normal")
    btn_sob.config(state="normal")
    img_g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # 轉為灰階圖
    gray = Image.fromarray(img_g) #array轉image
    img_c = ImageTk.PhotoImage(image=gray)
    lb_img.imgtk = img_c
    lb_img.configure(image=img_c)
    img_s = img_g
    Gcheck = True
    return img_g

def img_thresh(gray):
    global img_s,Bcheck
    ret,thresh1 = cv2.threshold(gray,127,255,0)
    img_s = thresh1
    thresh = Image.fromarray(thresh1)
    img_c = ImageTk.PhotoImage(image=thresh)
    lb_img.imgtk = img_c
    lb_img.configure(image=img_c)
    Bcheck = True
    return thresh1

def eroded(img):    #侵蝕
    global img_s
    erode = cv2.erode(img,(3,3))
    img_s = erode
    erode = Image.fromarray(erode)
    erode = ImageTk.PhotoImage(image=erode)
    lb_img.imgtk = erode
    lb_img.configure(image=erode)
    return eroded

def dilated(img):   #膨脹
    global img_s
    dilate = cv2.dilate(img,(-1,-1))
    img_s = dilate
    dilate = Image.fromarray(dilate)
    dilate = ImageTk.PhotoImage(image=dilate)
    lb_img.imgtk = dilate
    lb_img.configure(image=dilate)

def sobel(img):
    global img_s
    x = cv2.Sobel(img,cv2.CV_16S,1,0)# Sobel會有負值，及大於255的值，因此要轉為16位元
    y = cv2.Sobel(img,cv2.CV_16S,0,1)
    absX = cv2.convertScaleAbs(x)   # 轉回uint8
    absY = cv2.convertScaleAbs(y)
    dst = cv2.addWeighted(absX,0.5,absY,0.5,0)# 由於sobel是在兩個方向做計算，最後還需將其組合起來
    img_s = dst
    dst = Image.fromarray(dst)
    dst = ImageTk.PhotoImage(image=dst)
    lb_img.imgtk = dst
    lb_img.configure(image=dst)
    return dst
    
def addsalt(img, SNR):
    global img_gs,img_s
    if(en1.get()==""):
        messagebox.showerror("Error", "Please Input Value")  #標題，訊息
    else:
        SNR = float(en1.get())
        if(SNR >1 ):
            messagebox.showerror("Error", "Please Input Value(<1)")  #標題，訊息
        else:
            btn_ero.config(state="normal")
            btn_dil.config(state="normal")
            btn_med.config(state="normal")
            img_gs = img.copy()
            c, h, w = img_gs.shape
            mask = np.random.choice((0, 1, 2), size=(1, h, w), p=[SNR, (1 - SNR) / 2., (1 - SNR) / 2.])
            mask = np.repeat(mask, c, axis=0)# 按channel 複製到 與img具有相同的shape
            img_gs[mask == 1] = 255    # 鹽噪
            img_gs[mask == 2] = 0      # 椒噪
            img_gs = img_gs.transpose(2, 1, 0)#圖像旋轉
            img_s = img_gs
            img_c = Image.fromarray(img_gs)
            img_c = ImageTk.PhotoImage(image=img_c)
            lb_img.imgtk = img_c
            lb_img.configure(image=img_c)  
            return img_gs
    
def addgasuss(img, mean=0, sigma=0):
    global img_gs,img_s
    if(en2.get()==""):
        messagebox.showerror("Error", "Please Input Value")  #標題，訊息
    else:
        sigma = float(en2.get())
        if(sigma >1):
            messagebox.showerror("Error", "Please Input Value(<1)")  #標題，訊息
        else:
            btn_ero.config(state="normal")
            btn_dil.config(state="normal")
            btn_med.config(state="normal")
            img = np.array(img/255, dtype=float)    # 將像素點/255建立陣列
            noise = np.random.normal(mean, sigma ** 0.5, img.shape)   # 建立一個與image一樣大的陣列，內容為(mean~sigma^0.5)隨機產生
            out = img + noise     # 陣列內容相+
            if out.min() < 0:       # 若out中有小於0的值則執行
                low_clip = -1.
            else:
                low_clip = 0.
            img_gs = np.clip(out, low_clip, 1.0)   # 強制內容介於low_clip~1.0之間
            img_gs = np.uint8(out*255)
            img_s = img_gs
            img_c = Image.fromarray(img_gs)
            img_c = ImageTk.PhotoImage(image=img_c)
            lb_img.imgtk = img_c
            lb_img.configure(image=img_c)  
            return img_gs

def medianBlur(img, SNR):
    global img_gs,img_s
    if(en3.get()==""):
        messagebox.showerror("Error", "Please Input Value")  #標題，訊息
    else:
        SNR = int(en3.get())
        if(SNR <1 ):
            messagebox.showerror("Error", "Please Input Value(>=1)")  #標題，訊息
        else:
            img = cv2.medianBlur(img_gs,SNR)
            img_s = img
            img = Image.fromarray(img)
            img_c = ImageTk.PhotoImage(image=img)
            lb_img.imgtk = img_c
            lb_img.configure(image=img_c)

def main():
    #Main
    #print(sys.getdefaultencoding()) #目前系統編碼
    win = tk.Tk() #建立視窗
    win.title("Image GUI") #標題
    win.geometry("1400x908") # W*H 起始大小
    win.resizable(False,0) # W*H 禁止縮放
    win.config(background="skyblue") # 顏色(也可打bg)
    win.iconbitmap("py.ico") # ico
    #win.attributes("-topmost",1) # 置頂
    BG = tk.PhotoImage(file="skyblue.png")

    #標題
    title_text=tk.Label(text="Image FinalExport",bg="skyblue")
    title_text.config(font="標楷體 30")
    title_text.pack()

    #圖片
    lb_img = tk.Label(image=BG,width=1024,height=768,bg="skyblue")
    lb_img.pack()

    #Openfile
    btn_op = tk.Button(width=10,height=1,text="OpenFile",bg="gray")
    btn_op.config(command=open_file) # 呼叫funtion
    btn_op.place(x=255,y=820)

    #Savefile
    en4=tk.Entry(win,width=10)#輸入
    en4.place(x=55,y=815)
    btn_sv = tk.Button(width=10,height=1,text="SaveFile",bg="gray",state="disabled")
    btn_sv.config(command=save_file) # 呼叫funtion
    btn_sv.place(x=55,y=840)

    #還原鍵
    btn_re = tk.Button(width=10,height=1,text="Recover",bg="gray",state="disabled")
    btn_re.config(command=recover) # 呼叫funtion
    btn_re.place(x=405,y=820)

    #灰階化
    btn_gray = tk.Button(width=10,height=1,text="Gray",bg="gray",state="disabled")
    btn_gray.config(command=lambda :img_gray(img)) # lambda可加入引數
    btn_gray.place(x=555,y=820)

    #二值化
    btn_thresh = tk.Button(width=10,height=1,text="Binary",bg="gray",state="disabled")
    btn_thresh.config(command=lambda :img_thresh(img_g))
    btn_thresh.place(x=705,y=820)

    #侵蝕
    btn_ero = tk.Button(width=10,height=1,text="Eroded",bg="gray",state="disabled")
    btn_ero.config(command = lambda: eroded(img_gs))
    btn_ero.place(x=855,y=820)

    #膨脹
    btn_dil = tk.Button(width=10,height=1,text="Dilated",bg="gray",state="disabled")
    btn_dil.config(command = lambda:dilated(img_gs))
    btn_dil.place(x=1005,y=820)

    #Sobel
    btn_sob = tk.Button(width=10,height=1,text="Sobel",bg="gray",state="disabled")
    btn_sob.config(command = lambda:sobel(img_g))
    btn_sob.place(x=1155,y=820)

    #椒鹽雜訊
    en1=tk.Entry(win,width=8)#輸入
    en1.place(x=410,y=853)
    btn_salt = tk.Button(width=10,height=1,text="Salt",bg="gray",state="disabled")
    btn_salt.config(command=lambda :addsalt(img.transpose(2, 1, 0),0.9))
    btn_salt.place(x=485,y=850)

    #高斯雜訊
    en2=tk.Entry(win,width=8)#輸入
    en2.place(x=630,y=854)
    btn_gasuss = tk.Button(width=10,height=1,text="Gasuss",bg="gray",state="disabled")
    btn_gasuss.config(command=lambda :addgasuss(img,0,0.9))
    btn_gasuss.place(x=705,y=850)

    #中值濾波
    en3=tk.Entry(win,width=8)#輸入
    en3.place(x=860,y=855)
    btn_med = tk.Button(width=10,height=1,text="MedianBlur",bg="gray",state="disabled")
    btn_med.config(command=lambda :medianBlur(img_gs,0))
    btn_med.place(x=935,y=850)

    #關閉視窗
    btn_quit = tk.Button(width=10,height=1,text="Quit",command=Quit)
    btn_quit.pack(side="bottom")

    """
    佈局方式pack grid place
    grid(row=0,column=0) #網格佈局位置
    place(x=,y=) #座標位置 anchor=方位(N，S，W，E)改變原點位置
    """
    win.mainloop() #常駐

if __name__ == "__main__":
    main()