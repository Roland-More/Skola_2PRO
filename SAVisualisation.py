import tkinter as tk
from tkinter import Canvas
import random


def bubbleSort(canvas, width, height, array):
    for i in range(len(array) - 1):
        swapped = False
        for j in range(len(array) - i - 1):
            # Drawing call
            drawArray(canvas, width, height, array, j)
            root.update() # force UI update

            if array[j] > array[j+1]:
                array[j], array[j+1] = array[j+1], array[j]
                swapped = True

            # Drawing call
            drawArray(canvas, width, height, array, j)
            root.update() # force UI update

        if not swapped:
            return

def randomizeArray(canvas, width, height, array):
    random.shuffle(array)

    drawArray(canvas, width, height, array, -2)

def drawArray(canvas, width, height, array, position):
    size = len(array)
    spacing = width / size # Space inbetween columns
    step = height / max(array) # The height of integer 1 in the canvas

    canvas.delete("all")
    for i in range(size):
        color = "PaleGreen2" if i+1 > position >= i-1 else "green2"
        canvas.create_rectangle(i * spacing, height, i * spacing + spacing, height - array[i] * step, fill=color, outline=color)


width1 = 800
height1 = 450

sorting_array = [random.randint(0, 100) for _ in range(100)]

root = tk.Tk()
root.geometry("800x700")

label1 = tk.Label(root, text="Bubble sort", font=("Arial", 72))
label1.grid(row=0, column=0)

canvas1 = tk.Canvas(root, width=width1, height=height1, bg="white")
canvas1.grid(row=1, column=0)

drawArray(canvas1, width1, height1, sorting_array, -2)

buttons1 = tk.Button(root, text="Start sorting", font=("Arial", 24), command=lambda: bubbleSort(canvas1, width1, height1, sorting_array))
buttons1.grid(row=2, column=0)

bubbler1 = tk.Button(root, text="Randomize", font=("Arial", 24), command=lambda: randomizeArray(canvas1, width1, height1, sorting_array))
bubbler1.grid(row=3, column=0)

root.mainloop()
