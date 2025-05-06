import tkinter as tk
from tkinter import Canvas, colorchooser, messagebox
import threading
import random
import time
import math
import json


def bubbleSort(canvas, width, height, array, delay):
    global visualization_active

    for i in range(len(array) - 1):
        swapped = False
        for j in range(len(array) - i - 1):
            if not visualization_active:
                return
            # Comparison visualization
            drawArray(canvas, width, height, array, [j, j+1])
            time.sleep(delay/2)
                
            if array[j] > array[j+1]:
                array[j], array[j+1] = array[j+1], array[j]
                swapped = True

                # Swap visualization
                drawArray(canvas, width, height, array, [j, j+1])
                time.sleep(delay)

        if not swapped:
            return

def oddEvenSort(canvas, width, height, array, delay):
    global visualization_active

    n = len(array)
    sorted_ = False

    while not sorted_:
        sorted_ = True

        for i in range(1, n - 1, 2):
            if not visualization_active:
                return
            # Comparison visualization
            drawArray(canvas, width, height, array, [i, i+1])
            time.sleep(delay/2)
            
            if array[i] > array[i + 1]:
                array[i], array[i + 1] = array[i + 1], array[i]
                sorted_ = False

                # Swap visualization
                drawArray(canvas, width, height, array, [i, i+1])
                time.sleep(delay)

        for i in range(0, n - 1, 2):
            if not visualization_active:
                return
            # Comparison visualization
            drawArray(canvas, width, height, array, [i, i+1])
            time.sleep(delay/2)
            
            if array[i] > array[i + 1]:
                array[i], array[i + 1] = array[i + 1], array[i]
                sorted_ = False

                # Swap visualization
                drawArray(canvas, width, height, array, [i, i+1])
                time.sleep(delay)

def cocktailSort(canvas, width, height, array, delay):
    global visualization_active

    n = len(array)
    start = 0
    end = n - 1
    swapped = True

    while swapped:
        swapped = False

        for i in range(start, end):
            if not visualization_active:
                return
            # Comparison visualization
            drawArray(canvas, width, height, array, [i, i+1])
            time.sleep(delay/2)
            
            if array[i] > array[i + 1]:
                array[i], array[i + 1] = array[i + 1], array[i]
                swapped = True

                # Swap visualization
                drawArray(canvas, width, height, array, [i, i + 1])
                time.sleep(delay)

        if not swapped:
            break

        swapped = False
        end -= 1

        for i in range(end - 1, start - 1, -1):
            if not visualization_active:
                return
            # Comparison visualization
            drawArray(canvas, width, height, array, [i, i+1])
            time.sleep(delay/2)
            
            if array[i] > array[i + 1]:
                array[i], array[i + 1] = array[i + 1], array[i]
                swapped = True

                # Swap visualization
                drawArray(canvas, width, height, array, [i, i + 1])
                time.sleep(delay)

        start += 1

def pancakeSort(canvas, width, height, array, delay):
    n = len(array)
    
    # Perform pancake sort
    for curr_size in range(n, 1, -1):
        if not visualization_active:
                return
        # Find the index of the maximum element in arr[0...curr_size-1]
        max_idx = findMax(array, curr_size, canvas, width, height, delay)
        
        if max_idx != curr_size - 1:
            # If the max element is not already in the right place, flip the maximum element
            if max_idx != 0:
                # Flip the maximum element to the beginning
                flip(array, max_idx, canvas, width, height, delay)
            
            # Flip the max element to its correct position
            flip(array, curr_size - 1, canvas, width, height, delay)

        # Visualize after every full pass
        drawArray(canvas, width, height, array, [curr_size-1])
        time.sleep(delay)

def flip(arr, i, canvas, width, height, delay):
    global visualization_active
    # Visualize the flip process step-by-step
    start = 0
    while start < i:
        if not visualization_active:
                return

        arr[start], arr[i] = arr[i], arr[start]  # Swap the elements
        # Visualize the array after each individual swap
        drawArray(canvas, width, height, arr, [start, i])
        time.sleep(delay)
        start += 1
        i -= 1
    
    # Final visualization of the flipped subarray
    drawArray(canvas, width, height, arr, [i, start])
    time.sleep(delay)

def findMax(array, end, canvas, width, height, delay):
    global visualization_active

    max_idx = 0
    for i in range(1, end):
        if not visualization_active:
                return

        if array[i] > array[max_idx]:
            max_idx = i
        # Visualize the comparison
        drawArray(canvas, width, height, array, [i, max_idx])
        time.sleep(delay)
    
    # Final visualization of the found max element
    drawArray(canvas, width, height, array, [max_idx])
    time.sleep(delay)
    return max_idx

def combSort(canvas, width, height, array, delay):
    global visualization_active

    def getNextGap(gap):
        gap = int(gap * 10 // 13)
        return max(1, gap)

    n = len(array)
    gap = n
    swapped = True

    while gap != 1 or swapped:
        gap = getNextGap(gap)
        swapped = False

        for i in range(n - gap):
            if not visualization_active:
                return
            # Comparison visualization
            drawArray(canvas, width, height, array, [i, i + gap])
            time.sleep(delay/2)
            
            if array[i] > array[i + gap]:
                array[i], array[i + gap] = array[i + gap], array[i]
                swapped = True

                # Swap visualization
                drawArray(canvas, width, height, array, [i, i + gap])
                time.sleep(delay)

def selectionSort(canvas, width, height, array, delay):
    global visualization_active

    for i in range(len(array)):
        min_pos = i
        for j in range(i, len(array)):
            if not visualization_active:
                return
            # Comparison visualization
            drawArray(canvas, width, height, array, [j, min_pos])
            time.sleep(delay/2)

            if array[j] < array[min_pos]:
                min_pos = j
                
                # Position update visualization
                drawArray(canvas, width, height, array, [j, min_pos])
                time.sleep(delay/2)
            
        # Swap visualization
        array[i], array[min_pos] = array[min_pos], array[i]
        drawArray(canvas, width, height, array, [i, min_pos])
        time.sleep(delay)

def gnomeSort(canvas, width, height, array, delay):
    global visualization_active

    index = 0
    size = len(array)

    while index < size:
        if not visualization_active:
                return

        if index == 0:
            index += 1
            
        # Comparison visualization (when possible)
        if index > 0:
            drawArray(canvas, width, height, array, [index, index - 1])
            time.sleep(delay/2)
            
        if index == 0 or array[index] >= array[index - 1]:
            index += 1
        else:
            array[index], array[index - 1] = array[index - 1], array[index]
            
            # Swap visualization
            drawArray(canvas, width, height, array, [index, index - 1])
            time.sleep(delay)
            
            index -= 1

def insertionSort(canvas, width, height, array, delay):
    global visualization_active

    for i in range(1, len(array)):
        if not visualization_active:
            return

        key = array[i]
        
        # Key selection visualization
        drawArray(canvas, width, height, array, [i])
        time.sleep(delay/2)

        j = i - 1
        while j >= 0:
            if not visualization_active:
                return
            # Comparison visualization
            drawArray(canvas, width, height, array, [j, j+1])
            time.sleep(delay/2)
            
            if array[j] > key:
                array[j + 1] = array[j]

                # Assignment visualization
                drawArray(canvas, width, height, array, [j, j+1])
                time.sleep(delay/2)

                j -= 1
            else:
                break

        array[j + 1] = key

        # Key placement visualization
        drawArray(canvas, width, height, array, [j+1])
        time.sleep(delay)
    
def shellSort(canvas, width, height, array, delay):
    global visualization_active

    n = len(array)
    gap = n // 2

    while gap > 0:
        if not visualization_active:
                return

        for i in range(gap, n):
            if not visualization_active:
                return
            
            temp = array[i]
            j = i
            
            # Combined selection and first comparison visualization
            drawArray(canvas, width, height, array, [i, max(0, j-gap)])
            time.sleep(delay/2)
            
            # Optimize by combining comparison and assignment in one loop
            while j >= gap and array[j - gap] > temp:
                if not visualization_active:
                    return

                array[j] = array[j - gap]
                j -= gap
                
                # Single visualization per shift operation
                if j >= gap:  # Only visualize if we're continuing the loop
                    drawArray(canvas, width, height, array, [j, j-gap])
                    time.sleep(delay/2)
            
            # Only visualize the final placement if it actually moved
            if j != i:
                array[j] = temp
                drawArray(canvas, width, height, array, [j])
                time.sleep(delay)
            
        gap //= 2

def stoogeSortWrapper(canvas, width, height, array, delay):
    stoogeSort(canvas, width, height, array, 0, len(array) - 1, delay)

def stoogeSort(canvas, width, height, array, l, h, delay):
    global visualization_active

    if not visualization_active:
        return

    if l >= h:
        return

    # Comparison visualization
    drawArray(canvas, width, height, array, [l, h])
    time.sleep(delay/2)
    
    if array[l] > array[h]:
        array[l], array[h] = array[h], array[l]

        # Swap visualization
        drawArray(canvas, width, height, array, [l, h])
        time.sleep(delay)

    if h - l + 1 > 2:
        t = (h - l + 1) // 3
        stoogeSort(canvas, width, height, array, l, h - t, delay)
        stoogeSort(canvas, width, height, array, l + t, h, delay)
        stoogeSort(canvas, width, height, array, l, h - t, delay)

def heapSortWrapper(canvas, width, height, array, delay):
    heapSort(canvas, width, height, array, 0, len(array) - 1, delay)

def heapSort(canvas, width, height, array, low, high, delay):
    global visualization_active

    n = high + 1

    for i in range(n // 2 - 1, low - 1, -1):
        heapify(canvas, width, height, array, n, i, delay)

    for i in range(n - 1, low - 1, -1):
        if not visualization_active:
            return
        # Swap visualization
        array[i], array[low] = array[low], array[i]
        drawArray(canvas, width, height, array, [low, i])
        time.sleep(delay)

        heapify(canvas, width, height, array, i, low, delay)

def heapify(canvas, width, height, array, n, i, delay):
    global visualization_active

    if not visualization_active:
        return

    largest = i
    left = 2 * i + 1
    right = 2 * i + 2

    # Left child comparison
    if left < n:
        drawArray(canvas, width, height, array, [largest, left])
        time.sleep(delay/2)
        
        if array[left] > array[largest]:
            largest = left

    # Right child comparison
    if right < n:
        drawArray(canvas, width, height, array, [largest, right])
        time.sleep(delay/2)
        
        if array[right] > array[largest]:
            largest = right

    if largest != i:
        # Swap visualization
        array[i], array[largest] = array[largest], array[i]
        drawArray(canvas, width, height, array, [i, largest])
        time.sleep(delay)

        heapify(canvas, width, height, array, n, largest, delay)

def mergeSortWrapper(canvas, width, height, array, delay):
    mergeSort(canvas, width, height, array, 0, len(array) - 1, delay)

def mergeSort(canvas, width, height, array, l, r, delay):
    global visualization_active

    if not visualization_active:
        return

    if l < r:
        m = (l + r) // 2
        mergeSort(canvas, width, height, array, l, m, delay)
        mergeSort(canvas, width, height, array, m+1, r, delay)
        merge(canvas, width, height, array, l, m, r, delay)

def merge(canvas, width, height, array, l, m, r, delay):
    global visualization_active

    left = array[l:m+1]
    right = array[m+1:r+1]
    i = j = 0
    k = l

    while i < len(left) and j < len(right):
        if not visualization_active:
            return
        # Comparison visualization
        drawArray(canvas, width, height, array, [l+i, m+1+j])
        time.sleep(delay/2)
        
        if left[i] <= right[j]:
            array[k] = left[i]
            i += 1
        else:
            array[k] = right[j]
            j += 1

        # Assignment visualization
        drawArray(canvas, width, height, array, [k])
        time.sleep(delay/2)

        k += 1

    while i < len(left):
        if not visualization_active:
            return

        array[k] = left[i]

        # Assignment visualization
        drawArray(canvas, width, height, array, [k, l+i])
        time.sleep(delay/2)

        i += 1
        k += 1

    while j < len(right):
        if not visualization_active:
            return

        array[k] = right[j]

        # Assignment visualization
        drawArray(canvas, width, height, array, [k, m+1+j])
        time.sleep(delay/2)

        j += 1
        k += 1

def timSort(canvas, width, height, array, delay):
    global visualization_active

    n = len(array)
    min_run = getMinRun(n)

    runs = []
    i = 0
    while i < n:
        if not visualization_active:
            return
        
        run_start = i
        run_end = i + 1

        # Check if descending
        if run_end < n:
            # Comparison visualization
            drawArray(canvas, width, height, array, [run_start, run_end])
            time.sleep(delay/2)
            
            if array[run_end] < array[run_start]:
                while run_end < n:
                    if not visualization_active:
                        return
                    
                    if run_end + 1 < n:
                        # Comparison visualization
                        drawArray(canvas, width, height, array, [run_end, run_end+1])
                        time.sleep(delay/2)
                        
                        if array[run_end+1] >= array[run_end]:
                            break
                    
                    run_end += 1
                    if run_end >= n:
                        break
                
                # Reverse the array portion
                temp = array[run_start:run_end]
                temp.reverse()
                for idx, val in enumerate(temp):
                    if not visualization_active:
                        return
                    
                    array[run_start + idx] = val
                    # Assignment visualization
                    drawArray(canvas, width, height, array, [run_start + idx])
                    time.sleep(delay/2)
            else:
                while run_end < n:
                    if not visualization_active:
                        return

                    if run_end + 1 < n:
                        # Comparison visualization
                        drawArray(canvas, width, height, array, [run_end, run_end+1])
                        time.sleep(delay/2)
                        
                        if array[run_end+1] < array[run_end]:
                            break
                    
                    run_end += 1
                    if run_end >= n:
                        break

        actual_end = min(run_start + min_run, n)
        insertionSortPartial(canvas, width, height, array, run_start, actual_end, delay)

        runs.append((run_start, actual_end))
        i = actual_end

    while len(runs) > 1:
        if not visualization_active:
            return

        new_runs = []
        for i in range(0, len(runs) - 1, 2):
            if not visualization_active:
                return

            if i + 1 < len(runs):
                l, m = runs[i][0], runs[i][1] - 1
                r = runs[i+1][1] - 1
                merge(canvas, width, height, array, l, m, r, delay)
                new_runs.append((l, r + 1))
            else:
                new_runs.append(runs[i])
                
        if len(runs) % 2 == 1:
            new_runs.append(runs[-1])
        runs = new_runs

def getMinRun(n):
    global visualization_active

    r = 0
    while n >= 64:
        if not visualization_active:
            return

        r |= n & 1
        n >>= 1
    return n + r

def insertionSortPartial(canvas, width, height, array, left, right, delay):
    global visualization_active

    for i in range(left + 1, right):
        if not visualization_active:
            return
            
        key = array[i]
        
        # Key selection visualization
        drawArray(canvas, width, height, array, [i])
        time.sleep(delay/2)
        
        j = i - 1
        while j >= left:
            if not visualization_active:
                return
            # Comparison visualization
            drawArray(canvas, width, height, array, [j, j+1])
            time.sleep(delay/2)
            
            if array[j] > key:
                array[j + 1] = array[j]

                # Assignment visualization
                drawArray(canvas, width, height, array, [j, j + 1])
                time.sleep(delay/2)

                j -= 1
            else:
                break
                
        array[j + 1] = key

        # Final key placement visualization
        drawArray(canvas, width, height, array, [j + 1])
        time.sleep(delay/2)

def quickSortWrapper(canvas, width, height, array, delay):
    quickSort(canvas, width, height, array, 0, len(array) - 1, delay)

def quickSort(canvas, width, height, array, low, high, delay):
    global visualization_active

    if not visualization_active:
        return

    if low < high:
        pi = partition(canvas, width, height, array, low, high, delay)
        quickSort(canvas, width, height, array, low, pi - 1, delay)
        quickSort(canvas, width, height, array, pi + 1, high, delay)

def partition(canvas, width, height, array, low, high, delay):
    global visualization_active

    if not visualization_active:
        return

    mid = (low + high) // 2
    pivot_candidates = [(array[low], low), (array[mid], mid), (array[high], high)]
    pivot_candidates.sort(key=lambda x: x[0])
    pivot_index = pivot_candidates[1][1]

    array[pivot_index], array[high] = array[high], array[pivot_index]  # Move pivot to end
    pivot = array[high]

    i = low - 1

    # # Pivot visualization
    drawArray(canvas, width, height, array, [high])
    time.sleep(delay/2)

    for j in range(low, high):
        if not visualization_active:
            return
        # Comparison visualization
        drawArray(canvas, width, height, array, [j, high])
        time.sleep(delay/2)
        
        if array[j] < pivot:
            i += 1
            
            # Swap visualization
            array[i], array[j] = array[j], array[i]
            drawArray(canvas, width, height, array, [i, j])
            time.sleep(delay)

    # Final pivot swap visualization
    array[i + 1], array[high] = array[high], array[i + 1]
    drawArray(canvas, width, height, array, [i + 1, high])
    time.sleep(delay)

    return i + 1

def introSort(canvas, width, height, array, delay):
    global visualization_active

    max_depth = int(math.log2(len(array)) * 2)
    introSortHelper(canvas, width, height, array, 0, len(array) - 1, max_depth, delay)

def introSortHelper(canvas, width, height, array, low, high, max_depth, delay):
    global visualization_active

    if not visualization_active:
        return
    
    if low < high:
        # Depth check visualization
        drawArray(canvas, width, height, array, [low, high])
        time.sleep(delay/2)
        
        if max_depth == 0:
            heapSort(canvas, width, height, array, low, high, delay)
        else:
            pi = partition(canvas, width, height, array, low, high, delay)
            introSortHelper(canvas, width, height, array, low, pi - 1, max_depth - 1, delay)
            introSortHelper(canvas, width, height, array, pi + 1, high, max_depth - 1, delay)
 
def radixSort(canvas, width, height, array, delay):
    global visualization_active

    def countingSort(exp):
        global visualization_active

        if not visualization_active:
            return
        
        n = len(array)
        output = [0] * n
        count = [0] * 10

        # Step 1: Count digit frequencies
        for i in range(n):
            if not visualization_active:
                return
            
            index = (array[i] // exp) % 10
            count[index] += 1

        # Step 2: Compute cumulative count
        for i in range(1, 10):
            count[i] += count[i - 1]

        # Step 3: Build output array from right to left (stable)
        i = n - 1
        while i >= 0:
            if not visualization_active:
                return
            
            index = (array[i] // exp) % 10
            output[count[index] - 1] = array[i]
            count[index] -= 1
            i -= 1

        # Step 4: Copy back to original array with visualization
        for i in range(n):
            if not visualization_active:
                return
            
            array[i] = output[i]
            drawArray(canvas, width, height, array, [i])
            time.sleep(delay/2)

    # Start with least significant digit
    max_val = max(array)
    exp = 1

    # Visualize finding the max
    drawArray(canvas, width, height, array, [array.index(max_val)])
    time.sleep(delay / 2)

    while max_val // exp > 0:
        countingSort(exp)
        exp *= 10


def randomizeArray(canvases, width, height, arrays):
    for i in range(len(arrays)):
        random.shuffle(arrays[i])
        drawArray(canvases[i], width, height, arrays[i], [-2])

def drawArray(canvas, width, height, array, positions):
    def draw():
        global element_color, highlight_color, bgcol, visualization_active

        if not visualization_active:
            return

        size = len(array)
        spacing = width / size # Space inbetween columns
        step = height / max(array) # The height of integer 1 in the canvas

        canvas.delete("all")
        for i in range(size):
            color = highlight_color if i in positions else element_color
            canvas.create_rectangle(i * spacing, height, i * spacing + spacing, height - array[i] * step, fill=color, outline=bgcol)
            
    canvas.after(0, draw)

def allSort(canvases, width, height, arrays, delay, selected_algorithms):
    global algorithms, active_threads

    for i in range(len(selected_algorithms)):
        thread = threading.Thread(target=algorithms[selected_algorithms[i]], args=(canvases[i], width, height, arrays[i], delay))
        thread.daemon = True  # Mark thread as daemon so it exits when main thread exits
        active_threads.append(thread)
        thread.start()

def start_algorithm_thread(algorithm_name, canvas, width, height, array, delay):
    global algorithms, active_threads
    
    thread = threading.Thread(target=algorithms[algorithm_name], args=(canvas, width, height, array, delay))
    thread.daemon = True  # Mark thread as daemon so it exits when main thread exits
    active_threads.append(thread)
    thread.start()

def chooseColor(color_type):
        global bgcol, element_color, highlight_color, bg_color_btn, element_color_btn, highlight_color_btn

        current_color = {
            "bg": bgcol,
            "element": element_color,
            "highlight": highlight_color
        }.get(color_type)

        color = colorchooser.askcolor(color=current_color)[1]
        if color:
            if color_type == "bg":
                bgcol = color
                bg_color_btn.config(bg=color)
            elif color_type == "element":
                element_color = color
                element_color_btn.config(bg=color)
            elif color_type == "highlight":
                highlight_color = color
                highlight_color_btn.config(bg=color)

def saveSettings():
    global settings, array_size_var, delay_var, randomize_var, alg_vars
    global bgcol, element_color, highlight_color

    settings["array_size"] = array_size_var.get()
    settings["delay"] = delay_var.get()
    settings["randomize"] = randomize_var.get()
    settings["bgcol"] = bgcol
    settings["element_color"] = element_color
    settings["highlight_color"] = highlight_color

    for alg_name, var in alg_vars.items():
        settings["alg_vars"][alg_name] = var.get()

    with open("settings.json", "w") as file:
        json.dump(settings, file, indent=4)        

def loadSettings():
    global settings, array_size_var, delay_var, randomize_var, bgcol, element_color, highlight_color, alg_vars
    global bg_color_btn, element_color_btn, highlight_color_btn

    try:
        with open("settings.json", "r") as file:
            settings = json.load(file)

        array_size_var.set(settings["array_size"])
        delay_var.set(settings["delay"])
        randomize_var.set(settings["randomize"])

        bgcol = settings["bgcol"]
        element_color = settings["element_color"]
        highlight_color = settings["highlight_color"]

        # Update GUI
        if bg_color_btn:
            bg_color_btn.config(bg=bgcol)
            element_color_btn.config(bg=element_color)
            highlight_color_btn.config(bg=highlight_color)

        for alg_name, var in settings["alg_vars"].items():
            if alg_name in alg_vars:
                alg_vars[alg_name].set(var)

    except (FileNotFoundError, json.JSONDecodeError):
        pass

def generateVisualizations():
    global alg_vars, array_size_var, delay_var, randomize_var, algorithms, visualization_active
    global bgcol, element_color, highlight_color, active_threads

    # Set up variables for rendering
    selected_algorithms = []

    for key in alg_vars.keys():
        if alg_vars[key].get():
            selected_algorithms.append(key)

    algs = len(selected_algorithms)

    delay = delay_var.get()
    array_size = array_size_var.get()

    # Validation
    if algs < 1:
        messagebox.showerror("Error", "Select a minimum of 1 algorithm")
        return
    if algs > 6:
        messagebox.showerror("Error", "Select a maximum of 6 algorithms")
        return
    if delay <= 0:
        messagebox.showerror("Error", "Delay cant be zero or less")
        return
    if delay > 2:
        messagebox.showerror("Error", "Delay must me at most 2")
        return
    if array_size < 2:
        messagebox.showerror("Error", "Array size must be at least 5")
        return
    if array_size > 200:
        messagebox.showerror("Error", "Array size must be at most 200")
        return
    if array_size < 5:
        messagebox.showinfo("Interesting...", "What are you even comparing at this point?")
    
    # Set up variables for rendering
    randomize = randomize_var.get()

    rows = 1 if algs <= 2 else 2
    cols = 1
    if 2 <= algs <= 4:
        cols = 2
    elif algs >= 5:
        cols = 3

    width = 636
    height = 420 if rows == 1 else 400

    positions = [
        {"r": 0, "c": 0},
        {"r": 0, "c": 2},
        {"r": 3, "c": 0 if algs % 2 == 0 else 1},
        {"r": 3, "c": 2 if algs % 2 == 0 else 3},
        {"r": 0, "c": 4},
        {"r": 3, "c": 4}
    ]

    labels = [0 for _ in range(algs)]
    canvases = [0 for _ in range(algs)]
    buttonses = [0 for _ in range(algs)]
    buttonrs = [0 for _ in range(algs)]

    # Generating arrays
    sorting_array = [array_size - x for x in range(array_size)]
    if randomize:
        random.shuffle(sorting_array)
    arrays = [sorting_array]
    for i in range(algs-1):
        arrays.append(sorting_array.copy())

    # Clean up any existing threads before creating a new window
    active_threads = []

    # Window
    root = tk.Toplevel(settings_root)  # Use Toplevel instead of Tk
    root.title("Sorting Algorithm Visualization")
    root.geometry(f"{cols * 640}x{rows * 540 + (10 if algs > 1 else -30)}")
    
    # Set up the window close handler
    def on_close():
        global active_threads, visualization_active

        active_threads = []
        root.destroy()
        visualization_active = False
    
    root.protocol("WM_DELETE_WINDOW", on_close)

    # Algorithms
    for i in range(algs):
        labels[i] = tk.Label(root, text=selected_algorithms[i], font=("Arial", 18))
        labels[i].grid(row=positions[i]["r"], column=positions[i]["c"], columnspan=2)

        canvases[i] = tk.Canvas(root, width=width, height=height, bg=bgcol)
        canvases[i].grid(row=positions[i]["r"] + 1, column=positions[i]["c"], columnspan=2)

        drawArray(canvases[i], width, height, arrays[i], [-2])

        # For each algorithm button, create thread with daemon=True
        buttonses[i] = tk.Button(root, text="Sort", font=("Arial", 16),
            command=lambda i=i: start_algorithm_thread(selected_algorithms[i], canvases[i], width, height, arrays[i], delay))
        buttonses[i].grid(row=positions[i]["r"] + 2, column=positions[i]["c"])

        buttonrs[i] = tk.Button(root, text="Randomize", font=("Arial", 16), command=lambda i=i: randomizeArray([canvases[i]], width, height, [arrays[i]]))
        buttonrs[i].grid(row=positions[i]["r"] + 2, column=positions[i]["c"] + 1)

    # Buttons for controlling all algorithms
    if algs > 1:
        buttonsall = tk.Button(root, text="Sort All", font=("Arial", 16), command=lambda: allSort(canvases, width, height, arrays, delay, selected_algorithms))
        buttonsall.grid(row=rows*3, column=0, pady=5)

        buttonrall = tk.Button(root, text="Randomize All", font=("Arial", 16), command=lambda: randomizeArray(canvases, width, height, arrays))
        buttonrall.grid(row=rows*3, column=1, pady=5)

    visualization_active = True

# Handle cleanup when the settings window is closed
def on_settings_close():
    global active_threads, visualization_active

    active_threads = []
    settings_root.destroy()
    visualization_active = False


visualization_active = False

algorithms = {
    "Bubble Sort": bubbleSort,
    "Odd-Even Sort": oddEvenSort,
    "Cocktail Sort": cocktailSort,
    "Pancake Sort": pancakeSort,
    "Comb Sort": combSort,
    "Selection Sort": selectionSort,
    "Gnome Sort": gnomeSort,
    "Insertion Sort": insertionSort,
    "Shell Sort": shellSort,
    "Stooge Sort": stoogeSortWrapper,
    "Heap Sort": heapSortWrapper,
    "Merge Sort": mergeSortWrapper,
    "Tim Sort": timSort,
    "Quick Sort": quickSortWrapper,
    "Intro Sort": introSort,
    "Radix Sort": radixSort,
}

# Settings
settings_root = tk.Tk()
settings_root.title("Sorting Algorithm Visualizer - Settings")
settings_root.geometry("660x340")

array_size_var = tk.IntVar(value=50)  # Default value
delay_var = tk.DoubleVar(value=0.05)  # Default value
randomize_var = tk.BooleanVar(value=True)  # Default value

# Default colors
bgcol = "#ffffff"  # White
element_color = "#3498db"  # Blue
highlight_color = "#e74c3c"  # Red

# Create default settings if not loaded
settings = {
    "array_size": 50,
    "delay": 0.02,
    "randomize": True,
    "bgcol": bgcol,
    "element_color": element_color,
    "highlight_color": highlight_color,
    "alg_vars": {}
}

alg_vars = {}
for alg_name in algorithms.keys():
    var = tk.BooleanVar(value=False)
    alg_vars[alg_name] = var
    settings["alg_vars"][alg_name] = False

# Placeholders for loadSettings
bg_color_btn = None
element_color_btn = None
highlight_color_btn = None

# Try to load settings, but use defaults if it fails
loadSettings()

# Keep track of active threads
active_threads = []

settings_frame = tk.LabelFrame(settings_root, text="Visualization Settings", padx=10, pady=10, font=("Arial", 12))
settings_frame.pack(fill=tk.X, padx=10, pady=10)

# Array size
tk.Label(settings_frame, text="Array Size:").grid(row=0, column=0, sticky="w")
array_size_entry = tk.Spinbox(settings_frame, from_=5, to=200, textvariable=array_size_var, width=5)
array_size_entry.grid(row=0, column=1, sticky="w", padx=5)

# Delay
tk.Label(settings_frame, text="Animation Delay (s):").grid(row=0, column=2, sticky="w", padx=(20, 0))
delay_entry = tk.Spinbox(settings_frame, from_=0.01, to=2, increment=0.01, textvariable=delay_var, width=5)
delay_entry.grid(row=0, column=3, sticky="w", padx=5)

# Randomize option
randomize_check = tk.Checkbutton(settings_frame, text="Randomize Arrays", variable=randomize_var)
randomize_check.grid(row=0, column=4, sticky="w", padx=(20, 0))

# Canvas background color
tk.Label(settings_frame, text="Background Color:").grid(row=1, column=0, sticky="w")
bg_color_btn = tk.Button(settings_frame, text="Choose", bg=bgcol, width=8, 
                            command=lambda: chooseColor("bg"))
bg_color_btn.grid(row=1, column=1, sticky="w", padx=5, pady=5)

# Element color
tk.Label(settings_frame, text="Element Color:").grid(row=1, column=2, sticky="w", padx=(20, 0))
element_color_btn = tk.Button(settings_frame, text="Choose", bg=element_color, width=8,
                                command=lambda: chooseColor("element"))
element_color_btn.grid(row=1, column=3, sticky="w", padx=5, pady=5)

# Highlight color
tk.Label(settings_frame, text="Highlight Color:").grid(row=1, column=4, sticky="w", padx=(20, 0))
highlight_color_btn = tk.Button(settings_frame, text="Choose", bg=highlight_color, width=8,
                                command=lambda: chooseColor("highlight"))
highlight_color_btn.grid(row=1, column=5, sticky="w", padx=5, pady=5)

alg_frame = tk.LabelFrame(settings_root, text="Select Algorithms to Compare", padx=10, pady=10, font=("Arial", 12))
alg_frame.pack(fill=tk.X, padx=10, pady=10)

# Create checkboxes for each algorithm
for i, alg_name in enumerate(algorithms.keys()):
    cb = tk.Checkbutton(alg_frame, text=alg_name, variable=alg_vars[alg_name])
    cb.grid(row=i // 4, column=i % 4, sticky="w", padx=5, pady=2)

generate_btn = tk.Button(settings_root, text="Generate Visualizations", 
                    command=generateVisualizations, font=("Arial", 12))
generate_btn.pack(side=tk.LEFT, padx=5, pady=5)

save_settings_btn = tk.Button(settings_root, text="Save Settings", 
                    command=saveSettings, font=("Arial", 12))
save_settings_btn.pack(side=tk.LEFT, padx=5, pady=5)

load_settings_btn = tk.Button(settings_root, text="Load Settings", 
                    command=loadSettings, font=("Arial", 12))
load_settings_btn.pack(side=tk.LEFT, padx=5, pady=5)


settings_root.protocol("WM_DELETE_WINDOW", on_settings_close)

settings_root.mainloop()
