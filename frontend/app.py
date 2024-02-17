import tkinter as tk
from tkinter import messagebox
import numpy
from implementation.neuralNetwork import neuralNetwork

class CanvasApp:
    def __init__(self, root):
        self.root = root
        self.canvas = tk.Canvas(root, width=280, height=280, bg="white")
        self.canvas.pack()
        self.image_scale = 10 # Scaling due to difficulties while drawing this shit
        self.max_pixel_value = 255
        
        # Initialization of drawing area
        self.drawing_area = [[0 for _ in range(28)] for _ in range(28)]

        # mouse bindings
        self.canvas.bind("<B1-Motion>", self.paint)

        # reset and guess button
        self.reset_button = tk.Button(root, text="Reset", command=self.reset_canvas)
        self.guess_button = tk.Button(root, text="Guess", command=self.guess_digit)
        self.reset_button.pack()
        self.guess_button.pack()

    def paint(self, event):
        x, y = (event.x // self.image_scale), (event.y // self.image_scale)
        brush_size = 1  # Radius for the intensity effect

        if 0 <= x < 28 and 0 <= y < 28:
            for i in range(-brush_size, brush_size + 1):
                for j in range(-brush_size, brush_size + 1):
                    if 0 <= x + i < 28 and 0 <= y + j < 28:
                        distance = ((i**2 + j**2)**0.5)  # euclidian distance to the center
                        # reduce intensity based on distance
                        intensity_reduction = int(distance * 140)
                        # ensure to stay within bounds
                        current_intensity = self.drawing_area[y + j][x + i]
                        new_intensity = max(0, self.max_pixel_value - intensity_reduction)
                        self.drawing_area[y + j][x + i] = max(current_intensity, new_intensity)
            self.update_canvas()


    
    def update_canvas(self):
        self.canvas.delete("all")
        for y, row in enumerate(self.drawing_area):
            for x, value in enumerate(row):
                intensity = max(0, min(value, 255))
                hex_intensity = '{:02x}'.format(255 - intensity)
                color = f'#{hex_intensity}{hex_intensity}{hex_intensity}'
                self.canvas.create_rectangle(x * self.image_scale, y * self.image_scale,
                                            (x + 1) * self.image_scale, (y + 1) * self.image_scale,
                                            fill=color, outline=color)



    def reset_canvas(self):
        self.drawing_area = [[0 for _ in range(28)] for _ in range(28)]
        self.update_canvas()

    def guess_digit(self):
        digit = [value for row in self.drawing_area for value in row]
        guess = numpy.argmax(n.query((numpy.asfarray(digit) / 255.0 * 0.99) + 0.01))
        messagebox.showinfo("Guess", guess)
        self.reset_canvas()

if __name__ == "__main__":
    n = neuralNetwork(784, 200, 10, 0.1)
    n.load_weights()
    root = tk.Tk()
    app = CanvasApp(root)
    root.title("Let's predict handwritten digits")
    root.mainloop()