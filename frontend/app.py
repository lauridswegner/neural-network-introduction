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

        # check if cursor is in the drawing area before initiating the update of the canvas
        if 0 <= x < 28 and 0 <= y < 28:
            self.drawing_area[y][x] = 1
            self.update_canvas()
    
    def update_canvas(self):
        self.canvas.delete("all")
        for y, row in enumerate(self.drawing_area):
            for x, value in enumerate(row):
                if value:
                    self.canvas.create_rectangle(x * self.image_scale, y * self.image_scale,
                                                 (x + 1) * self.image_scale, (y + 1) * self.image_scale,
                                                 fill='black', outline='black')

    def reset_canvas(self):
        self.drawing_area = [[0 for _ in range(28)] for _ in range(28)]
        self.update_canvas()

    def guess_digit(self):
        digit = [value * 255 if value == 1 else value for row in self.drawing_area for value in row]
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