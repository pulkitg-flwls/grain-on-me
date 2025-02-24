import cv2
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector, Button

from argparse import ArgumentParser
def parse_args():
    parser = ArgumentParser(description="Load image dataset")
    parser.add_argument("--dir1",type=str,default="")
    parser.add_argument("--dir2",type=str,default="")
    parser.add_argument("--output_dir",type=str,default="")
    parser.add_argument("--face_detect",action='store_true')
    args = parser.parse_args()
    return args

# Load image
image_path = "your_image.jpg"  # Change this to your image path
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB

# Global variable to store box coordinates
box_coords = []

def onselect(eclick, erelease):
    """Capture the coordinates when the box is drawn."""
    global box_coords
    x1, y1 = int(eclick.xdata), int(eclick.ydata)  # Top-left corner
    x2, y2 = int(erelease.xdata), int(erelease.ydata)  # Bottom-right corner
    box_coords = [(x1, y1), (x2, y2)]

def confirm(event):
    """Print the coordinates only when the button is pressed."""
    if box_coords:
        print(f"Final Box Coordinates: {box_coords}")
        plt.close()  # Close the figure after confirmation

if __name__=="__main__":
    args = parse_args()
    
    # Create figure and axis
    fig, ax = plt.subplots()
    ax.imshow(image)
    ax.set_title("Drag to create a bounding box, then press Confirm")

    # Create Rectangle Selector
    rectangle_selector = RectangleSelector(ax, onselect, drawtype='box', useblit=True, interactive=True)

    # Add Confirm Button
    button_ax = plt.axes([0.8, 0.02, 0.15, 0.05])  # x, y, width, height
    confirm_button = Button(button_ax, "Confirm")
    confirm_button.on_clicked(confirm)

    plt.show()