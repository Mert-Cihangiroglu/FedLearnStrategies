import torch
import cv2
import numpy as np


class BackdoorStrategy(object):
    def __init__(self, trigger_type, triggerX, triggerY):
        self.trigger_type = trigger_type
        self.triggerX = triggerX
        self.triggerY = triggerY

    def add_square_trigger(self, image):
        pixel_max = torch.max(image) if torch.max(image) > 1 else 1
        image[:, self.triggerY : self.triggerY + 5, self.triggerX : self.triggerX + 5] = pixel_max
        return image

    def add_pattern_trigger(self, image):
        pixel_max = torch.max(image) if torch.max(image) > 1 else 1
        image[:, self.triggerY + 0, self.triggerX + 0] = pixel_max
        image[:, self.triggerY + 1, self.triggerX + 1] = pixel_max
        image[:, self.triggerY - 1, self.triggerX + 1] = pixel_max
        image[:, self.triggerY + 1, self.triggerX - 1] = pixel_max
        return image

    def add_watermark_trigger(self, image, watermark_path="./watermarks/watermark.png"):
        watermark = cv2.imread(watermark_path, cv2.IMREAD_GRAYSCALE)
        watermark = cv2.bitwise_not(watermark)
        watermark = cv2.resize(watermark, dsize=image[0].shape, interpolation=cv2.INTER_CUBIC)
        pixel_max = np.max(watermark)
        watermark = watermark.astype(np.float64) / pixel_max
        pixel_max_dataset = torch.max(image).item() if torch.max(image).item() > 1 else 1
        watermark *= pixel_max_dataset
        max_pixel = max(np.max(watermark), torch.max(image))
        image += watermark
        image[image > max_pixel] = max_pixel
        return image.float()

    def add_backdoor(self, image):
        if self.trigger_type == "pattern":
            image = self.add_pattern_trigger(image)
        elif self.trigger_type == "square":
            image = self.add_square_trigger(image)
        elif self.trigger_type == "watermark":
            image = self.add_watermark_trigger(image)
        return image
