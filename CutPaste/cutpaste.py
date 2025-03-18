import random
import numpy as np
from torchvision import transforms
from PIL import ImageDraw
import time
from PIL import Image, ImageDraw, ImageFilter
class CutPaste(object):

    def __init__(self, transform = True, type = 'binary'):
        '''
        This class creates to different augmentation CutPaste and CutPaste-Scar. Moreover, it returns augmented images
        for binary and 3 way classification

        :arg
        :transform[binary]: - if True use Color Jitter augmentations for patches
        :type[str]: options ['binary' or '3way'] - classification type
        '''
        self.type = type
        if transform:
            self.transform = transforms.ColorJitter(brightness = 0.1,
                                                      contrast = 0.1,
                                                      saturation = 0.1,
                                                      hue = 0.1)
        else:
            self.transform = None

        self.dot_random = random.Random()
        self.white_dot_random = random.Random()
        

    @staticmethod
    def crop_and_paste_patch(image, patch_w, patch_h, transform, rotation=False):
        """
        Crop patch from original image and paste it randomly on the same image.

        :image: [PIL] _ original image
        :patch_w: [int] _ width of the patch
        :patch_h: [int] _ height of the patch
        :transform: [binary] _ if True use Color Jitter augmentation
        :rotation: [binary[ _ if True randomly rotates image from (-45, 45) range

        :return: augmented image
        """

        org_w, org_h = image.size
        mask = None

        patch_left, patch_top = random.randint(0, org_w - patch_w), random.randint(0, org_h - patch_h)
        patch_right, patch_bottom = patch_left + patch_w, patch_top + patch_h
        patch = image.crop((patch_left, patch_top, patch_right, patch_bottom))
        if transform:
            patch= transform(patch)

        if rotation:
            random_rotate = random.uniform(*rotation)
            patch = patch.convert("RGBA").rotate(random_rotate, expand=True)
            mask = patch.split()[-1]

        # new location
        paste_left, paste_top = random.randint(0, org_w - patch_w), random.randint(0, org_h - patch_h)
        aug_image = image.copy()
        aug_image.paste(patch, (paste_left, paste_top), mask=mask)
        return aug_image
    
    

    def cutpaste(self, image, area_ratio = (0.02, 0.15), aspect_ratio = ((0.3, 1) , (1, 3.3))):
        '''
        CutPaste augmentation

        :image: [PIL] - original image
        :area_ratio: [tuple] - range for area ratio for patch
        :aspect_ratio: [tuple] -  range for aspect ratio

        :return: PIL image after CutPaste transformation
        '''

        img_area = image.size[0] * image.size[1]
        patch_area = random.uniform(*area_ratio) * img_area
        patch_aspect = random.choice([random.uniform(*aspect_ratio[0]), random.uniform(*aspect_ratio[1])])
        patch_w  = int(np.sqrt(patch_area*patch_aspect))
        patch_h = int(np.sqrt(patch_area/patch_aspect))
        cutpaste = self.crop_and_paste_patch(image, patch_w, patch_h, self.transform, rotation = False)
        return cutpaste

    def cutpaste_scar(self, image, width = [2,12], length = [20,50], rotation = (-45, 45)): #originnlay 10 , 25
        '''
        Apply a scar on the image using CutPaste augmentation
        :image: [PIL] - original image
        :width: [list] - range for width of patch
        :length: [list] - range for length of patch
        :rotation: [tuple] - range for rotation

        :return: PIL image after CutPaste-Scare transformation
        '''
        patch_w, patch_h = random.randint(*width), random.randint(*length)
        cutpaste_scar = self.crop_and_paste_patch(image, patch_w, patch_h, self.transform, rotation = rotation)
        return cutpaste_scar

    def insertDot(self, image, size=None, color = (0,0,0), transparency=15):
        """
        Apply a black dot of a different size each time to a random location on the image.

        :image: [PIL] - original image
        :size: [int] - diameter of the dot
        :color: [tuple] - RGB color of the dot

        :return: PIL image with a dot applied
        """
        size = random.randint(6, 10) if size is None else size
        aug_image = image.copy()
        draw = ImageDraw.Draw(aug_image, 'RGBA')
        
        # Generate random coordinates for the center of the dot
        org_w, org_h = image.size
        dot_x = self.dot_random.randint(size, org_w - size)
        dot_y = self.dot_random.randint(size, org_h - size)

        # Generate a slight color variation
        color_variation = [max(0, min(255, color[i] + random.randint(-10, 10))) for i in range(3)]

        # Draw concentric circles to create a gradient effect
        for r in range(size // 2, 0, -1):
            # Calculate alpha for gradient effect, decreasing opacity towards the edges
            alpha = int((r / (size / 2)) * transparency)
            gradient_color = (*color_variation, alpha)
            draw.ellipse([(dot_x - r, dot_y - r), (dot_x + r, dot_y + r)], fill=gradient_color)

        # Final blur for a soft blend
        aug_image = aug_image.filter(ImageFilter.GaussianBlur(radius=0.5))

        return aug_image
        
        
    def whiteDot_blackCircle(self, image, size = None , color=(245, 254, 220), border_color=(0, 0, 0), transparency=15):
        """
        Apply a dot with a different size each time to a random location on the image. 
        The dot has a white center and a black border.

        :image: [PIL] - original image
        :size: [int] - diameter of the dot
        :color: [tuple] - RGB color of the inner dot (default white)
        :border_color: [tuple] - RGB color of the outer border (default black)

        :return: PIL image with a bordered dot applied
        """
        if size is None:
            size = random.randint(6, 12)
        
        aug_image = image.copy()
        draw = ImageDraw.Draw(aug_image, 'RGBA')
        
        # Generate random coordinates for the center of the dot
        org_w, org_h = image.size
        dot_x = self.white_dot_random.randint(size, org_w - size)
        dot_y = self.white_dot_random.randint(size, org_h - size)

        # Outer circle (black border) with a gradient for seamless blending
        outer_radius = size // 2
        for r in range(outer_radius, 0, -1):
            # Only the black border has a gradient
            alpha = int((r / outer_radius) * transparency)
            gradient_border_color = (*border_color, alpha)
            draw.ellipse([(dot_x - r, dot_y - r), (dot_x + r, dot_y + r)], fill=gradient_border_color)

        # Inner circle (white dot) with solid color and transparency
        inner_radius = int(size * 0.7) // 2
        inner_left_up = (dot_x - inner_radius, dot_y - inner_radius)
        inner_right_down = (dot_x + inner_radius, dot_y + inner_radius)
        #inner_color = (*color, transparency)  # Add transparency to the inner color
        draw.ellipse([inner_left_up, inner_right_down], fill=color)  # Solid white dot in the center with transparency

        # Apply a very slight blur only to the outer edge for seamless integration
        aug_image = aug_image.filter(ImageFilter.GaussianBlur(radius=0.3))

        return aug_image
    
    def cutpaste_arc_scar(self, image, arc_radius_range=(35, 90), thickness=5, color=(0, 0, 0), transparency=15):
        """
        Add a black scar that looks like a 1/6th of a circular arc on the image. 

        :image: [PIL] - original image
        :arc_radius_range: [tuple] - range of radius for the arc scar
        :thickness: [int] - thickness of the arc
        :color: [tuple] - RGB color of the scar (default black)
        :transparency: [int] - transparency level for the scar

        :return: PIL image with the arc scar applied
        """
        # Copy the image to avoid modifying the original
        aug_image = image.copy()
        draw = ImageDraw.Draw(aug_image, 'RGBA')
        
        # Generate random coordinates for the center of the arc
        org_w, org_h = image.size
        arc_radius = random.randint(*arc_radius_range)
        center_x = random.randint(arc_radius, org_w - arc_radius)
        center_y = random.randint(arc_radius, org_h - arc_radius)

        # Define start and end angle for 1/6th of a circle (60 degrees)
        start_angle = random.randint(0, 360)  # Random start angle for variety
        end_angle = start_angle + 60  # 60 degrees for 1/6th of the circle
        
        # Draw the arc with the specified thickness
        for t in range(thickness):
            alpha = int((1 - t / thickness) * transparency)  # Vary transparency for gradient effect
            gradient_color = (*color, alpha)
            draw.arc(
                [(center_x - arc_radius + t, center_y - arc_radius + t),
                (center_x + arc_radius - t, center_y + arc_radius - t)],
                start=start_angle, end=end_angle, fill=gradient_color
            )
        aug_image = aug_image.filter(ImageFilter.GaussianBlur(radius=0.3))
        return aug_image


    def __call__(self, image):
        
        if self.type == 'binary':
            aug = random.choice([self.cutpaste, self.cutpaste_scar])
            return image, aug(image)

        elif self.type == '3way':
            cutpaste = self.cutpaste(image),
            scar = self.cutpaste_scar(image)
            return image, cutpaste, scar
        
        elif self.type == 'dot': #this is the used case in our experiments
            
            Cases=[1,2,3,4,5,6,7]
            #apply a case randomly
            
            case=random.choice(Cases)
            
            if case==1: # black > white > scar
                #print('case 1')
                anomaly_img = self.insertDot(image=image)
                anomaly_img_withdot = self.whiteDot_blackCircle(image=anomaly_img)
                final=self.cutpaste_scar(anomaly_img_withdot)
                
            elif case==2: # white > black
                #print('case 2')
                anomaly_img_withdot = self.whiteDot_blackCircle(image=image)
                final = self.insertDot(image=anomaly_img_withdot)
                
            elif case==3: # white > white
                #print('case 3')
                anomaly_img_withdot = self.whiteDot_blackCircle(image=image)
                final = self.whiteDot_blackCircle(image=anomaly_img_withdot)
                
            elif case==4: # white > scar
                #print('case 4')
                anomaly_img_withdot = self.whiteDot_blackCircle(image=image)
                final=self.cutpaste_scar(anomaly_img_withdot)
            
            elif case==5: # black > scar
                #print('case 5')
                anomaly_img = self.insertDot(image=image)
                final=self.cutpaste_scar(anomaly_img)
            
            elif case==6: # scar > scar
                #print('case 6') 
                scar_img=self.cutpaste_scar(image)
                final=self.cutpaste_scar(scar_img)
                
            elif case==7: # arc scar
                #print('case 7') 
                final = self.cutpaste_arc_scar(image=image)
                
            else:
                print('Error')
            
            return image, final
