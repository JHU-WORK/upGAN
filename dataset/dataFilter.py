import os
import cv2
import matplotlib.pyplot as plt
import shutil

pathToData = "C:\\Users\\neody\\Downloads\\img_celeba\\img_celeba"
pathOut = "C:\\Users\\neody\\Downloads\\img_celeba\\filtered"

total_pixel_counts = []
mapping = open("dataMapping.txt", "w")
imagesCopied = 0

for image_file in os.listdir(pathToData):
    image_path = os.path.join(pathToData, image_file)
    
    image = cv2.imread(image_path)
    count = int(image.size / 3)
    
    if count < 2000000:
        total_pixel_counts.append(count)
    if len(total_pixel_counts) % 10000 == 0:
        print(len(total_pixel_counts))
    
    if count > 2000000 or image.shape[0] < 800 or image.shape[1] < 800:  #Too large or too small
        continue
    
    #These are now good images
    outName = "i" + str(imagesCopied).zfill(6) + ".jpg"
    shutil.copy(image_path, os.path.join(pathOut, outName))
    mapping.write(outName + " " + image_file + "\n")
    imagesCopied += 1

mapping.close()

counts, bins, _ = plt.hist(total_pixel_counts, bins=200, color='blue', edgecolor='black')
    
# Add labels and title
plt.xlabel('Total Pixel Count')
plt.ylabel('Frequency')
plt.title('Histogram of Total Pixel Count in Images')

# Save or show the histogram
plt.savefig('imageSizeDistribution.png')
plt.show()