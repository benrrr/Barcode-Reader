# Barcode Reader

OpenCV program designed to find, extract, correct orientation and read a barcode from an image.

This was completed in a team of 3 inlcuding myself, Robert Vaughan and Mohamad Zabad. Much of my research and development process was documented [here](https://2284benryan.blogspot.com/ "https://2284benryan.blogspot.com/").

## Sample Images

### Input Image
![Imgur](https://i.imgur.com/KhxaADM.jpg "Original")

### Barcode Found
![Imgur](https://i.imgur.com/dDhI7IP.jpg "Found")

### Cleaned
![Imgur](https://i.imgur.com/pBTqqIc.jpg "Cleaned")

### ROI Cropped and Rotated
![Imgur](https://i.imgur.com/9r40wp4.jpg "ROI")

### Output:
From the previou state the image is read as binary, then the standard UPC-A barcode rules are automatically applied to extract decimal numbers.

Binary: 10101110110001101011000101011110111101001001101010111001010010001001110111010010111001000010101

Decimal: 705632085943