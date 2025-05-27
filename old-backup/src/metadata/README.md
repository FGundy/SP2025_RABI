# Metadata Processing
This directory contains scripts for metadata extraction and processing.


YOU HAVE TO MOUNT THE SD CARD TO YOUR WSL /mnt/
First, check if D: is listed as a mounted drive:
bash
Copy
Edit
df -h | grep /mnt/d
If nothing shows up, try manually mounting it:

bash
Copy
Edit
sudo mkdir -p /mnt/d
sudo mount -t drvfs D: /mnt/d
Now, check if your files are accessible:

bash
Copy
Edit
ls /mnt/d/DCIM


