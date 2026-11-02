OralScan Classifier Dataset
===========================

Classes:
1. cancer
2. leukoplakia
3. leukoplakia_like
4. normal

Split policy:
Original YOLO train/valid/test splits are preserved.

Each sample contains:
- *_rgb.jpg  : RGB classifier crop
- *_mask.png : paired binary lesion mask

Disease classes:
- RGB crop is generated from the YOLO lesion region.
- Mask contains the predicted lesion.

Normal class:
- RGB crop is generated from processing_area.
- Mask is completely black because no lesion is expected.

Image size:
224 x 224

YOLO model:
/content/best_oralscan_yolo26m_seg.pt

YOLO classes:
0 = processing_area
1 = lesion

Important:
Do not randomly re-split individual generated files.
Use the existing train/valid/test directories to avoid leakage.
