# SCRPT# Sudden Change Rotary Pattern Tracker (SCRPT)  
*A high-performance alternative to Convolutional Neural Networks (CNNs) for pattern recognition.*  

## Features  
- ✅ **Fully Transparent** – Every validation can be traced back to a specific pattern type.  
- 🚀 **Higher Accuracy** – Outperforms traditional CNNs in benchmark tests.  
- 🚀 **Faster Processing** – Efficient background management reduces computational overhead.
-  ✨ Key Features
🎯 Background Management
Enhanced Learning & Validation: Manages backgrounds to speed up both the learning process and the validation process, significantly improving SCRPT's efficiency.
Intelligent Pattern Separation:
Stores background patterns separately from useful patterns during data collection.
Uses a simple conditional rule:
python
IF PATTERN == PATTERN[-1]: STORE AS BACKGROUND_PATTERN  
ELSE: STORE AS USEFUL_PATTERN 

Optimized Data Processing:
This separation allows the model to cluster, clean, and optimize useful patterns, which typically represent ~25% of the total data.

🔄 Pattern Collection
SCRPT employs an advanced double-rotation method to efficiently extract patterns from datasets:

Core Collection Process:

For each non-zero element identified as a pattern center, SCRPT:

Captures the 8 surrounding elements (primary rotation)

For each of these 8 elements, captures their 8 surrounding elements (secondary rotation)

Sums these secondary rotation elements

Data Storage:
Each collected pattern is stored in an array containing:

Coordinates of the pattern center

Data number (identifier)

Associated label

The 8 summed values from secondary rotations

MNIST Dataset Example:

python
if pixel_value != 0:  # Identify pattern center
    store_pattern(
        center_coordinates,
        data_number,
        label,
        [sum(secondary_rotation_1), ..., sum(secondary_rotation_8)]
    )
Key Advantages:

Efficient spatial pattern extraction through double-rotation

Comprehensive data preservation (coordinates, identifiers, and processed values)

Optimized for image datasets like MNIST
## Installation  
```bash
git clone https://github.com/yourusername/SCRPT.git  
cd SCRPT  
pip install -r requirements.txt  # (if applicable)
## License  
**Open-Source License**:
## License  
This project is **dual-licensed**:  
1. **AGPL-3.0** (default): You must share source code if used in a networked service.  
2. **Commercial**: Bypasses AGPL requirements. Contact **[sales@example.com]** for pricing.  
Copyright (C) 2024 [Abdelouahab Mahtali].  
This program is free software: you can redistribute it and/or modify it under the terms of the **GNU General Public License (AGPL-3.0)** as published by the Free Software Foundation.  
See the full terms at [LICENSE](LICENSE) or [https://www.gnu.org/licenses/gpl-3.0.html](https://www.gnu.org/licenses/gpl-3.0.html).  

**Commercial License**:  
If you need to use this software **without complying with the AGPL** (e.g., for proprietary modifications, SaaS, or closed-source distribution), a commercial license is available.  
- 📧 Contact: **mahtali@hotmail.com.com**  
- 💰 Pricing: **Starting at $[50000]/year** (scales with usage).  

