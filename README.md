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

code how to collect patterns
import torch
import time
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch.utils.data import Subset


def extract_precise_patterns(dataset):
    """
    Precision pattern extraction with:
    - Full floating-point precision preservation
    - Strict non-zero pattern validation
    - Exact output format: [sum1-8, x, y, data_num, label]
    """
    start_time = time.perf_counter()
    patterns = []

    for data_num, (image, label) in enumerate(dataset):
        image = image.squeeze()
        padded = torch.nn.functional.pad(image, (1, 1, 1, 1), 'constant', 0)

        for i in range(1, 29):
            for j in range(1, 29):
                if padded[i, j] > 0:
                    sums = []
                    for di, dj in [(-1, -1), (-1, 0), (-1, 1),
                                   (0, -1), (0, 1),
                                   (1, -1), (1, 0), (1, 1)]:
                        # 3x3 sum with precise floating-point accumulation
                        window = padded[i + di - 1:i + di + 2, j + dj - 1:j + dj + 2]
                        sums.append(torch.sum(window).item())

                    if any(s != 0 for s in sums):
                        # Format: [8 sums, x, y, data_num, label]
                        pattern = sums + [i - 1, j - 1, data_num, label]
                        patterns.append(pattern)

    return patterns, time.perf_counter() - start_time


if __name__ == "__main__":
    print("Running precision extraction...")
    mnist = MNIST(root='./data', train=True, download=True, transform=ToTensor())

    # Create a subset of first 10 samples for verification
    test_data = Subset(mnist, range(10))
    patterns, time_elapsed = extract_precise_patterns(test_data)

    print(f"\nExtracted {len(patterns)} valid patterns in {time_elapsed:.2f}s")

    if patterns:
        print("\nFirst valid pattern (exact format):")
        print(patterns[0])

        print("\nStructure verification:")
        print(f"Type: {type(patterns[0])} (length: {len(patterns[0])})")
        print(f"Sum values (8): {patterns[0][:8]}")
        print(f"Coordinates (x,y): {patterns[0][8:10]}")
        print(f"Metadata (data_num, label): {patterns[0][10:]}")

        # Save sample output for verification
        with open("pattern_sample.txt", "w") as f:
            f.write(str(patterns[0]))
        print("\nSample pattern saved to 'pattern_sample.txt'")


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

