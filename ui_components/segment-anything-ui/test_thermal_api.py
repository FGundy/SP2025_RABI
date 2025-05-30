import sys
import os
sys.path.append(os.getcwd())

# Test the thermal adapter without GUI
print("🧪 Testing Thermal API Adapter...")

try:
    from segment_anything_ui.thermal_api_adapter import ThermalAPIAdapter
    print("✅ Successfully imported ThermalAPIAdapter")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Let's check what's available...")
    import segment_anything_ui
    print(f"Available in segment_anything_ui: {dir(segment_anything_ui)}")
    sys.exit(1)

import numpy as np

# Create adapter
adapter = ThermalAPIAdapter()
print(f"✅ Adapter created, authenticated: {adapter.is_authenticated}")

# Test image setting with dummy data
test_image = np.zeros((480, 640, 3), dtype=np.uint8)
result = adapter.set_image(test_image)
print(f"✅ Set image result: {result}")

# Test prediction with dummy points
test_points = np.array([[320, 240], [400, 300]])
test_labels = np.array([1, 1])

masks, scores, logits = adapter.predict(
    point_coords=test_points,
    point_labels=test_labels
)

print(f"✅ Prediction result: masks shape = {masks.shape if masks.size > 0 else 'empty'}")
print("🎉 Thermal API integration test complete!")
