"""
Quick validation script for license plate detection model
"""

from ultralytics import YOLO

# Load model
model = YOLO('weights/best.pt')

# Validate on test dataset
print("=" * 60)
print("VALIDATING LICENSE PLATE DETECTION MODEL")
print("=" * 60)
print(f"Model: weights/best.pt")
print(f"Device: CPU")
print("=" * 60)

results = model.val(
    data='data/dataset/data.yaml',
    device='cpu',
    conf=0.25,
    iou=0.45,
    plots=True
)

print("\n" + "=" * 60)
print("RESULTS")
print("=" * 60)
print(f"📊 mAP50: {results.box.map50:.4f} ({results.box.map50*100:.1f}%)")
print(f"📊 mAP50-95: {results.box.map:.4f}")
print(f"📊 Precision: {results.box.mp:.4f}")
print(f"📊 Recall: {results.box.mr:.4f}")
print("=" * 60)

# Classification
if results.box.map50 > 0.8:
    print("🏆 S-Tier Result! (>0.8 mAP)")
elif results.box.map50 > 0.6:
    print("✅ Good Result! (>0.6 mAP)")
elif results.box.map50 > 0.4:
    print("⚠️ Baseline Result (>0.4 mAP)")
else:
    print("❌ Poor Result (<0.4 mAP)")