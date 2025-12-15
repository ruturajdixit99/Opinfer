# Using Opinfer with Vision-Language Models (VLMs)

This guide shows how to use the opinfer package with Vision-Language Models for efficient video inference.

---

## 🎯 Quick Start with VLMs

### Example 1: OWL-ViT (Already Supported)

OWL-ViT is a Vision-Language Model for open-vocabulary object detection. It's already fully supported:

```python
from opinfer import OptimizedInference

# Initialize with OWL-ViT detector
inf = OptimizedInference(
    model_name="owlvit-base",  # or "owlvit-large"
    model_type="detector",
    technique="motion_gating",  # Motion gating works with detectors
    auto_optimize=True,
)

# Process video
results = inf.process_video_file("video.mp4", max_frames=200)

print(f"Skip rate: {results['stats']['skip_rate_pct']:.1f}%")
print(f"Effective FPS: {results['stats']['effective_fps']:.2f}")
```

**Note**: OWL-ViT uses text queries for detection. The package automatically uses default queries, but you can customize them.

---

## 🔧 Using Custom VLMs

### Method 1: Wrap Your VLM Model

If you have a custom VLM, wrap it to work with opinfer:

```python
from opinfer import AdaptiveMotionGater, ModelLoader
import torch
import torch.nn as nn

# Your custom VLM model
class YourVLM(nn.Module):
    def __init__(self):
        super().__init__()
        # Your VLM architecture
        pass
    
    def forward(self, images, text_inputs=None):
        # Your VLM forward pass
        return outputs

# Load or create your VLM
vlm_model = YourVLM().to("cuda")
vlm_model.eval()

# Use with motion gating
gater = AdaptiveMotionGater(
    model=vlm_model,
    device="cuda",
    auto_optimize=True,
)

# Process frames
frames = load_video_frames("video.mp4")
results = gater.process_video(frames, analyze_first=True)
```

### Method 2: Custom Processor Function

For VLMs that need special preprocessing:

```python
from opinfer.core import MotionGatedInference
from opinfer.detectors import VideoCharacteristicDetector
from opinfer.optimizer import ParameterOptimizer

# Your VLM
vlm_model = YourVLM().to("cuda")

# Custom preprocessing function
def preprocess_vlm_frame(frame_bgr, text_query):
    # Your VLM-specific preprocessing
    # Convert BGR to RGB, resize, tokenize text, etc.
    image_tensor = preprocess_image(frame_bgr)
    text_tensor = tokenize_text(text_query)
    return image_tensor, text_tensor

# Create motion-gated engine
engine = MotionGatedInference(
    model=vlm_model,
    device="cuda",
    motion_threshold=4.0,
    min_frames_between_calls=2,
)

# Process frames with custom logic
for frame in frames:
    # Your custom preprocessing
    img, text = preprocess_vlm_frame(frame, "your query")
    
    # Motion gating decision
    motion_score = engine.compute_motion_score(frame)
    if engine.should_call_model(motion_score):
        with torch.no_grad():
            output = vlm_model(img, text)
        engine.last_output = output
    else:
        output = engine.last_output
```

---

## 📚 Examples with Popular VLMs

### Example 1: CLIP (Contrastive Language-Image Pre-training)

```python
import torch
from opinfer import AdaptiveMotionGater
import clip

# Load CLIP model
device = "cuda"
model, preprocess = clip.load("ViT-B/32", device=device)

# Wrap CLIP for motion gating
class CLIPWrapper(torch.nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.clip_model = clip_model
        self.text_features = None
    
    def set_text_query(self, text):
        text_tokens = clip.tokenize([text]).to(device)
        with torch.no_grad():
            self.text_features = self.clip_model.encode_text(text_tokens)
    
    def forward(self, image):
        image_features = self.clip_model.encode_image(image)
        # Compute similarity
        similarity = (100.0 * image_features @ self.text_features.T).softmax(dim=-1)
        return similarity

# Create wrapper
clip_wrapper = CLIPWrapper(model)
clip_wrapper.set_text_query("a person walking")

# Use with motion gating
gater = AdaptiveMotionGater(
    model=clip_wrapper,
    device=device,
    auto_optimize=True,
)

# Process video
frames = load_video_frames("video.mp4")
results = gater.process_video(frames, analyze_first=True)
```

### Example 2: BLIP (Bootstrapping Language-Image Pre-training)

```python
from opinfer import AdaptiveMotionGater
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch

# Load BLIP
device = "cuda"
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base"
).to(device)

# Wrap BLIP
class BLIPWrapper(torch.nn.Module):
    def __init__(self, blip_model, processor):
        super().__init__()
        self.model = blip_model
        self.processor = processor
    
    def forward(self, image):
        # BLIP expects PIL images, so preprocess accordingly
        inputs = self.processor(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = self.model.generate(**inputs)
        return outputs

# Use with motion gating
blip_wrapper = BLIPWrapper(model, processor)
gater = AdaptiveMotionGater(
    model=blip_wrapper,
    device=device,
    auto_optimize=True,
)
```

### Example 3: LLaVA (Large Language and Vision Assistant)

```python
from opinfer import AdaptiveMotionGater
from llava.model.builder import load_pretrained_model
import torch

# Load LLaVA
device = "cuda"
model_path = "liuhaotian/llava-v1.5-7b"
tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path=model_path,
    model_base=None,
    model_name="llava_v1_5",
    device_map="auto"
)

# Wrap LLaVA
class LLaVAWrapper(torch.nn.Module):
    def __init__(self, llava_model, tokenizer, image_processor):
        super().__init__()
        self.model = llava_model
        self.tokenizer = tokenizer
        self.image_processor = image_processor
    
    def forward(self, image, question="What is in this image?"):
        # Process image and text
        # Your LLaVA-specific processing
        return outputs

# Use with motion gating
llava_wrapper = LLaVAWrapper(model, tokenizer, image_processor)
gater = AdaptiveMotionGater(
    model=llava_wrapper,
    device=device,
    auto_optimize=True,
)
```

---

## 🎬 Complete VLM Workflow Example

```python
from opinfer import OptimizedInference, load_video_frames
import cv2

# Step 1: Choose your VLM
# Option A: Use built-in OWL-ViT
inf = OptimizedInference(
    model_name="owlvit-base",
    model_type="detector",
    technique="motion_gating",
    auto_optimize=True,
)

# Option B: Use custom VLM (see examples above)

# Step 2: Process video
video_path = "your_video.mp4"
results = inf.process_video_file(video_path, max_frames=500)

# Step 3: Get results
print(f"Processed {results['stats']['total_frames']} frames")
print(f"Skip rate: {results['stats']['skip_rate_pct']:.1f}%")
print(f"Effective FPS: {results['stats']['effective_fps']:.2f}")

# Step 4: Access outputs (if needed)
# For OWL-ViT, outputs contain detections with boxes, scores, labels
```

---

## 🔄 Real-Time VLM Processing

For real-time video streams:

```python
from opinfer import OptimizedInference
import cv2

# Initialize
inf = OptimizedInference(
    model_name="owlvit-base",
    model_type="detector",
    technique="motion_gating",
    auto_optimize=True,
)

# Initialize with sample frames first
cap = cv2.VideoCapture(0)  # Webcam
sample_frames = []
for _ in range(50):
    ret, frame = cap.read()
    if ret:
        sample_frames.append(frame)

# Analyze and optimize
inf.process_video_frames(sample_frames, analyze_first=True)

# Real-time processing
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Process frame
    output, stats = inf.process_frame(frame)
    
    # Use VLM output
    # For OWL-ViT: output contains detections
    # For other VLMs: output contains your model's predictions
    
    # Display or process results
    print(f"Motion: {stats['motion_score']:.2f}, Called: {stats['did_call']}")
    
    cv2.imshow("VLM Inference", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

---

## 🎯 Best Practices for VLMs

### 1. Text Query Handling

For VLMs that use text queries:

```python
# Store text features once (if possible)
text_query = "person, car, road"
text_features = encode_text(text_query)  # Pre-compute

# Use in model wrapper
class VLMWithText(torch.nn.Module):
    def __init__(self, vlm_model, text_features):
        super().__init__()
        self.model = vlm_model
        self.text_features = text_features
    
    def forward(self, image):
        return self.model(image, self.text_features)
```

### 2. Batch Processing with Queuing

For VLMs that benefit from batching:

```python
inf = OptimizedInference(
    model_name="your_vlm",
    technique="queuing",  # Use queuing for batch efficiency
    queue_size=8,
    batch_size=8,
)
```

### 3. Memory Management

```python
# For large VLMs, use smaller batch sizes
inf = OptimizedInference(
    model_name="large_vlm",
    technique="queuing",
    queue_size=2,  # Smaller queue
    batch_size=2,  # Smaller batch
)
```

---

## 📊 Supported VLM Models

### Currently Supported:
- ✅ **OWL-ViT** (google/owlvit-base-patch16, google/owlvit-large-patch14)
  - Open-vocabulary object detection
  - Works with motion gating
  - Text queries supported

### Can Be Extended:
- 🔧 **CLIP** - Image-text similarity
- 🔧 **BLIP** - Image captioning
- 🔧 **LLaVA** - Vision-language assistant
- 🔧 **Flamingo** - Few-shot learning
- 🔧 **Any custom VLM** - Wrap and use

---

## 🛠️ Integration Checklist

To use your VLM with opinfer:

- [ ] Load your VLM model
- [ ] Wrap it in a `torch.nn.Module` if needed
- [ ] Ensure it accepts image tensors (or adapt preprocessing)
- [ ] Choose technique: `motion_gating` or `queuing`
- [ ] Test with sample frames
- [ ] Process video or stream

---

## 💡 Tips

1. **Motion Gating**: Best for VLMs when you want to skip redundant frames
2. **Queuing**: Best for VLMs when you need consistent processing
3. **Text Queries**: Pre-compute text features when possible
4. **Batch Size**: Adjust based on VLM memory requirements
5. **Preprocessing**: Adapt the preprocessing function for your VLM's requirements

---

## 📝 Example: Custom VLM Integration

```python
from opinfer import AdaptiveMotionGater
import torch
import torch.nn as nn

# Your VLM
class MyVLM(nn.Module):
    def forward(self, image, text=None):
        # Your VLM logic
        return predictions

# Initialize
vlm = MyVLM().to("cuda").eval()

# Use with opinfer
gater = AdaptiveMotionGater(
    model=vlm,
    device="cuda",
    auto_optimize=True,
)

# Process
frames = load_video_frames("video.mp4")
results = gater.process_video(frames)

print(f"FPS: {results['stats']['effective_fps']:.2f}")
```

---

**That's it!** You can now use opinfer with any Vision-Language Model. The package handles the optimization, you just provide your VLM model! 🚀

