# Custom Models

Guide for using Opinfer with custom models.

## Custom Classifier Models

To use a custom classifier model:

```python
from opinfer import AdaptiveMotionGater
import torch.nn as nn

# Your custom model
class YourClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        # Your architecture
    
    def forward(self, x):
        # x is preprocessed: (batch, 3, 224, 224), normalized
        return predictions

# Load your model
model = YourClassifier()
model.load_state_dict(torch.load("your_model.pth"))
model.eval().to("cuda")

# Use with opinfer
gater = AdaptiveMotionGater(
    model=model,
    device="cuda",
    auto_optimize=True,
)

results = gater.process_video(frames)
```

## Custom Detector Models

For detector models, you need to provide a processor function:

```python
from opinfer import AdaptiveMotionGater

# Your detector model
class YourDetector(nn.Module):
    def forward(self, images, **kwargs):
        # Process images
        return detections

# Custom processor function
def your_processor(frame_bgr, text_queries):
    # Preprocess frame and queries
    # Return inputs for your model
    pass

# Use with opinfer
model = YourDetector().eval().to("cuda")

gater = AdaptiveMotionGater(
    model=model,
    device="cuda",
    processor=your_processor,
    text_queries=["object1", "object2"],
)
```

## Model Requirements

### Classifier Models

Your model should:
- Accept input shape: `(batch, 3, 224, 224)`
- Input is normalized (ImageNet normalization)
- Return predictions (logits or probabilities)

### Detector Models

Your model should:
- Accept preprocessed images
- Return format compatible with your processor
- Support text queries if using vision-language model

## Integration Tips

1. **Preprocessing**: Ensure your model's preprocessing matches what Opinfer expects
2. **Output Format**: Make sure output format is consistent
3. **Device**: Ensure model is on the correct device (CUDA/CPU)
4. **Evaluation Mode**: Set model to `eval()` mode for inference

## Next Steps

- [API Reference](../api/intro.md)
- [Examples](../getting-started/examples.md)





