import torch

from hand_gesture_recognition.pipeline.models.neural.parallel_conv import (
    ParallelConvModule,
    ParallelConvLayer,
    FeatureExtractionModule,
)


def test_parallel_conv_layer_output_channels():
    layer = ParallelConvLayer(in_channels=8)
    x = torch.randn(4, 8, 23, 4)
    out = layer(x)
    assert out.shape == (4, 4 * 18, 23, 4)  # 4 branches x 18 = 72 channels


def test_feature_extraction_module_shape():
    m = FeatureExtractionModule(in_channels=8)
    x = torch.randn(4, 8, 23, 4)
    out = m(x)
    assert out.shape == (4, 72, 23, 4)


def test_parallel_conv_module_forward_logits():
    model = ParallelConvModule(n_channels=8, n_classes=3)
    x = torch.randn(4, 8, 23, 4)
    logits = model(x)
    assert logits.shape == (4, 3)


def test_parallel_conv_module_has_f1_metric():
    model = ParallelConvModule(n_channels=8, n_classes=3)
    assert model.val_f1 is not None
