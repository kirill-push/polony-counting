from models.utils import predict


def test_predict():
    test_img = "resources/raw/test/test_img.tif"
    state_dict = "checkpoints/test.pth"
    prediction = predict(
        path=test_img,
        path_to_model=state_dict,
    )
    assert "result" in prediction[0][0].keys()
