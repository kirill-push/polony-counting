from .efficacy.polony_to_phage import efficacy_to_csv
from .models.predict_model import predict, save_predictions_to_csv


def run_counter(
    path_to_folder: str,
    path_to_predictions: str,
    path_to_efficacy: str,
    virus_type: str,
    path_to_model: str | None = None,
) -> None:
    # make predictions for all images from folder
    predictions = predict(path=path_to_folder)
    # save predictions
    save_predictions_to_csv(
        predictions=predictions,
        output_file=path_to_predictions,
        virus_type=virus_type,
        sample_volume_per_slide=5.0,
        field=2.9,
        grid=0.1,
    )
    # calculate efficacy and save to csv
    efficacy_to_csv(
        virus_type=virus_type,
        input_path=path_to_predictions,
        output_path=path_to_efficacy,
        alpha=0.05,
        n=10000,
    )
