import argparse
from predict_utils import parse_datetime_input, predict_all_runs_to_nc


def main():
    parser = argparse.ArgumentParser(description="Run SST correction for full forecast file.")
    parser.add_argument("--model", required=True, help="Path to model .pth file")
    parser.add_argument("--forecast", required=True, help="Path to forecast_structured.nc")
    parser.add_argument("--output-nc", required=True, help="Output corrected structured nc (all runs)")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for full-file prediction")
    parser.add_argument("--device", default=None, help="cuda/cpu (default from config)")
    parser.add_argument("--save-bias", action="store_true", help="Save pred_bias variable in output nc")
    parser.add_argument("--start-date", default=None, help="Inclusive start datetime for run filtering")
    parser.add_argument("--end-date", default=None, help="Inclusive end datetime for run filtering")
    args = parser.parse_args()

    start_date = parse_datetime_input(args.start_date)
    end_date = parse_datetime_input(args.end_date)
    if start_date and end_date and start_date > end_date:
        raise ValueError("--start-date cannot be later than --end-date")

    predict_all_runs_to_nc(
        model_path=args.model,
        forecast_path=args.forecast,
        output_nc=args.output_nc,
        device=args.device,
        batch_size=args.batch_size,
        save_bias=args.save_bias,
        start_date=start_date,
        end_date=end_date,
    )


if __name__ == "__main__":
    main()
