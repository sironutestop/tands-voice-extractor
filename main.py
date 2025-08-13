import argparse
import glob
import logging
import time
import whisper


def setup_logging() -> None:
    """ログファイルの出力を設定するための関数"""
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler("debug.log", mode="a", encoding="utf-8"), logging.StreamHandler()],
    )


def main(directory: str, filename: str, output: str, debug: bool) -> None:
    """ボイスのファイルを読み込んで、文字起こしした内容をファイルに出力する関数
    Args:
        directory (str): ボイスファイルが格納されているディレクトリパス。
        filename (str): 対象となるボイスファイルの名前（ワイルドカード含む）。
        output (str): ボイスファイルを文字起こしした結果を出力するファイル名。
        debug (bool): デバッグ用のロギングを有効にするか否か。
    """

    # debug オプションを指定した際にログファイルを出力するようにする
    if debug:
        setup_logging()

    voice_files_list = glob.glob(f"{directory}/{filename}")
    file_count = len(voice_files_list)
    print(f"{file_count} 個のファイルの処理を行います")
    model = whisper.load_model("medium")

    # 進捗率を表示するためのカウンタ変数
    count = 1

    # 存在するファイル分、文字起こしを行いファイルに出力する
    with open(output, "a", encoding="utf-8") as result_file:
        for voice_file in voice_files_list:
            print(f"{file_count} 個中、 {count} 個目を処理中\n進捗率 {count/file_count}")

            if debug:
                start_time = time.time()
                logging.debug(f"{voice_file} の処理を開始({start_time})")

            result = model.transcribe(voice_file, fp16=False)

            result_file.write(f"[{voice_file}]\n\t{result['text']}\n\n")

            if debug:
                end_time = time.time()
                logging.debug(f"{voice_file} の処理を完了({end_time})\n実行時間:{end_time - start_time}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", required=True, help="音声ファイルが保存されているディレクトリを指定します")
    parser.add_argument(
        "--filename", default="LINE_*.m4a", help="音声ファイルの名前を指定します（ワイルドカードが使えます）"
    )
    parser.add_argument("--output", default="results.txt", help="文字起こしした結果を記載するファイル名を指定します")
    parser.add_argument("--debug", action="store_true", help="デバッグ用のロギングを有効化します")
    args = parser.parse_args()

    main(args.directory, args.filename, args.output, args.debug)
