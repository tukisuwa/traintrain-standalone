## このスクリプトは何？

`train_json_edit.py` は、機械学習の学習プロセスを実行する際に便利なランチャースクリプトです。
主に `train_j.py`と連携して動作します。

このスクリプトを使用すると、以下のようなことができます。

- **元の設定ファイルを変更せずに、コマンドラインから設定を一時的に変更できます。**
  例えば、学習率やバッチサイズを少しずつ変えて試したい場合、基本となるJSONファイルを複数用意する必要がありません。
- **変更した設定は、新しい一時ファイルとして保存されます。**
  これにより、どのような設定で学習を行ったか後から確認できるようになります。
- **様々な設定での学習を、より手軽に実行できるようになります。**

## 使用方法

基本的なコマンドの形は以下の通りです。

``` bash
python train_json_edit.py <もとになるJSONファイルのパス> [オプションいろいろ]
```

例:

``` bash
python train_json_edit.py configs/ADDifT_Style_SDXL.json --override train_learning_rate:0.0001
```

## 引数の説明

このスクリプトは、2種類の引数を扱います。

1.  **学習スクリプト (`train_j.py` など) に直接渡される引数**
2.  **このランチャースクリプト (`train_json_edit.py`) 自身の動作を決める引数**

順番に見ていきましょう。

### 1. 学習スクリプト (`train_j.py` など) に渡す引数

これらは、モデルの場所やチェックポイントの保存先など、学習プロセスそのものに関わる設定です。
`train_json_edit.py` はこれらの引数を解釈せず、そのまま学習スクリプトに渡します。

利用可能な引数やその詳細については、**実際に学習処理を行うスクリプト（例: `train_j.py`）のヘルプやドキュメントを参照してください。**

`train_j.py`で使用される引数：

- `--models-dir <ディレクトリパス>`
- `--ckpt-dir <ディレクトリパス>`
- `--vae-dir <ディレクトリパス>`
- `--lora-dir <ディレクトリパス>`

これらの引数は、`train_json_edit.py` を介して使用する場合も、学習スクリプトを直接実行する場合と同じように指定できます。

例:

``` bash
python train_json_edit.py configs/ADDifT_Style_SDXL.json --models-dir /path/to/models --ckpt-dir /path/to/checkpoints
```

このコマンドを実行すると、`/path/to/models` や `/path/to/checkpoints` といった情報は、このランチャーが生成した一時設定ファイルと共に、最終的に `train_j.py` へ渡されます。

### 2. ランチャースクリプト (`train_json_edit.py`) 固有の引数

ここからは、`train_json_edit.py` 自身の動作を制御するための引数です。

- **`json_path`** (必須)
  - 説明: 学習のベースとなる、元のJSON設定ファイルへのパスを指定します。
  - 例: `configs/ADDifT_Style_SDXL.json`
- **`--override KEY:VALUE [KEY:VALUE ...]`**
  - 説明: JSON設定ファイルの内容を、コマンドラインから上書きします。
    - `KEY:VALUE` のペアで指定し、複数指定も可能です。
    - `2nd pass.train_learning_rate:0.0001` のように、JSONの深い階層にあるキーも `.` (ドット) でつないで指定できます。
    - 値の型（数値、文字、True/False、リスト、辞書など）は、スクリプトが自動的に解釈します。
    - **注意:** リストや辞書、または値にスペースが含まれる場合は、コマンドラインで正しく渡すために `"` (ダブルクォーテーション) で囲むことを推奨します。
  - デフォルト: 指定がない場合、上書きは行われません。
  - 例: `--override train_batch_size:4 model:"MyCustomModel.safetensors"`
- **`--train-script-path <実行したいスクリプトのパス>`**
  - 説明: 実際に学習を行うPythonスクリプトのパスを指定します、基本的には指定不要です。
  - デフォルト: `train_j.py` (このランチャースクリプトと同じディレクトリにあると想定)
  - 例: `--train-script-path scripts/my_custom_trainer.py`
- **`--temp-config-dir <一時ファイルを保存するディレクトリのパス>`**
  - 説明: 上書きされた設定が保存される、一時的なJSONファイルの置き場所を指定します。
  - デフォルト: `temp_configs` (このランチャースクリプトと同じ場所に作成)
  - 例: `--temp-config-dir /tmp/exp_configs`
- **`--delete-temp-config`**
  - 説明: このフラグを指定すると、学習終了後に、生成された一時設定ファイルが自動的に削除されます。
  - デフォルト: 指定しない場合、一時ファイルは保持されます。

## 具体的な使い方 (例)

1.  **基本となるJSON設定を元に、学習率だけを変えて実行:**

    ``` bash
    python train_json_edit.py configs/base_settings.json --override train_learning_rate:0.0001
    python train_json_edit.py configs/base_settings.json --override train_learning_rate:0.00005
    python train_json_edit.py configs/base_settings.json --override train_learning_rate:0.00001
    ```

    このように、`base_settings.json` という一つのファイルを元に、学習率の異なる複数の実験を簡単に行えます。多数のJSONファイルを用意する必要がありません。

2.  **バッチサイズと学習ステップ数を上書きし、モデルの保存先も指定:**

    ``` bash
    python train_json_edit.py configs/base_settings.json \
        --override train_batch_size:4 \
        --override train_iterations:100 \
        --models-dir /data/trained_models \
        --ckpt-dir /data/trained_models/checkpoints
    ```

3.  **一時設定ファイルの保存場所を変更し、実験終了後に自動削除:**

    ``` bash
    python train_json_edit.py configs/base_settings.json \
        --temp-config-dir /tmp/cv_configs \
        --delete-temp-config
    ```
