# Azure Machine Learning services Samples

`docs.microsoft.com` には、かなりの数のサンプルがあります。
幾つか、そこに無いものをこちらに置いておきます。

Azure Macine Learning services サンプルコード:
https://docs.microsoft.com/ja-jp/azure/machine-learning/service/samples-notebooks

# 事前準備

1. Azure の Subscription を作成

    Azure Machine Learning を利用するために必要です。

    無料トライアル: https://azure.microsoft.com/ja-jp/free/

2. Azure Machine Learning workspace の作成

    Jupyter Notebook で、[0.config.ipynb](0.config.ipynb) を実行します。

    Azure の Subscriptionを作成後、執筆時点 (2019/10/07)では、このコードから Workspace を作成してください。Azure Portal から Workspace を作成すると、CPU / GPU のデフォルトの `AmlCompute` が設定されないため、幾つかのサンプルコードが動作しないためです。勿論、作成した AmlCompute 名を直接クエリすれば、動作します。

    参考: Azure Machine Learning service ワークスペースを作成する:

    https://docs.microsoft.com/ja-jp/azure/machine-learning/service/setup-create-workspace

3. Jupyter Notebook から、Azure Machine Learning 参照のための設定

    Azure Portal から `構成ファイル` である `config.json` をダウンロードして、この Notebook のルート直下にUploadします。

    参考: 構成ファイルをダウンロードする

    https://docs.microsoft.com/ja-jp/azure/machine-learning/service/how-to-manage-workspace#download-a-configuration-file

# 1. MNIST HyperParameter Turning by HyperDrive using Keras with TensorFlow

 - [train-hyperparameter-tune-deploy-with-keras.ipynb](1.Hyperparameter-Turning-keras-mnist/README.md)

    Azure Machine Learning の  Hyperparameters Turning を使ったサンプル。 `HyperDrive` は、`Automated Machine Learning` の機能と切り離して利用ができます。執筆時点 (2019/7/3) 時点だと、Automated Machine Learning は、Deep Learning には使えないため、単独のサンプルとして。

# 2. Keras to ONNX for WindowsML

 - [train-hyperparameter-with-keras-for-WindowsML.ipynb](2.onnx-WindowsML/README.md)

    `WindowsML` は、Windows 10 / Windows 2019 での推論実行に特化した WindowsのAPIです。`ONNX` のみをサポートしています。
ここでは、自身で学習・作成した `keras` のモデルを、WindowsML で実行できる形式に変換します。


# 3. Text classification using AutoML with BERT featurization in Japanese

 - [auto-ml-classification-text-dnn.ipynb](3.classification-text-dnn-jpn/README.md)

    `AutoML` は Modelの学習における feature engineering, Hyper-parameter Turning, Job management などをまとめて行ってくれる機能になります。

    その中でも、テキスト・文字列 のデータがあった際に Featurization Embedding を BiLSTMあるいは 'BERT'を使って行ってくれる機能があります。


# 4. Deploy AutoML model to Azure Functions (Preview)

 - [AML-AzureFunctionsPackager.ipynb](4.AML-Functions-notebook/README.md)

    `Azure Functions` に Azure Machine Learning で管理されているモデルを Docker Container 化をしてデプロイするサンプルです。

# 5. REST API Client for AutoML Model deployment via Portal to ACI

 - [Program.cs](5.CSharp-REST-API-Client-For-AutoML-GUI-Deploy-To-ACI/README.md)

    `Auto ML` を使って作成し、ACIに展開したモデル。それを、C#から呼び出すサンプルです。

# 6. AutoML GUI deploy Probability

 - [scoring_file_v_1_0_0.py](6.AutoML-Probability/README.md)

    `Auto ML` を使って作成し、ACIに展開する際に、Probability (確度)の数字を出力するサンプルです。

# 7. HyperParameter Turning by HyperDrive using PyTorch Lightning

 - [ImageClassification-hyperparameterTune-PyTorchLightning.ipynb](7.ImageClassification-HyperparameterTurning-PyTorchLightning/README.md)

    Azure Machine Learning の  Hyperparameters Turning を使ったサンプル。PyTorch Lightning での Image Classification。Fine Tuning/Transfer Learning 前提で。

# 8. MLFlow with Azure Machine Learning

 - [train-and-deploy-keras-auto-logging.ipynb](8.%20MLFlow/README.md)

    `MLflow` は、実験のパラメーター・メトリック・成果物を記録するための OSS です。Azure Machine Learning のワークスペースは MLflow 互換なので、**既存の学習スクリプトをなるべく変更せずに**実験を追跡できます。

    ここでは `Keras` の学習を題材に、**自動ロギング (auto logging)** を使った記録とデプロイを行います。

# 9. Reinforcement Learning for Robotics on Azure ML (Hands-on)

 - [README.md](9.ReinforcementLearning/README.md)

    仮想空間上のロボットの `Pick and Place` を題材にした、**強化学習 (Reinforcement Learning) のハンズオン**です。機械学習の経験はあるが強化学習は初めて、という方を対象にしています。

    Azure Machine Learning には強化学習**専用**の機能はもうありません（`azureml-contrib-reinforcementlearning` は非推奨）。そのため、**汎用の `Command Job` / `Sweep Job` の上で OSS の強化学習ライブラリ（`Gymnasium` / `Stable-Baselines3` / `panda-gym`）を動かす**構成を取っています。

    強化学習は「同じ設定でも結果が大きくばらつく」ため、**実験の記録と比較**が普通の機械学習以上に重要になります。このハンズオンでは、`MLflow` による実験追跡、`Sweep Job` で乱数シードを振った再現性評価、対話型ジョブによるデバッグ、そして**報酬ハッキング (Reward Hacking) の検出**までを扱います。

    使用する OSS はすべて `MIT` / `zlib` ライセンス（商用利用可）です。根拠 URL は [A2_OSSライセンス一覧.md](9.ReinforcementLearning/docs/A2_OSSライセンス一覧.md) にまとめてあります。

# 10. Imitation Learning on Azure ML (Hands-on)

 - [README.md](10.ImitationLearning/README.md)

    「上手なやり方を見せて真似させる」**模倣学習 (Imitation Learning) のハンズオン**です。9 章と違い、**機械学習・強化学習の知識を前提にしません**（初級者のソフトウェア エンジニア向け）。

    強化学習が**報酬関数を設計する**のに対し、模倣学習は**その設計を回避して専門家のお手本から学びます**。題材は 9 章と同じ **Franka Emika Panda ロボットのピックアンドプレース**（`panda-gym` の `PandaPickAndPlace-v3`）で、**`Command Job` / `Sweep Job` の上で OSS の `imitation` / `Stable-Baselines3` / `Gymnasium` / `panda-gym` を動かす**構成です。GPU は不要です。

    このハンズオンの主題は「手法を成功させること」ではなく、**実験の落とし穴に気づくこと**です。**比較実験の交絡**、**単一シードでの比較**、**乱数シードを固定したつもりで固定できていない問題**、**可変ホライズン環境で評価が壊れる問題**を、いずれも実測にもとづいて扱います。

    ⚠ **ローカル実行のみ検証済みで、Azure ML 上でのジョブ実行は未検証**です。そのため所要時間・費用は記載していません。ライセンスの確認結果は [A2_OSSライセンス一覧.md](10.ImitationLearning/docs/A2_OSSライセンス一覧.md) にまとめてあります。

## 参考

Azure Machine Learning Services ドキュメント:

https://docs.microsoft.com/ja-jp/azure/machine-learning/service/
